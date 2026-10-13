# eval/run_ablation.py
"""
评测脚本（uncovered-PSNR 版）

关键设计：
- psnr_glob:   全图 PSNR（会被背景支配，仅作参考）
- psnr_roi:    ROI（蛇头 + 食物格）PSNR，cov≥2 后会 120 dB，仅作参考
- psnr_uncov:  未被本次 coverage 覆盖的前景像素上的 PSNR（论文核心指标）
- psnr_fg:     所有前景像素的 PSNR（保留用于对比）
"""
import os
import sys
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# 让同目录模块可 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_DIR, ABLATION_NPZ

# ========== 配置 ==========
N_STEPS = 800
CELL = 4
H = W = 64


# ========== 颜色 ==========
def _norm(rgb):
    return [c / 127.5 - 1.0 for c in rgb]

BG_COLOR    = _norm((20, 20, 20))
SNAKE_COLOR = _norm((0, 200, 0))
HEAD_COLOR  = _norm((0, 255, 0))
FOOD_COLOR  = _norm((255, 50, 50))


def load_step(step):
    raw = np.load(f"{TEST_DIR}/raw_gen_{step:03d}.npy")
    gt = np.load(f"{TEST_DIR}/gt_{step:03d}.npy")
    mask = np.load(f"{TEST_DIR}/body_mask_{step:03d}.npy")
    state = np.load(f"{TEST_DIR}/state_{step:03d}.npz", allow_pickle=True)
    return raw, gt, mask, state


def load_meta():
    m = np.load(f"{TEST_DIR}/meta.npz")
    return {k: m[k] for k in m.files}


# ========== 后处理：硬覆盖 ==========
def postprocess(raw, state, coverage, cell=CELL):
    img = raw.copy()
    covered = np.zeros((H, W), dtype=bool)

    def fill(y, x, color):
        y0, y1 = y * cell, (y + 1) * cell
        x0, x1 = x * cell, (x + 1) * cell
        img[:, y0:y1, x0:x1] = np.array(color, dtype=np.float32)[:, None, None]
        covered[y0:y1, x0:x1] = True

    if coverage >= 1:
        hy, hx = int(state["head"][0]), int(state["head"][1])
        fill(hy, hx, HEAD_COLOR)
    if coverage >= 2:
        fy, fx = int(state["food"][0]), int(state["food"][1])
        fill(fy, fx, FOOD_COLOR)
    if coverage >= 3:
        for b in state["body"][1:2]:
            fill(int(b[0]), int(b[1]), SNAKE_COLOR)
    if coverage >= 4:
        for b in state["body"][1:4]:
            fill(int(b[0]), int(b[1]), SNAKE_COLOR)

    return img, covered


# ========== 后处理：高斯羽化 ==========
def gaussian_mask(shape, center, radius, sigma):
    H_, W_ = shape
    yy, xx = np.mgrid[0:H_, 0:W_]
    dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    mask = np.zeros((H_, W_), dtype=np.float32)
    mask[dist <= radius] = 1.0
    transition = (dist > radius) & (dist <= radius + 2 * sigma)
    mask[transition] = np.exp(-((dist[transition] - radius) ** 2) / (2 * sigma ** 2))
    return mask


def postprocess_feathered(raw, state, coverage, cell=CELL, sigma=0.6):
    img = raw.copy()
    covered = np.zeros((H, W), dtype=bool)

    def apply_cover(y, x, color, radius_px):
        cy = y * cell + cell / 2
        cx = x * cell + cell / 2
        mask = gaussian_mask((H, W), (cy, cx), radius_px, sigma)
        color_arr = np.array(color, dtype=np.float32)[:, None, None]
        for c in range(3):
            img[c] = mask * color_arr[c] + (1 - mask) * img[c]
        y0, y1 = max(0, y * cell - int(radius_px)), min(H, (y + 1) * cell + int(radius_px))
        x0, x1 = max(0, x * cell - int(radius_px)), min(W, (x + 1) * cell + int(radius_px))
        covered[y0:y1, x0:x1] = True

    if coverage >= 1:
        hy, hx = int(state["head"][0]), int(state["head"][1])
        apply_cover(hy, hx, HEAD_COLOR, radius_px=cell * 0.6)
    if coverage >= 2:
        fy, fx = int(state["food"][0]), int(state["food"][1])
        apply_cover(fy, fx, FOOD_COLOR, radius_px=cell * 0.8)
    if coverage >= 3:
        for b in state["body"][1:2]:
            apply_cover(int(b[0]), int(b[1]), SNAKE_COLOR, radius_px=cell * 0.6)
    if coverage >= 4:
        for b in state["body"][1:4]:
            apply_cover(int(b[0]), int(b[1]), SNAKE_COLOR, radius_px=cell * 0.6)

    return img, covered


# ========== 掩码 ==========
def make_roi_strict(state):
    roi = np.zeros((H, W), dtype=np.float32)
    def mark(y, x):
        roi[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = 1.0
    mark(int(state["head"][0]), int(state["head"][1]))
    mark(int(state["food"][0]), int(state["food"][1]))
    return roi


def make_fg_mask(gt_frame):
    bg = np.array(BG_COLOR, dtype=np.float32)[:, None, None]
    diff = np.abs(gt_frame - bg).max(axis=0)
    return (diff > 1e-3).astype(np.float32)


def make_uncovered_fg_mask(gt_frame, covered_mask):
    fg = make_fg_mask(gt_frame) > 0.5
    return (fg & (~covered_mask)).astype(np.float32)


# ========== 逐样本指标 ==========
def per_sample_metrics(gen_list, gt_list, roi_list, fg_list, uncov_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    n = len(gen_list)
    out = {k: np.full(n, np.nan) for k in
           ["psnr_glob", "ssim_glob", "psnr_roi", "psnr_fg",
            "psnr_uncov", "uncov_ratio"]}

    for i, (g, t, roi, fg, uncov) in enumerate(
            zip(gen_list, gt_list, roi_list, fg_list, uncov_list)):
        g_t = torch.from_numpy(g).float().to(device)
        t_t = torch.from_numpy(t).float().to(device)
        roi_t = torch.from_numpy(roi).float().to(device)
        fg_t = torch.from_numpy(fg).float().to(device)
        uncov_t = torch.from_numpy(uncov).float().to(device)

        g_t = (g_t + 1) / 2
        t_t = (t_t + 1) / 2

        mse_g = ((g_t - t_t) ** 2).mean().clamp_min(1e-12)
        out["psnr_glob"][i] = 10 * torch.log10(1.0 / mse_g).item()

        out["ssim_glob"][i] = ssim_metric(g_t[None], t_t[None]).item()

        roi_sum = roi_t.sum().clamp_min(1.0)
        mse_r = ((g_t - t_t) ** 2 * roi_t[None]).sum() / (roi_sum * 3)
        out["psnr_roi"][i] = 10 * torch.log10(1.0 / mse_r.clamp_min(1e-12)).item()

        fg_sum = fg_t.sum().item()
        if fg_sum > 0:
            mse_f = ((g_t - t_t) ** 2 * fg_t[None]).sum() / (fg_sum * 3)
            out["psnr_fg"][i] = 10 * torch.log10(1.0 / mse_f.clamp_min(1e-12)).item()

        uncov_sum = uncov_t.sum().item()
        if uncov_sum >= 1:
            mse_u = ((g_t - t_t) ** 2 * uncov_t[None]).sum() / (uncov_sum * 3)
            out["psnr_uncov"][i] = 10 * torch.log10(1.0 / mse_u.clamp_min(1e-12)).item()
            out["uncov_ratio"][i] = uncov_sum / max(fg_sum, 1.0)

    return out


def batch_metric(gen_list, gt_list, metric="psnr"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.stack([torch.from_numpy(f) for f in gen_list]).float().to(device)
    gt  = torch.stack([torch.from_numpy(f) for f in gt_list]).float().to(device)
    gen = (gen + 1) / 2
    gt  = (gt + 1) / 2
    if metric == "psnr":
        m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    elif metric == "ssim":
        m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    else:
        m = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    return float(m(gen, gt).item())


# ========== 主评测 ==========
def evaluate(postprocess_fn, label):
    tables = {k: np.full((N_STEPS, 5), np.nan) for k in
              ["psnr_glob", "ssim_glob", "psnr_roi",
               "psnr_fg", "psnr_uncov", "uncov_ratio"]}
    batch = {k: np.full(5, np.nan) for k in
             ["psnr", "ssim", "lpips",
              "psnr_roi_mean", "psnr_fg_mean",
              "psnr_uncov_mean", "uncov_ratio_mean"]}

    for cov in range(5):
        gen_list, gt_list, roi_list, fg_list, uncov_list = [], [], [], [], []
        for i in range(N_STEPS):
            raw, gt, _, state = load_step(i)
            gen, covered = postprocess_fn(raw, state, cov)
            gen_list.append(gen)
            gt_list.append(gt)
            roi_list.append(make_roi_strict(state))
            fg_list.append(make_fg_mask(gt))
            uncov_list.append(make_uncovered_fg_mask(gt, covered))

        if not gen_list:
            continue

        m = per_sample_metrics(gen_list, gt_list, roi_list, fg_list, uncov_list)
        for k in tables:
            tables[k][:, cov] = m[k]

        batch["psnr"][cov] = batch_metric(gen_list, gt_list, "psnr")
        batch["ssim"][cov] = batch_metric(gen_list, gt_list, "ssim")
        batch["lpips"][cov] = batch_metric(gen_list, gt_list, "lpips")

        for k_src, k_dst in [("psnr_roi", "psnr_roi_mean"),
                             ("psnr_fg", "psnr_fg_mean"),
                             ("psnr_uncov", "psnr_uncov_mean"),
                             ("uncov_ratio", "uncov_ratio_mean")]:
            v = m[k_src]
            v = v[~np.isnan(v)]
            batch[k_dst][cov] = float(np.mean(v)) if len(v) else np.nan

        print(f"[{label}] cov={cov}  "
              f"PSNR={batch['psnr'][cov]:.3f}  "
              f"SSIM={batch['ssim'][cov]:.4f}  "
              f"LPIPS={batch['lpips'][cov]:.4f}  |  "
              f"ROI={batch['psnr_roi_mean'][cov]:>7.3f}  "
              f"FG={batch['psnr_fg_mean'][cov]:>7.3f}  "
              f"UNCOV={batch['psnr_uncov_mean'][cov]:>7.3f}  "
              f"(ratio={batch['uncov_ratio_mean'][cov]:.3f})")

    return tables, batch


# ========== 主流程 ==========
def main():
    meta = load_meta() if os.path.exists(f"{TEST_DIR}/meta.npz") else None

    print("=" * 100)
    print("硬覆盖")
    print("=" * 100)
    hard_t, hard_b = evaluate(postprocess, "硬覆盖")

    print("\n" + "=" * 100)
    print("高斯羽化 σ=0.6")
    print("=" * 100)
    feat_t, feat_b = evaluate(postprocess_feathered, "羽化")

    print("\n" + "=" * 100)
    print("核心曲线")
    print("=" * 100)
    print(f"{'cov':>4} | "
          f"{'硬-PSNR':>9} {'硬-FG':>9} {'硬-UNCOV':>10} {'硬-ratio':>9} | "
          f"{'羽-PSNR':>9} {'羽-FG':>9} {'羽-UNCOV':>10} {'羽-ratio':>9}")
    for cov in range(5):
        print(f"{cov:>4} | "
              f"{hard_b['psnr'][cov]:>9.3f} "
              f"{hard_b['psnr_fg_mean'][cov]:>9.3f} "
              f"{hard_b['psnr_uncov_mean'][cov]:>10.3f} "
              f"{hard_b['uncov_ratio_mean'][cov]:>9.3f} | "
              f"{feat_b['psnr'][cov]:>9.3f} "
              f"{feat_b['psnr_fg_mean'][cov]:>9.3f} "
              f"{feat_b['psnr_uncov_mean'][cov]:>10.3f} "
              f"{feat_b['uncov_ratio_mean'][cov]:>9.3f}")

    print("\n" + "=" * 100)
    print("拐点检测（uncovered-PSNR）")
    print("=" * 100)
    for label, arr in [("硬覆盖-UNCOV", hard_b["psnr_uncov_mean"]),
                       ("羽化-UNCOV",   feat_b["psnr_uncov_mean"])]:
        peak = int(np.nanargmax(arr))
        print(f"  {label}: 峰值 cov={peak}  "
              f"值={arr[peak]:.3f}  "
              f"cov0={arr[0]:.3f}  Δ={arr[peak]-arr[0]:+.3f}")

    if meta is not None:
        lengths = meta["snake_len"]
        med = np.median(lengths)
        print("\n" + "=" * 100)
        print(f"分组：按蛇长 > 中位数 ({med:.0f})")
        print("=" * 100)
        short_mask = lengths <= med
        long_mask = lengths > med
        for name, mask in [("短蛇", short_mask), ("长蛇", long_mask)]:
            print(f"\n{name} (n={mask.sum()}):")
            for cov in range(5):
                arr = hard_t["psnr_uncov"][mask, cov]
                arr = arr[~np.isnan(arr)]
                if len(arr):
                    print(f"  cov={cov}: UNCOV median={np.median(arr):>7.3f}  "
                          f"mean={arr.mean():>7.3f}  n={len(arr)}")

    np.savez(ABLATION_NPZ,
             hard_psnr_glob=hard_t["psnr_glob"],
             hard_ssim_glob=hard_t["ssim_glob"],
             hard_psnr_roi=hard_t["psnr_roi"],
             hard_psnr_fg=hard_t["psnr_fg"],
             hard_psnr_uncov=hard_t["psnr_uncov"],
             hard_uncov_ratio=hard_t["uncov_ratio"],
             feather_psnr_glob=feat_t["psnr_glob"],
             feather_ssim_glob=feat_t["ssim_glob"],
             feather_psnr_roi=feat_t["psnr_roi"],
             feather_psnr_fg=feat_t["psnr_fg"],
             feather_psnr_uncov=feat_t["psnr_uncov"],
             feather_uncov_ratio=feat_t["uncov_ratio"],
             hard_batch_psnr=hard_b["psnr"],
             hard_batch_ssim=hard_b["ssim"],
             hard_batch_lpips=hard_b["lpips"],
             hard_batch_psnr_uncov=hard_b["psnr_uncov_mean"],
             hard_batch_uncov_ratio=hard_b["uncov_ratio_mean"],
             feather_batch_psnr=feat_b["psnr"],
             feather_batch_ssim=feat_b["ssim"],
             feather_batch_lpips=feat_b["lpips"],
             feather_batch_psnr_uncov=feat_b["psnr_uncov_mean"],
             feather_batch_uncov_ratio=feat_b["uncov_ratio_mean"])
    print(f"\nSaved to {ABLATION_NPZ}")


if __name__ == "__main__":
    main()