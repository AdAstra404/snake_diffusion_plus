# eval/run_ablation_quantize.py
"""
三色量化补充实验。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import ABLATION_QUANTIZE_NPZ
from run_ablation import (
    load_step, load_meta, postprocess,
    make_fg_mask, make_roi_strict, make_uncovered_fg_mask,
    per_sample_metrics, batch_metric,
    N_STEPS, H, W, CELL,
)
from quantize import (
    quantize_grid_level, quantize_pixel_level, PALETTE,
    state_to_grid_labels, grid_accuracy,
)


def postprocess_quantize(raw, state, coverage=0):
    img, _ = quantize_grid_level(raw)
    covered = np.zeros((H, W), dtype=bool)
    return img, covered


def postprocess_hard_then_quantize(raw, state, coverage=4):
    img, _ = postprocess(raw, state, coverage)
    img, _ = quantize_grid_level(img)
    covered = np.zeros((H, W), dtype=bool)
    return img, covered


def evaluate_one(postprocess_fn, label, coverage=0, with_grid_acc=False):
    gen_list, gt_list, roi_list, fg_list, uncov_list = [], [], [], [], []
    grid_accs = []

    for i in range(N_STEPS):
        raw, gt, _, state = load_step(i)
        gen, covered = postprocess_fn(raw, state, coverage)
        gen_list.append(gen)
        gt_list.append(gt)
        roi_list.append(make_roi_strict(state))
        fg_list.append(make_fg_mask(gt))
        uncov_list.append(make_uncovered_fg_mask(gt, covered))

        if with_grid_acc:
            img = gen.transpose(1, 2, 0)
            pred = np.zeros((16, 16), dtype=np.int64)
            for gy in range(16):
                for gx in range(16):
                    y0, y1 = gy * CELL, (gy + 1) * CELL
                    x0, x1 = gx * CELL, (gx + 1) * CELL
                    mean_color = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
                    dist = np.linalg.norm(PALETTE - mean_color[None, :], axis=-1)
                    pred[gy, gx] = int(np.argmin(dist))
            gt_labels = state_to_grid_labels(state, 16)
            grid_accs.append(grid_accuracy(pred, gt_labels))

    m = per_sample_metrics(gen_list, gt_list, roi_list, fg_list, uncov_list)

    result = {
        "psnr_glob": batch_metric(gen_list, gt_list, "psnr"),
        "ssim_glob": batch_metric(gen_list, gt_list, "ssim"),
        "lpips_glob": batch_metric(gen_list, gt_list, "lpips"),
        "psnr_uncov_mean": float(np.nanmean(m["psnr_uncov"])),
        "psnr_fg_mean": float(np.nanmean(m["psnr_fg"])),
        "psnr_roi_mean": float(np.nanmean(m["psnr_roi"])),
        "uncov_ratio_mean": float(np.nanmean(m["uncov_ratio"])),
        "psnr_uncov_per_sample": m["psnr_uncov"],
    }
    if with_grid_acc:
        result["grid_acc_mean"] = float(np.mean(grid_accs))
        result["grid_acc_per_sample"] = np.array(grid_accs)

    print(f"[{label}] "
          f"PSNR={result['psnr_glob']:.3f}  "
          f"SSIM={result['ssim_glob']:.4f}  "
          f"LPIPS={result['lpips_glob']:.4f}  "
          f"UNCOV={result['psnr_uncov_mean']:.3f}"
          + (f"  GridAcc={result['grid_acc_mean']:.4f}" if with_grid_acc else ""))
    return result


def main():
    print("=" * 100)
    print("三色量化补充实验")
    print("=" * 100)

    conditions = {
        "C0_无约束":        (lambda raw, s: postprocess(raw, s, 0),  False),
        "C1_硬覆盖头":      (lambda raw, s: postprocess(raw, s, 1),  False),
        "C2_硬覆盖cov4":    (lambda raw, s: postprocess(raw, s, 4),  True),
        "C3_三色量化":      (lambda raw, s: postprocess_quantize(raw, s), True),
        "C4_硬覆盖+量化":   (lambda raw, s: postprocess_hard_then_quantize(raw, s, 4), True),
    }

    results = {}
    for name, (fn, use_acc) in conditions.items():
        results[name] = evaluate_one(
            lambda raw, s, cov=0, f=fn: f(raw, s),
            name, coverage=0, with_grid_acc=use_acc,
        )

    print("\n" + "=" * 100)
    print("汇总表")
    print("=" * 100)
    print(f"{'条件':<18} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9} "
          f"{'UNCOV':>9} {'GridAcc':>9}")
    for name, r in results.items():
        ga = f"{r['grid_acc_mean']:.4f}" if "grid_acc_mean" in r else "  —"
        print(f"{name:<18} {r['psnr_glob']:>9.3f} {r['ssim_glob']:>9.4f} "
              f"{r['lpips_glob']:>9.4f} {r['psnr_uncov_mean']:>9.3f} {ga:>9}")

    print("\n" + "=" * 100)
    print("关键对比")
    print("=" * 100)
    pairs = [
        ("C1_硬覆盖头", "C3_三色量化", "只修位置 vs 只修颜色"),
        ("C0_无约束", "C3_三色量化", "纯颜色量化的效果"),
        ("C2_硬覆盖cov4", "C4_硬覆盖+量化", "加量化是否有额外收益"),
        ("C3_三色量化", "C4_硬覆盖+量化", "加位置修正是否有额外收益"),
    ]
    for a, b, desc in pairs:
        d_psnr = results[b]["psnr_glob"] - results[a]["psnr_glob"]
        d_lpips = results[b]["lpips_glob"] - results[a]["lpips_glob"]
        print(f"{desc:<30} ΔPSNR={d_psnr:+.3f}  ΔLPIPS={d_lpips:+.4f}")

    meta = load_meta()
    lengths = meta["snake_len"]
    med = np.median(lengths)
    short_mask = lengths <= med
    long_mask = lengths > med

    print("\n" + "=" * 100)
    print(f"分组：按蛇长 > 中位数 ({med:.0f})，指标为 uncovered-PSNR 中位数")
    print("=" * 100)
    print(f"{'条件':<18} {'短蛇 median':>14} {'长蛇 median':>14}")
    for name, r in results.items():
        per = r["psnr_uncov_per_sample"]
        s_med = np.nanmedian(per[short_mask]) if np.isfinite(per[short_mask]).any() else np.nan
        l_med = np.nanmedian(per[long_mask]) if np.isfinite(per[long_mask]).any() else np.nan
        print(f"{name:<18} {s_med:>14.3f} {l_med:>14.3f}")

    save_dict = {}
    for name, r in results.items():
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                save_dict[f"{name}_{k}"] = v
            else:
                save_dict[f"{name}_{k}"] = np.array(v)
    np.savez(ABLATION_QUANTIZE_NPZ, **save_dict)
    print(f"\nSaved to {ABLATION_QUANTIZE_NPZ}")


if __name__ == "__main__":
    main()