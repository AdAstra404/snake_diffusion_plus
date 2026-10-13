# eval/check_per_sample.py
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import PSNR_TABLE_NPY
from run_ablation import (
    load_step, postprocess, load_meta,
    make_fg_mask, N_STEPS as N,
)
import run_ablation

H = W = 64
np.random.seed(0)
N_BOOT = 2000

print(f"Loading {N} samples...")
all_data = [load_step(i) for i in range(N)]
meta = load_meta()


def compute_fg_weighted_psnr(gen, gt, fg):
    g = torch.from_numpy(gen).float()
    t = torch.from_numpy(gt).float()
    f = torch.from_numpy(fg).float()
    g = (g + 1) / 2
    t = (t + 1) / 2
    mse = ((g - t) ** 2 * f[None]).sum() / (H * W * 3)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-12)).item()


psnr_table = np.zeros((N, 5))
for cov in range(5):
    for i in range(N):
        raw, gt, _, state = load_step(i)
        gen, _ = postprocess(raw, state, cov)
        fg = make_fg_mask(gt)
        psnr_table[i, cov] = compute_fg_weighted_psnr(gen, gt, fg)
    print(f"  cov={cov} done")

np.save(PSNR_TABLE_NPY, psnr_table)

print("\n" + "=" * 70)
print("主表：FG 加权 PSNR（中位数 / IQR）")
print("=" * 70)
print(f"{'cov':>4} | {'median':>9} {'Q1':>9} {'Q3':>9} {'mean':>9}")
for cov in range(5):
    a = psnr_table[:, cov]
    q1, q3 = np.percentile(a, [25, 75])
    print(f"{cov:>4} | {np.median(a):>9.3f} {q1:>9.3f} {q3:>9.3f} {a.mean():>9.3f}")

print("\n" + "=" * 70)
print("逐样本最优覆盖率分布")
print("=" * 70)
best = np.argmax(psnr_table, axis=1)
for cov in range(5):
    cnt = (best == cov).sum()
    print(f"  最优为 cov={cov}: {cnt}/{N} ({cnt/N*100:.1f}%)")

print("\n" + "=" * 70)
print("分层分析：按蛇长分组")
print("=" * 70)
lengths = meta["snake_len"]
q33, q67 = np.percentile(lengths, [33, 67])
groups = {
    f"短蛇 (len ≤ {int(q33)})":  lengths <= q33,
    f"中蛇 ({int(q33)} < len ≤ {int(q67)})": (lengths > q33) & (lengths <= q67),
    f"长蛇 (len > {int(q67)})":  lengths > q67,
}
for name, mask in groups.items():
    n = mask.sum()
    print(f"\n{name}  n={n}")
    base = np.median(psnr_table[mask, 0])
    for cov in range(5):
        med = np.median(psnr_table[mask, cov])
        print(f"  cov={cov}: median={med:>8.3f}  Δ={med - base:>+7.3f}")

print("\n" + "=" * 70)
print(f"Bootstrap 中位数 95% CI (n={N_BOOT})")
print("=" * 70)
boot = np.zeros((N_BOOT, 5))
for b in range(N_BOOT):
    idx = np.random.choice(N, N, replace=True)
    boot[b] = np.median(psnr_table[idx], axis=0)
for cov in range(5):
    lo, hi = np.percentile(boot[:, cov], [2.5, 97.5])
    print(f"  cov={cov}: median={np.median(psnr_table[:,cov]):.3f}  "
          f"95% CI=[{lo:.3f}, {hi:.3f}]")

print("\n" + "=" * 70)
print("成对显著性（Bootstrap）")
print("=" * 70)
for cov in [0, 2, 3, 4]:
    diff = boot[:, 1] - boot[:, cov]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    p = (diff <= 0).mean()
    print(f"  cov1 - cov{cov}: Δ={np.median(psnr_table[:,1])-np.median(psnr_table[:,cov]):+.3f}  "
          f"95% CI=[{lo:.3f}, {hi:.3f}]  p={p:.4f}")

print("\n" + "=" * 70)
print("Markdown 表")
print("=" * 70)
print("| Coverage | Median FG-PSNR | IQR | Mean |")
print("|---|---|---|---|")
for cov in range(5):
    a = psnr_table[:, cov]
    q1, q3 = np.percentile(a, [25, 75])
    print(f"| {cov} | {np.median(a):.2f} | [{q1:.2f}, {q3:.2f}] | {a.mean():.2f} |")