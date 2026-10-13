# eval/evaluate_final.py
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_ablation import load_step, postprocess

N = 800
all_data = [load_step(i) for i in range(N)]

lengths = np.array([len(s["body"]) for _, _, _, s in all_data])
print(f"蛇长: min={lengths.min()}, max={lengths.max()}, "
      f"median={np.median(lengths)}, mean={lengths.mean():.1f}")

psnr = np.zeros((N, 2))
for cov in [0, 1]:
    g, t = [], []
    for i in range(N):
        raw, gt, _, state = all_data[i]
        gen, _ = postprocess(raw, state, cov)
        g.append(gen); t.append(gt)
    g = torch.from_numpy(np.stack(g)).float()
    t = torch.from_numpy(np.stack(t)).float()
    g = (g + 1) / 2; t = (t + 1) / 2
    mse = ((g - t) ** 2).mean(dim=[1, 2, 3]).clamp_min(1e-12)
    psnr[:, cov] = (10 * torch.log10(1.0 / mse)).numpy()

diff = psnr[:, 1] - psnr[:, 0]
print(f"\n=== cov=0 vs cov=1 ===")
print(f"cov=1 更好的样本: {(diff > 0).sum()}/{N}")
print(f"median 提升: {np.median(psnr[:,1]) - np.median(psnr[:,0]):+.3f} dB")
print(f"mean   提升: {psnr[:,1].mean() - psnr[:,0].mean():+.3f} dB")

np.random.seed(0)
boot = np.zeros((2000, 2))
for b in range(2000):
    idx = np.random.choice(N, N, replace=True)
    boot[b] = np.median(psnr[idx], axis=0)

lo, hi = np.percentile(boot[:,1] - boot[:,0], [2.5, 97.5])
print(f"\ncov1-cov0 中位数差的 95% CI: [{lo:+.3f}, {hi:+.3f}]")