# eval/quantize.py
"""
三色量化：红=食物，绿=蛇，黑=背景。

两种实现：
- quantize_pixel_level:  逐像素最近邻
- quantize_grid_level:   格子级（推荐，与 GT 生成规则一致）
"""
import numpy as np


def _norm(rgb):
    return np.array(rgb, dtype=np.float32) / 127.5 - 1.0

BG_N    = _norm((20, 20, 20))
SNAKE_N = _norm((0, 200, 0))
HEAD_N  = _norm((0, 255, 0))
FOOD_N  = _norm((255, 50, 50))

PALETTE = np.stack([BG_N, SNAKE_N, FOOD_N], axis=0)   # (3, 3)
PALETTE_NAMES = ["background", "snake", "food"]


def quantize_pixel_level(raw_frame):
    """逐像素最近邻量化"""
    img = raw_frame.transpose(1, 2, 0)
    dist = np.linalg.norm(img[:, :, None, :] - PALETTE[None, None, :, :], axis=-1)
    idx = np.argmin(dist, axis=-1)
    out = PALETTE[idx]
    return out.transpose(2, 0, 1).astype(np.float32)


def quantize_grid_level(raw_frame, grid_size=16, image_size=64):
    """格子级量化。返回 (img, grid_labels)"""
    img = raw_frame.transpose(1, 2, 0)
    cell = image_size // grid_size
    out = np.zeros_like(img)
    grid_labels = np.zeros((grid_size, grid_size), dtype=np.int64)

    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = gy * cell, (gy + 1) * cell
            x0, x1 = gx * cell, (gx + 1) * cell
            mean_color = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
            dist = np.linalg.norm(PALETTE - mean_color[None, :], axis=-1)
            idx = int(np.argmin(dist))
            grid_labels[gy, gx] = idx
            out[y0:y1, x0:x1] = PALETTE[idx]

    return out.transpose(2, 0, 1).astype(np.float32), grid_labels


def state_to_grid_labels(state, grid_size=16):
    """从游戏状态生成 GT 格子标签：0=背景, 1=蛇, 2=食物"""
    labels = np.zeros((grid_size, grid_size), dtype=np.int64)
    for y, x in state["body"]:
        labels[int(y), int(x)] = 1
    fy, fx = int(state["food"][0]), int(state["food"][1])
    labels[fy, fx] = 2
    return labels


def grid_accuracy(pred_labels, gt_labels):
    return float((pred_labels == gt_labels).mean())