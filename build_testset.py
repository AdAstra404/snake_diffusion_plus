# build_testset.py
"""
长蛇测试集生成脚本（多局版）。

新增：
- EPISODE_MAX_STEPS：每局最多跑多少步，超了就强制重置
  这样能保证样本来自多局游戏，覆盖不同蛇长
"""
import os
import sys
from collections import deque

import numpy as np
import torch

from models.unet import UNet
from models.edm import EDM
from data.game import SnakeGame
from data.render import Renderer
from play import build_inputs, render_body_mask

# ========== 配置 ==========
SEED = 42
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
EPISODE_MAX_STEPS = 120          # ← 新增：每局最多 60 步
SAVE_DIR = "testset"
os.makedirs(SAVE_DIR, exist_ok=True)

rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"N_STEPS = {N_STEPS}, EPISODE_MAX_STEPS = {EPISODE_MAX_STEPS}")

# ========== 加载模型 ==========
unet = UNet(context_length=4)
edm = EDM(unet, sigma_data=0.1).to(device)
edm.load_state_dict(torch.load("checkpoints/edm_final.pth", map_location=device))
edm.eval()

game = SnakeGame()
renderer = Renderer()
state = game.reset()
frames, actions = [], []

ACTIONS_DYDX = [(0, 1), (0, -1), (-1, 0), (1, 0)]


# ========== BFS 安全检查 ==========
def is_safe_after_move(game, action):
    if (action ^ 1) == game.direction:
        return False
    dy, dx = ACTIONS_DYDX[action]
    hy, hx = game.snake[0]
    ny, nx = hy + dy, hx + dx
    if not (0 <= ny < game.grid_size and 0 <= nx < game.grid_size):
        return False
    if (ny, nx) in game.snake[:-1]:
        return False

    will_eat = (ny, nx) == game.food
    if will_eat:
        new_snake = [(ny, nx)] + game.snake
        new_food = None
    else:
        new_snake = [(ny, nx)] + game.snake[:-1]
        new_food = game.food

    obstacles = set(new_snake[1:])
    start = (ny, nx)
    q = deque([start])
    visited = {start}
    while q:
        y, x = q.popleft()
        if new_food is not None and (y, x) == new_food:
            return True
        for ddy, ddx in ACTIONS_DYDX:
            cy, cx = y + ddy, x + ddx
            if (0 <= cy < game.grid_size and 0 <= cx < game.grid_size
                    and (cy, cx) not in obstacles
                    and (cy, cx) not in visited):
                visited.add((cy, cx))
                q.append((cy, cx))
    if new_food is None:
        return len(visited) >= len(new_snake) + 1
    return False


def greedy_policy(game):
    hy, hx = game.snake[0]
    fy, fx = game.food
    legal = []
    for a in range(4):
        if (a ^ 1) == game.direction:
            continue
        dy, dx = ACTIONS_DYDX[a]
        ny, nx = hy + dy, hx + dx
        if not (0 <= ny < game.grid_size and 0 <= nx < game.grid_size):
            continue
        if (ny, nx) in game.snake[:-1]:
            continue
        dist = abs(ny - fy) + abs(nx - fx)
        legal.append((dist, a))
    if not legal:
        return int(rng.randint(4))
    legal.sort()
    for _, a in legal:
        if is_safe_after_move(game, a):
            return a
    return legal[0][1]


# ========== 主循环 ==========
step = 0
episode = 0
episode_step = 0        # ← 当前局内的步数
episode_lengths = []    # ← 每局结束时的蛇长
meta_records = []

while step < N_STEPS:
    # ---------- 重置条件：done 或 超过 EPISODE_MAX_STEPS ----------
    if state["done"] or episode_step >= EPISODE_MAX_STEPS:
        episode_lengths.append(len(state["body"]))
        episode += 1
        print(f"  [Episode {episode} 结束] score={state['score']}  "
              f"snake_len={len(state['body'])}  steps={episode_step}")
        state = game.reset()
        frames, actions = [], []
        episode_step = 0
        continue

    prev_frames_t, prev_actions_t, state_t, body_mask_t = build_inputs(
        frames, actions, state, 4, device
    )

    action = greedy_policy(game)
    next_state, _, done = game.step(action)
    ate_food = (next_state["score"] > state["score"])

    with torch.inference_mode():
        raw_gen = edm.sample(50, (3, 64, 64), prev_frames_t, prev_actions_t,
                             state_t, body_mask_t)

    np.save(f"{SAVE_DIR}/raw_gen_{step:03d}.npy", raw_gen[0].cpu().numpy())
    np.save(f"{SAVE_DIR}/gt_{step:03d}.npy", renderer.render(next_state))
    np.save(f"{SAVE_DIR}/body_mask_{step:03d}.npy",
            render_body_mask(next_state["body"]))
    np.savez(f"{SAVE_DIR}/state_{step:03d}.npz",
             head=np.array(next_state["head"], dtype=np.int64),
             food=np.array(next_state["food"], dtype=np.int64),
             body=np.array(next_state["body"], dtype=np.int64),
             direction=np.int64(next_state["direction"]),
             score=np.int64(next_state["score"]),
             done=np.bool_(next_state["done"]))

    meta_records.append({
        "step": step,
        "episode": episode,
        "episode_step": episode_step,
        "action": action,
        "score": int(next_state["score"]),
        "done": bool(next_state["done"]),
        "ate_food": bool(ate_food),
        "snake_len": len(next_state["body"]),
    })

    actions.append(action)
    state = next_state
    frames.append(renderer.render(next_state))

    step += 1
    episode_step += 1
    if step % 50 == 0:
        print(f"Step {step}/{N_STEPS}  episode={episode}  "
              f"score={state['score']}  snake_len={len(state['body'])}")

episode_lengths.append(len(state["body"]))

# 保存 meta
np.savez(f"{SAVE_DIR}/meta.npz",
         step=np.array([m["step"] for m in meta_records]),
         episode=np.array([m["episode"] for m in meta_records]),
         episode_step=np.array([m["episode_step"] for m in meta_records]),
         action=np.array([m["action"] for m in meta_records]),
         score=np.array([m["score"] for m in meta_records]),
         done=np.array([m["done"] for m in meta_records]),
         ate_food=np.array([m["ate_food"] for m in meta_records]),
         snake_len=np.array([m["snake_len"] for m in meta_records]))

# 统计
all_lengths = []
for i in range(N_STEPS):
    s = np.load(f"{SAVE_DIR}/state_{i:03d}.npz", allow_pickle=True)
    all_lengths.append(len(s["body"]))
all_lengths = np.array(all_lengths)

print("\n" + "=" * 60)
print("生成完成")
print("=" * 60)
print(f"总步数: {N_STEPS}")
print(f"总 episode 数: {episode + 1}")
print(f"\n所有 {N_STEPS} 步的蛇长分布:")
print(f"  min={all_lengths.min()}, max={all_lengths.max()}, "
      f"median={np.median(all_lengths):.1f}, mean={all_lengths.mean():.1f}")
print(f"  蛇长 ≤ 5 的样本:  {(all_lengths <= 5).sum()}/{N_STEPS} "
      f"({(all_lengths <= 5).mean()*100:.1f}%)")
print(f"  蛇长 ≥ 10 的样本: {(all_lengths >= 10).sum()}/{N_STEPS} "
      f"({(all_lengths >= 10).mean()*100:.1f}%)")
print(f"  蛇长 ≥ 15 的样本: {(all_lengths >= 15).sum()}/{N_STEPS} "
      f"({(all_lengths >= 15).mean()*100:.1f}%)")
print(f"\n每局结束时的蛇长:")
print(f"  {episode_lengths}")
print(f"\n保存目录: {SAVE_DIR}/")