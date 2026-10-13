import os
import numpy as np
from tqdm import tqdm
from .game import SnakeGame, ACTIONS
from .render import Renderer


def safe_action(game):
    valid = []
    for a in range(4):
        if (a + 2) % 4 == game.direction:
            continue
        dy, dx = ACTIONS[a]
        hy, hx = game.snake[0]
        ny, nx = hy + dy, hx + dx
        if not (0 <= ny < game.grid_size and 0 <= nx < game.grid_size):
            continue
        if (ny, nx) in game.snake[:-1]:
            continue
        valid.append(a)
    return int(np.random.choice(valid)) if valid else np.random.randint(4)


def render_body_mask(body, grid_size=16, image_size=64):
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    cell = image_size // grid_size
    for y, x in body:
        mask[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = 1.0
    return mask


def generate_dataset(out_dir="training_data", n_episodes=50000,
                     max_steps=300, context=4, flush_every=2000):
    """边生成边写磁盘，避免内存爆掉"""
    os.makedirs(out_dir, exist_ok=True)
    renderer = Renderer()

    # 临时文件前缀
    tmp_prefix = os.path.join(out_dir, "_tmp")

    # 分批缓冲区
    buf_frames = []
    buf_actions = []
    buf_states = []
    buf_masks = []
    buf_starts = []
    buf_ends = []

    # 已写入的累计计数
    total_actions = 0
    part_idx = 0
    part_files = []   # 记录所有临时文件名

    def flush():
        """把缓冲区写入磁盘，并清空"""
        nonlocal buf_frames, buf_actions, buf_states, buf_masks
        nonlocal buf_starts, buf_ends, part_idx, part_files, total_actions

        if len(buf_actions) == 0:
            return

        # 调整 episode_starts / ends 为全局索引
        starts = np.array(buf_starts) + (total_actions - len(buf_actions))
        ends = np.array(buf_ends) + (total_actions - len(buf_actions))

        np.save(f"{tmp_prefix}_frames_{part_idx}.npy", np.array(buf_frames))
        np.save(f"{tmp_prefix}_actions_{part_idx}.npy", np.array(buf_actions))
        np.save(f"{tmp_prefix}_states_{part_idx}.npy", np.array(buf_states))
        np.save(f"{tmp_prefix}_masks_{part_idx}.npy", np.array(buf_masks))
        np.save(f"{tmp_prefix}_starts_{part_idx}.npy", starts)
        np.save(f"{tmp_prefix}_ends_{part_idx}.npy", ends)

        part_files.append(part_idx)
        part_idx += 1

        # 清空缓冲
        buf_frames = []
        buf_actions = []
        buf_states = []
        buf_masks = []
        buf_starts = []
        buf_ends = []

    for ep in tqdm(range(n_episodes)):
        game = SnakeGame()
        state = game.reset()
        buf_starts.append(len(buf_actions))

        for t in range(max_steps):
            if state["done"]:
                break

            buf_frames.append(renderer.render(state))
            buf_masks.append(render_body_mask(state["body"]))

            action = safe_action(game)
            next_state, reward, done = game.step(action)

            buf_actions.append(action)
            buf_states.append(state)
            state = next_state

            if done:
                break

        buf_ends.append(len(buf_actions))

        # 每 flush_every 局写一次磁盘
        if (ep + 1) % flush_every == 0:
            total_actions += len(buf_actions)
            flush()

    # 写最后一批
    total_actions += len(buf_actions)
    flush()

    # 合并所有临时文件（用 memmap 逐片写入，避免内存爆）
    print("Merging temporary files...")

    # 先统计总大小
    total_frames = 0
    total_actions = 0
    total_eps = 0
    for i in part_files:
        total_frames += np.load(f"{tmp_prefix}_frames_{i}.npy", mmap_mode='r').shape[0]
        total_actions += np.load(f"{tmp_prefix}_actions_{i}.npy", mmap_mode='r').shape[0]
        total_eps += len(np.load(f"{tmp_prefix}_starts_{i}.npy"))

    print(f"Total frames: {total_frames}, actions: {total_actions}, episodes: {total_eps}")

    # 预分配 memmap
    frames_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "frames.npy"), mode='w+',
        dtype=np.float32, shape=(total_frames, 3, 64, 64)
    )
    actions_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "actions.npy"), mode='w+',
        dtype=np.int64, shape=(total_actions,)
    )
    states_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "states.npy"), mode='w+',
        dtype=np.float32, shape=(total_actions, 10)
    )
    masks_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "body_masks.npy"), mode='w+',
        dtype=np.float32, shape=(total_frames, 64, 64)
    )
    starts_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "episode_starts.npy"), mode='w+',
        dtype=np.int64, shape=(total_eps,)
    )
    ends_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, "episode_ends.npy"), mode='w+',
        dtype=np.int64, shape=(total_eps,)
    )

    # 逐片写入
    frame_offset = 0
    action_offset = 0
    ep_offset = 0

    for i in part_files:
        f = np.load(f"{tmp_prefix}_frames_{i}.npy", mmap_mode='r')
        a = np.load(f"{tmp_prefix}_actions_{i}.npy", mmap_mode='r')
        s = np.load(f"{tmp_prefix}_states_{i}.npy", allow_pickle=True)
        m = np.load(f"{tmp_prefix}_masks_{i}.npy", mmap_mode='r')
        st = np.load(f"{tmp_prefix}_starts_{i}.npy")
        en = np.load(f"{tmp_prefix}_ends_{i}.npy")

        n_frames = f.shape[0]
        n_actions = a.shape[0]
        n_eps = len(st)

        frames_mm[frame_offset:frame_offset + n_frames] = f
        masks_mm[frame_offset:frame_offset + n_frames] = m
        actions_mm[action_offset:action_offset + n_actions] = a

        # states 需要从 dict 转成 10 维数组
        states_arr = np.array([
            [x["head"][0], x["head"][1], x["food"][0], x["food"][1],
             x["direction"], x["score"], int(x["done"]),
             len(x["body"]), x["body"][-1][0], x["body"][-1][1]]
            for x in s
        ], dtype=np.float32)
        states_mm[action_offset:action_offset + n_actions] = states_arr

        starts_mm[ep_offset:ep_offset + n_eps] = st
        ends_mm[ep_offset:ep_offset + n_eps] = en

        frame_offset += n_frames
        action_offset += n_actions
        ep_offset += n_eps

    # flush 到磁盘
    frames_mm.flush()
    actions_mm.flush()
    states_mm.flush()
    masks_mm.flush()
    starts_mm.flush()
    ends_mm.flush()

    # 删除临时文件
    print("Cleaning up temporary files...")
    for i in part_files:
        for suffix in ["frames", "actions", "states", "masks", "starts", "ends"]:
            os.remove(f"{tmp_prefix}_{suffix}_{i}.npy")

    print(f"Saved {total_frames} frames, {total_actions} actions")
    print(f"Average steps per episode: {total_actions / n_episodes:.1f}")


if __name__ == "__main__":
    generate_dataset()