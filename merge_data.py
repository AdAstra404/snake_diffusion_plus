import os
import glob
import numpy as np


def merge_data(out_dir="training_data"):
    tmp_prefix = os.path.join(out_dir, "_tmp")

    # 找到所有分片
    frame_files = sorted(glob.glob(f"{tmp_prefix}_frames_*.npy"),
                         key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not frame_files:
        print("没找到临时文件")
        return

    part_files = [int(p.split("_")[-1].split(".")[0]) for p in frame_files]
    print(f"找到 {len(part_files)} 个分片: {part_files}")

    # 统计总大小
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

    frame_offset = 0
    action_offset = 0
    ep_offset = 0

    for i in part_files:
        print(f"处理分片 {i}...")
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

    frames_mm.flush()
    actions_mm.flush()
    states_mm.flush()
    masks_mm.flush()
    starts_mm.flush()
    ends_mm.flush()

    print(f"合并完成: {total_frames} frames, {total_actions} actions")
    print(f"Average steps per episode: {total_actions / total_eps:.1f}")

    # 删除临时文件
    print("清理临时文件...")
    for i in part_files:
        for suffix in ["frames", "actions", "states", "masks", "starts", "ends"]:
            os.remove(f"{tmp_prefix}_{suffix}_{i}.npy")
    print("完成！")


if __name__ == "__main__":
    merge_data()