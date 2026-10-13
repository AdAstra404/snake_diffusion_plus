import numpy as np
import torch
from torch.utils.data import Dataset


class SnakeDataset(Dataset):
    def __init__(self, data_dir="training_data", context=4):
        self.frames = np.load(f"{data_dir}/frames.npy")
        self.actions = np.load(f"{data_dir}/actions.npy")
        self.states = np.load(f"{data_dir}/states.npy")
        self.body_masks = np.load(f"{data_dir}/body_masks.npy")
        self.episode_starts = np.load(f"{data_dir}/episode_starts.npy")
        self.episode_ends = np.load(f"{data_dir}/episode_ends.npy")
        self.context = context

        self.indices = []
        for start, end in zip(self.episode_starts, self.episode_ends):
            for i in range(start + context - 1, end - 1):
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        c = self.context

        prev_frames = torch.from_numpy(self.frames[i - c + 1: i + 1]).float()
        prev_actions = torch.from_numpy(self.actions[i - c + 1: i + 1]).long()
        next_frame = torch.from_numpy(self.frames[i + 1]).float()
        state = torch.from_numpy(self.states[i]).float()          # (10,)
        body_mask = torch.from_numpy(self.body_masks[i]).float()  # (64, 64)

        return {
            "prev_frames": prev_frames,
            "prev_actions": prev_actions,
            "next_frame": next_frame,
            "state": state,
            "body_mask": body_mask,
        }