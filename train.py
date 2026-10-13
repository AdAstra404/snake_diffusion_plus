import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import SnakeDataset
from models.unet import UNet
from models.edm import EDM


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs("checkpoints", exist_ok=True)

    ds = SnakeDataset(context=4)
    print(f"Original dataset size: {len(ds)}")

    # 降低样本数，避免内存/显存爆
    MAX_SAMPLES = 100000
    if len(ds) > MAX_SAMPLES:
        ds.indices = ds.indices[:MAX_SAMPLES]
    print(f"Using dataset size: {len(ds)}")

    num_workers = 0 if device == "cuda" else 0
    pin_memory = (device == "cuda")
    batch_size = 4 if device == "cuda" else 2

    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=(num_workers > 0),
    )

    unet = UNet(context_length=4)
    edm = EDM(unet, sigma_data=0.1).to(device)

    ema = copy.deepcopy(edm)
    for p in ema.parameters():
        p.requires_grad = False
    ema_decay = 0.999

    opt = torch.optim.AdamW(edm.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    n_epochs = 100

    for epoch in range(n_epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        total_loss = 0.0
        for batch in pbar:
            prev_frames = batch["prev_frames"].to(device, non_blocking=True)
            prev_actions = batch["prev_actions"].to(device, non_blocking=True)
            next_frame = batch["next_frame"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            body_mask = batch["body_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 只调用一次 forward
                loss_diff, head_pred, food_pred = edm(
                    next_frame, prev_frames, prev_actions, state, body_mask,
                    return_aux=True
                )
                if epoch >= 50:
                    loss_pos = (F.mse_loss(head_pred, state[:, :2]) +
                                F.mse_loss(food_pred, state[:, 2:4]))
                    loss = loss_diff + 0.1 * loss_pos
                else:
                    loss = loss_diff

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(edm.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                for p, ep in zip(edm.parameters(), ema.parameters()):
                    ep.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch} avg loss: {total_loss / len(dl):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(ema.state_dict(), f"checkpoints/edm_ema_epoch{epoch+1}.pth")
            torch.save(edm.state_dict(), f"checkpoints/edm_epoch{epoch+1}.pth")

    torch.save(ema.state_dict(), "checkpoints/edm_final.pth")


if __name__ == "__main__":
    train()