import torch
import numpy as np
from PIL import Image
from models.unet import UNet
from models.edm import EDM
from data.game import SnakeGame
from data.render import Renderer


def render_body_mask(body, grid_size=16, image_size=64):
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    cell = image_size // grid_size
    for y, x in body:
        mask[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = 1.0
    return mask


def postprocess(gen_frame, state, cell=4):
    """用真实坐标强制覆盖食物和蛇头"""
    img = gen_frame.copy()
    fy, fx = int(state["food"][0]), int(state["food"][1])
    img[:, fy * cell:(fy + 1) * cell, fx * cell:(fx + 1) * cell] = \
        np.array([1.0, -0.6, -0.6])[:, None, None]
    hy, hx = int(state["head"][0]), int(state["head"][1])
    img[:, hy * cell:(hy + 1) * cell, hx * cell:(hx + 1) * cell] = \
        np.array([-1.0, 1.0, -1.0])[:, None, None]
    return img


def build_inputs(frames, actions, state, context, device):
    """构造模型输入，状态 10 维，不归一化"""
    c = context
    if len(frames) == 0:
        prev_frames = np.zeros((c, 3, 64, 64), dtype=np.float32)
        prev_actions = np.zeros(c, dtype=np.int64)
    elif len(frames) < c:
        pad = [frames[0]] * (c - len(frames))
        prev_frames = np.stack(pad + frames)
        prev_actions = np.zeros(c, dtype=np.int64)
    else:
        prev_frames = np.stack(frames[-c:])
        act_hist = actions[-c:] if len(actions) >= c else actions
        pad_n = c - len(act_hist)
        prev_actions = np.array(list(act_hist) + [0] * pad_n, dtype=np.int64)

    prev_frames_t = torch.from_numpy(prev_frames).float().unsqueeze(0).to(device)
    prev_actions_t = torch.from_numpy(prev_actions).long().unsqueeze(0).to(device)

    # 10 维状态，与训练一致（不归一化）
    state_vec = [
        state["head"][0], state["head"][1],
        state["food"][0], state["food"][1],
        state["direction"], state["score"], float(state["done"]),
        len(state["body"]),
        state["body"][-1][0], state["body"][-1][1],
    ]
    state_t = torch.tensor([state_vec], dtype=torch.float32).to(device)
    body_mask_t = torch.from_numpy(render_body_mask(state["body"])).float().unsqueeze(0).to(device)
    return prev_frames_t, prev_actions_t, state_t, body_mask_t


def generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet(context_length=4)
    edm = EDM(unet, sigma_data=0.1).to(device)
    edm.load_state_dict(torch.load("checkpoints/edm_final.pth", map_location=device))
    edm.eval()

    game = SnakeGame()
    renderer = Renderer()
    state = game.reset()
    frames, actions = [], []

    for step in range(100):
        frames.append(renderer.render(state))

        if state["done"]:
            print("Game over!")
            break

        prev_frames_t, prev_actions_t, state_t, body_mask_t = build_inputs(
            frames, actions, state, 4, device
        )

        action = np.random.randint(4)
        next_state, _, done = game.step(action)

        with torch.inference_mode():
            gen = edm.sample(50, (3, 64, 64), prev_frames_t, prev_actions_t,
                             state_t, body_mask_t)

        gen_np = gen[0].cpu().numpy()
        gen_np = postprocess(gen_np, state)

        img = ((gen_np + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(img).resize((256, 256), Image.NEAREST).save(f"out_{step:03d}.png")

        actions.append(action)
        state = next_state


if __name__ == "__main__":
    generate()