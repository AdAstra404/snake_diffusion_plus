import pygame
import torch
import numpy as np
from models.unet import UNet
from models.edm import EDM
from data.game import SnakeGame, ACTIONS
from data.render import Renderer


def render_body_mask(body, grid_size=16, image_size=64):
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    cell = image_size // grid_size
    for y, x in body:
        mask[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = 1.0
    return mask


def postprocess(gen_frame, state, cell=4):
    img = gen_frame.copy()
    H = img.shape[1]

    def fill(y, x, color, expand=1):
        y0 = max(0, y * cell - expand)
        y1 = min(H, (y + 1) * cell + expand)
        x0 = max(0, x * cell - expand)
        x1 = min(H, (x + 1) * cell + expand)
        img[:, y0:y1, x0:x1] = np.array(color)[:, None, None]

    fy, fx = int(state["food"][0]), int(state["food"][1])
    fill(fy, fx, [1.0, -0.6, -0.6], expand=1)
    hy, hx = int(state["head"][0]), int(state["head"][1])
    fill(hy, hx, [-1.0, 1.0, -1.0], expand=0)
    return img


def build_inputs(frames, actions, state, context, device):
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


# 键盘 → 方向映射：0=右, 1=左, 2=上, 3=下
KEY_TO_ACTION = {
    pygame.K_d: 0,
    pygame.K_a: 1,
    pygame.K_w: 2,
    pygame.K_s: 3,
}


def play():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    unet = UNet(context_length=4)
    edm = EDM(unet, sigma_data=0.1).to(device)
    edm.load_state_dict(torch.load("checkpoints/edm_final.pth", map_location=device))
    edm.eval()

    game = SnakeGame()
    renderer = Renderer()
    state = game.reset()

    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    pygame.display.set_caption("Snake Diffusion - WASD to steer, R restart, SPACE start")
    clock = pygame.time.Clock()

    frames, actions = [], []
    game_over = False
    started = False

    MOVE_INTERVAL_MS = 1000     # 1 秒走一格，给玩家时间按方向键
    current_action = game.direction
    last_move_time = pygame.time.get_ticks()

    def render_frame(frame_np):
        img = ((frame_np + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (512, 512))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        if game_over:
            status = "GAME OVER - press R"
        elif not started:
            status = "Press SPACE to start"
        else:
            status = "WASD to steer"
        print(f"Score: {state['score']}  Dir: {state['direction']}  {status}", end="\r")

    def reset_all():
        nonlocal game, state, frames, actions, game_over, started, current_action, last_move_time
        game = SnakeGame()
        state = game.reset()
        frames, actions = [], []
        game_over = False
        started = False
        current_action = game.direction
        last_move_time = pygame.time.get_ticks()
        render_frame(renderer.render(state))
        print("\nReset. Press SPACE to start.")

    def step_and_generate(action):
        nonlocal state, game_over
        next_state, _, done = game.step(action)

        prev_frames_t, prev_actions_t, state_t, body_mask_t = build_inputs(
            frames, actions, state, 4, device
        )
        actions.append(action)

        with torch.inference_mode():
            gen = edm.sample(50, (3, 64, 64), prev_frames_t, prev_actions_t,
                             state_t, body_mask_t)
        gen_np = postprocess(gen[0].cpu().numpy(), next_state)

        # 训练数据里 prev_frames 是真实渲染帧，这里保持一致
        frames.append(renderer.render(next_state))

        state = next_state
        if done:
            game_over = True
            print(f"\nGame over! Final score: {state['score']}")

        render_frame(gen_np)

    # 初始画面：显示真实第一帧，等待用户按 SPACE
    render_frame(renderer.render(state))
    print("\nPress SPACE to start.")

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_all()

                elif event.key == pygame.K_SPACE:
                    if not started and not game_over:
                        started = True
                        last_move_time = pygame.time.get_ticks()
                        print("\nStarted!")

                elif event.key in KEY_TO_ACTION:
                    if started and not game_over:
                        new_action = KEY_TO_ACTION[event.key]
                        # 用 XOR 1 判断相反方向：
                        # 0(右) <-> 1(左)，2(上) <-> 3(下)
                        if (new_action ^ 1) != current_action:
                            current_action = new_action
                            last_move_time = 0   # 立即触发下一次步进
                            print(f"\nDirection -> {current_action}")

        if started and not game_over and (now - last_move_time) >= MOVE_INTERVAL_MS:
            step_and_generate(current_action)
            last_move_time = now

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    play()