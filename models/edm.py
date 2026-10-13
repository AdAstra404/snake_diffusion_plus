import torch
import torch.nn as nn


class EDM(nn.Module):
    def __init__(self, model, sigma_data=0.1, sigma_min=0.002,
                 sigma_max=80, rho=7, p_mean=-1.2, p_std=1.2):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std

    def _denoise(self, x, sigma, prev_frames, prev_actions, state, body_mask,
                 return_aux=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        frames_cat = torch.cat([(c_in * x)[:, None], prev_frames], dim=1).flatten(1, 2)
        noise_imgs = torch.cat([frames_cat, body_mask[:, None]], dim=1)
        F_x, head_pred, food_pred = self.model(
            noise_imgs, c_noise.flatten(), prev_actions, state
        )
        F_x = F_x.to(torch.float32)
        D_x = c_skip * x + c_out * F_x
        if return_aux:
            return D_x, head_pred, food_pred
        return D_x

    def forward(self, next_frame, prev_frames, prev_actions, state, body_mask,
                return_aux=False):
        B = next_frame.shape[0]
        current = prev_frames[:, -1]
        delta = next_frame - current

        sigma = (torch.randn(B, device=next_frame.device, dtype=torch.float32)
                 * self.p_std + self.p_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        n = torch.randn_like(delta, dtype=torch.float32) * sigma.reshape(-1, 1, 1, 1)
        noisy = (delta + n).to(torch.float32)

        if return_aux:
            D, head_pred, food_pred = self._denoise(
                noisy, sigma, prev_frames, prev_actions, state, body_mask,
                return_aux=True
            )
            loss_diff = (weight.reshape(-1, 1, 1, 1) * ((D - delta) ** 2)).mean()
            return loss_diff, head_pred, food_pred
        else:
            D = self._denoise(noisy, sigma, prev_frames, prev_actions, state, body_mask)
            loss_diff = (weight.reshape(-1, 1, 1, 1) * ((D - delta) ** 2)).mean()
            return loss_diff

    @torch.no_grad()
    def sample(self, steps, size, prev_frames, prev_actions, state, body_mask):
        device = prev_frames.device
        x = torch.randn(1, *size, device=device, dtype=torch.float32) * self.sigma_max

        step_indices = torch.arange(steps, device=device, dtype=torch.float32)
        t_steps = (self.sigma_max ** (1 / self.rho) +
                   step_indices / (steps - 1) *
                   (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            denoised = self._denoise(x, t_cur.reshape(1), prev_frames,
                                     prev_actions, state, body_mask)
            d_cur = (x - denoised) / t_cur
            x = x + (t_next - t_cur) * d_cur

        delta = x
        current = prev_frames[:, -1]
        next_frame = current + delta
        return next_frame