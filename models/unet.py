import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        x = x.to(torch.float32)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaGN(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        h = self.norm(x)
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adagn1 = AdaGN(out_ch, cond_dim)
        self.adagn2 = AdaGN(out_ch, cond_dim)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.act(self.adagn1(self.conv1(x), cond))
        h = self.act(self.adagn2(self.conv2(h), cond))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, cond_dim=256,
                 context_length=4, actions_count=5):
        super().__init__()
        self.context_length = context_length

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.action_emb = nn.Embedding(actions_count, cond_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(10, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        total_in = in_ch * (context_length + 1) + 1
        self.in_conv = nn.Conv2d(total_in, base_ch, 3, padding=1)

        # 编码器：3 层
        self.enc1 = ResBlock(base_ch, base_ch, cond_dim)           # 64×64
        self.enc2 = ResBlock(base_ch, base_ch * 2, cond_dim)       # 32×32
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, cond_dim)   # 16×16

        # 中间
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, cond_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, cond_dim)

        # 解码器：2 层
        self.dec2 = ResBlock(base_ch * 4 + base_ch * 2, base_ch * 2, cond_dim)  # 32×32
        self.dec1 = ResBlock(base_ch * 2 + base_ch, base_ch, cond_dim)          # 64×64

        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # 辅助头
        self.head_pos_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base_ch * 4, 2)
        )
        self.food_pos_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base_ch * 4, 2)
        )

    def forward(self, x, t, actions, state):
        t_emb = self.time_emb(t)
        a_emb = self.action_emb(actions).mean(dim=1)
        s_emb = self.state_proj(state)
        cond = t_emb + a_emb + s_emb

        h0 = self.in_conv(x)                        # 64×64, base_ch
        h1 = self.enc1(h0, cond)                    # 64×64, base_ch
        h2 = self.enc2(self.pool(h1), cond)         # 32×32, 2*base_ch
        h3 = self.enc3(self.pool(h2), cond)         # 16×16, 4*base_ch

        m = self.mid1(h3, cond)                     # 16×16, 4*base_ch
        m = self.mid2(m, cond)                      # 16×16, 4*base_ch

        u2 = self.dec2(torch.cat([self.up(m), h2], dim=1), cond)  # 32×32
        u1 = self.dec1(torch.cat([self.up(u2), h1], dim=1), cond) # 64×64

        out = self.out_conv(u1)                     # 64×64
        head_pos = self.head_pos_head(m)
        food_pos = self.food_pos_head(m)
        return out, head_pos, food_pos