import numpy as np
from PIL import Image

# 颜色
BG_COLOR = (20, 20, 20)       # 背景深灰
SNAKE_COLOR = (0, 200, 0)     # 蛇身绿色
HEAD_COLOR = (0, 255, 0)      # 蛇头亮绿
FOOD_COLOR = (255, 50, 50)    # 食物红色

class Renderer:
    def __init__(self, grid_size=16, image_size=64):
        self.grid_size = grid_size
        self.image_size = image_size
        self.cell = image_size // grid_size   # 每格 4 像素

    def render(self, state):
        """把状态渲染成 (3, H, W) 的 numpy 数组，范围 [-1, 1]"""
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img[:] = BG_COLOR

        # 画蛇身
        for y, x in state["body"][1:]:
            self._fill_cell(img, y, x, SNAKE_COLOR)
        # 画蛇头
        hy, hx = state["head"]
        self._fill_cell(img, hy, hx, HEAD_COLOR)
        # 画食物
        fy, fx = state["food"]
        self._fill_cell(img, fy, fx, FOOD_COLOR)

        # 归一化到 [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        return img.transpose(2, 0, 1)   # (3, H, W)

    def _fill_cell(self, img, y, x, color):
        y0, x0 = y * self.cell, x * self.cell
        img[y0:y0+self.cell, x0:x0+self.cell] = color