import numpy as np

# 动作：0=右, 1=左, 2=上, 3=下
ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # (dy, dx)

class SnakeGame:
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """重置游戏"""
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]  # 蛇头在前
        self.direction = 0                                          # 初始向右
        self.food = self._random_food()
        self.done = False
        self.score = 0
        return self._get_state()

    def _random_food(self):
        """在空位随机放食物"""
        empty = [(y, x) for y in range(self.grid_size)
                        for x in range(self.grid_size)
                        if (y, x) not in self.snake]
        return empty[np.random.randint(len(empty))]

    def step(self, action):
        """执行一步动作，返回 (state, reward, done)"""
        if self.done:
            return self._get_state(), 0, True

        # 不能 180 度掉头
        if (action ^ 1) == self.direction:
            action = self.direction
        self.direction = action

        dy, dx = ACTIONS[action]
        head_y, head_x = self.snake[0]
        new_head = (head_y + dy, head_x + dx)

        # 撞墙
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.done = True
            return self._get_state(), -1, True

        # 撞自己（尾巴会移动，所以尾巴不算）
        if new_head in self.snake[:-1]:
            self.done = True
            return self._get_state(), -1, True

        # 移动
        self.snake.insert(0, new_head)

        # 吃食物
        if new_head == self.food:
            self.score += 1
            self.food = self._random_food()
            reward = 1
        else:
            self.snake.pop()   # 没吃到，尾巴收缩
            reward = 0

        return self._get_state(), reward, self.done

    def _get_state(self):
        """返回当前状态（坐标）"""
        return {
            "head": self.snake[0],
            "food": self.food,
            "body": list(self.snake),
            "direction": self.direction,
            "score": self.score,
            "done": self.done,
        }