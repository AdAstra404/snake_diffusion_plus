# Snake Diffusion Plus

用扩散模型（EDM）预测贪吃蛇游戏的下一帧画面。

快速开始（无需训练）

1. 安装依赖
pip install -r requirements.txt

2. 直接玩
打开根目录控制台，输入.\.venv\Scripts\Activate.ps1
输入python play.py即可开始游戏

> 预训练权重 `checkpoints/edm_final.pth` 已随仓库提供，无需自行训练。
> 操作：WASD 控制方向，SPACE 开始，R 重开（需锁定大写键盘）。

自己训练（可选）

1. 生成训练数据
python -m data.generate_data

2. 合并分片
python merge_data.py

3. 训练
python train.py

项目结构

- `data/` — 游戏逻辑、渲染、数据集
- `models/` — UNet 去噪网络 + EDM 扩散模型
- `train.py` — 训练脚本
- `play.py` — 键盘游玩 + 实时生成
- `generate.py` — 纯生成脚本
- `build_testset.py` — 构建测试集

## 依赖

见 `requirements.txt`。
