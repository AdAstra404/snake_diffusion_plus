# eval/paths.py
"""
统一路径配置。

所有评估脚本从这里 import 路径，避免硬编码相对路径。
"""
import os

# 项目根目录 = eval/ 的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
TEST_DIR = os.path.join(PROJECT_ROOT, "testset")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "training_data")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "eval", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型权重
EDM_CKPT = os.path.join(CKPT_DIR, "edm_final.pth")

# 结果文件
ABLATION_NPZ = os.path.join(OUTPUT_DIR, "ablation_results.npz")
ABLATION_QUANTIZE_NPZ = os.path.join(OUTPUT_DIR, "ablation_quantize.npz")
PSNR_TABLE_NPY = os.path.join(OUTPUT_DIR, "psnr_table.npy")


def setup_sys_path():
    """把 eval/ 目录加到 sys.path，保证同目录模块可 import"""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)