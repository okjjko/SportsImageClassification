"""全局配置：路径、超参数"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据集路径
DATA_DIR = PROJECT_ROOT / "archive"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"
CSV_PATH = DATA_DIR / "sports.csv"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

# 确保输出目录存在
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 模型超参数
NUM_CLASSES = 100
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

# 学习率调度
LR_STEP_SIZE = 5
LR_GAMMA = 0.5

# 数据加载
NUM_WORKERS = 0  # Windows 兼容，避免 BrokenPipeError

# 模型保存路径
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

# matplotlib 中文字体
MATPLOTLIB_FONT = "SimHei"
