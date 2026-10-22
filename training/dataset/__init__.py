import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

# ==========================================
# 🚀 唯一需要保留的基类！
# DeepfakeBench 用它就能自动通吃 FF++ 和 Celeb-DF！
# ==========================================
from .abstract_dataset import DeepfakeAbstractBaseDataset

# 其他那些专属模型的 Dataset 全都不写，彻底杜绝依赖报错！
