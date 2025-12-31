# /root/hhq/main_code/methods/__init__.py

# Q-Frame Clean实现（当前测试用）
from .q_frame_clean import QFrameClean

# SceneGraph-Cap实现（ICML2025 baseline）
from .scenegraph_cap import SceneGraphCap

# ToMe实现（ICLR 2023 baseline）
from .tome import ToMe

# FastV实现（ECCV 2024 baseline）
from .fastv import FastV

# 注册表
METHOD_REGISTRY = {
    "Q-Frame-Clean": QFrameClean,
    "SceneGraph-Cap": SceneGraphCap,
    "ToMe": ToMe,
    "FastV": FastV
}