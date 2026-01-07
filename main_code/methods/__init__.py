# /root/hhq/main_code/methods/__init__.py

# Q-Frame Clean实现（当前测试用）
from .q_frame_clean import QFrameClean

# SceneGraph-Cap实现（ICML2025 baseline）
from .scenegraph_cap import SceneGraphCap

# ToMe实现（ICLR 2023 baseline）
from .tome import ToMe

# FastV实现（ECCV 2024 baseline）
from .fastv import FastV

# DyCoke实现（CVPR 2025 baseline - 完整TTM版本）
from .DyCoke import DyCokeMethod

# EventGraph-LMM实现（Ours - ICML 2026）
from .eventgraph import EventGraphLMM

# No-Compression Baseline（诊断用基准）
from .no_compression import NoCompression

# 别名导出 (兼容性)
DyCoke = DyCokeMethod  # ← 添加别名

# 注册表
METHOD_REGISTRY = {
    "Q-Frame-Clean": QFrameClean,
    "SceneGraph-Cap": SceneGraphCap,
    "ToMe": ToMe,
    "FastV": FastV,
    "DyCoke": DyCokeMethod,  # 使用完整TTM版本
    "EventGraph-LMM": EventGraphLMM,  # 主方法
    "No-Compression": NoCompression  # 诊断基准
}