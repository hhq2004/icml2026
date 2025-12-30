# /root/hhq/main_code/methods/__init__.py

# Q-Frame Clean实现（当前测试用）
from .q_frame_clean import QFrameClean

# 注册表
METHOD_REGISTRY = {
    "Q-Frame-Clean": QFrameClean
}