#!/usr/bin/env python3
"""
No-Compression Baseline for Video-LLM
无压缩基准方法：直接均匀采样K帧喂给LLM，不做任何query-aware选择或token压缩

目的：作为性能上界参考，诊断压缩方法的效果
- 如果压缩方法 << No-Compression: 说明压缩损失了关键信息
- 如果压缩方法 ≈ No-Compression: 说明压缩效果良好

实验设置（基于 icml2026.md）：
- Token budget B = 2048（与所有baseline一致）
- 选择帧数 K: 7B模型=8帧, 34B模型=3帧（自动计算）
- 采样方式: 均匀采样（不考虑query相关性）
"""

import torch
import numpy as np
from PIL import Image
import os
from .base_method import BaseMethod

try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Warning: decord not installed. Install with: pip install decord")
    VideoReader = None


class NoCompression(BaseMethod):
    """无压缩基准方法：均匀采样K帧直接送入LLM"""
    
    def __init__(self, args, model):
        """
        初始化 No-Compression Baseline
        
        Args:
            args: 命令行参数
            model: Video-LLM 模型（支持 Video-LLaVA-7B 或 LLaVA-NeXT-34B）
        """
        super().__init__(args, model)
        
        # 根据backbone自动检测tokens_per_frame
        backbone = getattr(args, 'backbone', 'Video-LLaVA-7B')
        
        if '34B' in backbone or '32B' in backbone:
            # LLaVA-NeXT-34B: 336x336 → (336/14)^2 = 576 tokens/frame
            self.tokens_per_frame = 576
        else:
            # Video-LLaVA-7B: 224x224 → (224/14)^2 = 256 tokens/frame
            self.tokens_per_frame = 256
        
        self.token_budget = getattr(args, 'token_budget', 2048)
        
        # 计算最大帧数（不超过token budget）
        self.num_frames = self.token_budget // self.tokens_per_frame
        
        print(f"[No-Compression] Initializing...")
        print(f"  - Backbone: {backbone}")
        print(f"  - Token budget B = {self.token_budget}")
        print(f"  - Tokens per frame = {self.tokens_per_frame}")
        print(f"  - Sampled frames K = {self.num_frames}")
        print(f"  - Estimated tokens = {self.num_frames * self.tokens_per_frame}")
        print(f"  - Method: Uniform sampling (NO query-aware selection)")
        print(f"  - Purpose: Performance upper-bound baseline")
    
    def _uniform_sample_frames(self, video_path, num_frames):
        """
        从视频中均匀采样帧（与Q-Frame相同的采样方式，但不做query-aware选择）
        
        Args:
            video_path: 视频文件路径
            num_frames: 采样帧数（K）
        
        Returns:
            frames: PIL Image 列表，长度为 num_frames
        """
        if VideoReader is None:
            raise ImportError("decord is required for video processing")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # 均匀采样索引
            if total_frames <= num_frames:
                # 如果视频帧数不足，使用所有帧
                indices = list(range(total_frames))
            else:
                # 均匀采样
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            # 读取帧
            frames = []
            for idx in indices:
                frame = vr[idx].asnumpy()  # (H, W, 3) numpy array
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)
            
            return frames
            
        except Exception as e:
            print(f"❌ Error sampling frames from {video_path}: {e}")
            raise e
    
    def process_and_inference(self, video_path, question, options):
        """
        No-Compression 主流程
        
        步骤:
        1. 均匀采样 K 帧（K由token budget决定）
        2. 直接将K帧送入 Video-LLM 推理（不做任何query-aware选择）
        
        Args:
            video_path: 视频路径
            question: 问题文本
            options: 选项列表
        
        Returns:
            answer: 模型预测的答案
        """
        try:
            # Step 1: 均匀采样 K 帧
            frames = self._uniform_sample_frames(video_path, self.num_frames)
            
            # Step 2: 直接使用 Video-LLM 推理（不做任何压缩或选择）
            # 调用底层模型（如 Video-LLaVA-7B 或 LLaVA-NeXT-34B）
            answer = self.model.generate(frames, question, options)
            
            return answer
            
        except Exception as e:
            print(f"❌ Error in No-Compression processing: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认答案，避免中断整个实验
            return "A" if options else "Error"


if __name__ == "__main__":
    # 简单测试
    print("=" * 80)
    print("Testing No-Compression Baseline Implementation")
    print("=" * 80)
    
    # 创建临时对象来测试
    class Args:
        token_budget = 2048
        backbone = 'Video-LLaVA-7B'
    
    class DummyModel:
        pass
    
    no_comp = NoCompression(Args(), DummyModel())
    
    print("\n✅ No-Compression baseline ready!")
    print("=" * 80)
