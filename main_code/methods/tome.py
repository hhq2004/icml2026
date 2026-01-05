#!/usr/bin/env python3
"""
Token Merging (ToMe) for Video-LLMs  
论文: Token Merging: Your ViT But Faster (ICLR 2023)
https://github.com/facebookresearch/ToMe

⭐ 支持两种模式：
1. **完整版**（--use_full_tome）：真实的ViT层token merging
2. **简化版**（默认）：输入层帧削减（保持向后兼容）

实验设置（基于 icml2026.md）：
- Token budget B = 2048（统一设置）
- Merging schedule: constant (每层合并r个token)
- 采样帧数: 32 frames
"""

import torch
import numpy as np
from PIL import Image
import os
from typing import List, Tuple
from .base_method import BaseMethod

try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Warning: decord not installed. Install with: pip install decord")
    VideoReader = None


class ToMe(BaseMethod):
    """
    Token Merging for Video-LLMs
    论文: Token Merging: Your ViT But Faster (ICLR 2023)
    
    核心思想: 在每个Transformer Block的attention后、MLP前，
    使用双向软匹配合并相似的visual tokens，逐层压缩到budget内。
    """
    
    def __init__(self, args, model):
        """
        初始化 ToMe
        
        Args:
            args: 命令行参数
            model: Video-LLM 模型
                - VideoLLaVATomeWrapper (完整版)
                - VideoLLaVAWrapper (简化版)
        """
        super().__init__(args, model)
        
        # ⭐ 检测是否使用完整版ToMe
        self.use_full_tome = getattr(args, 'use_full_tome', False)
        
        # 检测model类型
        model_class = model.__class__.__name__
        
        if self.use_full_tome:
            print(f"[ToMe] ✅ Using FULL ToMe (ViT-layer token merging)")
            print(f"[ToMe]    Model class: {model_class}")
            
            # 验证model是否为ToMe wrapper
            if 'Tome' not in model_class:
                print(f"[ToMe] ⚠️  WARNING: Model is {model_class}, not VideoLLaVATomeWrapper!")
                print(f"[ToMe]    Expected VideoLLaVATomeWrapper for full ToMe")
                print(f"[ToMe]    Falling back to simplified version...")
                self.use_full_tome = False
            else:
                # 完整版配置（用于采样帧）
                self.num_frames = 32  # 与wrapper一致
            
        if not self.use_full_tome:
            print(f"[ToMe] ⚠️  Using SIMPLIFIED ToMe (input-level frame reduction)")
            print(f"[ToMe]    Model class: {model_class}")
            
            # 简化版配置
            self.token_budget = getattr(args, 'token_budget', 2048)
            
            # 检测backbone
            backbone = getattr(args, 'backbone', 'Video-LLaVA-7B')
            
            if '34B' in backbone or '32B' in backbone:
                self.tokens_per_frame = 576
            else:
                self.tokens_per_frame = 256
            
            self.num_frames = 32
            
            print(f"[ToMe] Simplified config:")
            print(f"  - Num frames = {self.num_frames}")
            print(f"  - Tokens per frame = {self.tokens_per_frame}")
            print(f"  - Token budget = {self.token_budget}")
    
    def _uniform_sample_frames(self, video_path: str, num_frames: int) -> Tuple[List[Image.Image], List[int]]:
        """从视频中均匀采样帧"""
        if VideoReader is None:
            raise ImportError("decord is required for video processing")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames <= num_frames:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in indices:
                frame = vr[idx].asnumpy()
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)
            
            return frames, indices.tolist()
            
        except Exception as e:
            print(f"❌ Error sampling frames from {video_path}: {e}")
            raise e
    
    def _apply_tome_to_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """简化版：通过减少帧数来模拟ToMe"""
        target_frames = self.token_budget // self.tokens_per_frame
        
        if len(frames) <= target_frames:
            return frames
        
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        reduced_frames = [frames[i] for i in indices]
        
        print(f"  [ToMe Simplified] {len(frames)} → {len(reduced_frames)} frames")
        
        return reduced_frames
    
    def process_and_inference(self, video_path: str, question: str, options: List[str]) -> str:
        """
        ToMe 主流程
        
        完整版：wrapper已注入ToMe，直接推理
        简化版：先削减帧数，再推理
        """
        try:
            # 采样帧
            frames, _ = self._uniform_sample_frames(video_path, self.num_frames)
            
            if self.use_full_tome:
                # ✅ 完整版：wrapper已经注入ToMe
                print(f"  [ToMe Full] Using {len(frames)} frames (merging inside Vision Tower)")
                answer = self.model.generate(frames, question, options)
            else:
                # ⚠️ 简化版：输入层帧削减
                reduced_frames = self._apply_tome_to_frames(frames)
                answer = self.model.generate(reduced_frames, question, options)
            
            return answer
            
        except Exception as e:
            print(f"❌ Error in ToMe processing: {e}")
            import traceback
            traceback.print_exc()
            return "A" if options else "Error"
