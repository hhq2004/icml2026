#!/usr/bin/env python3
"""
LLaVA-NeXT-Video-34B 模型封装
基于 test_llava_next_34b_hf.py 的最佳实践

关键配置：
- 使用 LlavaNextVideoForConditionalGeneration
- FP16 完整精度（不量化）
- device_map="auto" 多卡自动分配
- 建议输入：32帧（论文最佳实践）
"""

import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from PIL import Image
import numpy as np
import re


class LLaVANext34BWrapper:
    def __init__(self, model_path="/root/hhq/models/LLaVA-NeXT-Video-34B-hf"):
        """
        初始化 LLaVA-NeXT-Video-34B 模型
        
        Args:
            model_path: 模型权重路径
        """
        print(f"[LLaVA-NeXT-34B] Loading model from {model_path}...")
        
        # 加载 Processor
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_path)
        print(f"  ✓ Processor loaded")
        
        # 加载模型（FP16 完整精度）
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # FP16 完整精度
            device_map="auto",  # 自动多卡分配
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print(f"  ✓ Model loaded (FP16, device_map=auto)")
        
        # 获取设备
        self.device = next(self.model.parameters()).device
        print(f"  ✓ Device: {self.device}")
        
        # 最佳帧数配置
        self.recommended_frames = 32
        
    def generate(self, frames, question, options=None):
        """
        使用 LLaVA-NeXT-34B 生成答案
        
        Args:
            frames: PIL Image 列表，长度 <= 32
            question: 问题文本
            options: 选项列表 (可选，用于多选题)
        
        Returns:
            answer: 生成的答案文本（如果是多选题，返回 A/B/C/D）
        """
        # 检查帧数
        num_frames = len(frames)
        if num_frames > self.recommended_frames:
            print(f"⚠️  Warning: {num_frames} frames provided, recommended max is {self.recommended_frames}")
        
        # 构造 prompt（LLaVA-NeXT 格式）
        if options:
            # 多选题格式
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            prompt = f"USER: <video>\n{question}\n{options_text}\nAnswer with the option's letter from the given choices directly.\nASSISTANT:"
        else:
            # 开放式问题
            prompt = f"USER: <video>\n{question}\nASSISTANT:"
        
        # 转换 PIL 图像列表为 numpy array
        # shape: (num_frames, H, W, 3)
        video_array = np.stack([np.array(f) for f in frames])
        
        # 使用 processor 处理
        inputs = self.processor(
            text=prompt,
            videos=video_array,
            padding=True,
            return_tensors="pt"
        )
        
        # 移动到 GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # pixel_values 需要转为 float16 以匹配模型
        if 'pixel_values_videos' in inputs:
            inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.float16)
        
        # 生成答案
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        # 解码
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # 清理输出（移除 prompt 部分）
        # LLaVA-NeXT 输出格式：USER: ... ASSISTANT: <answer>
        if "ASSISTANT:" in output_text:
            output_text = output_text.split("ASSISTANT:")[-1].strip()
        
        # 如果是多选题，提取选项字母
        if options:
            answer = self._extract_option(output_text, len(options))
        else:
            answer = output_text
        
        return answer
    
    def _extract_option(self, text, num_options):
        """
        从生成的文本中提取选项字母（A/B/C/D）
        
        Args:
            text: 生成的文本
            num_options: 选项数量
        
        Returns:
            option: 提取的选项字母（A/B/C/D），如果提取失败返回 "A"
        """
        # 清理文本
        text = text.strip().upper()
        
        # 尝试多种提取模式
        patterns = [
            r'^([A-D])[.\s]',  # "A. " 或 "A "
            r'^([A-D])$',       # 单独的 "A"
            r'ANSWER[:\s]*([A-D])',  # "ANSWER: A" 或 "ANSWER A"
            r'OPTION[:\s]*([A-D])',  # "OPTION: A"
            r'\(([A-D])\)',     # "(A)"
            r'([A-D])[.\s]',    # 文本中的 "A. "
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                option = match.group(1)
                # 验证选项在有效范围内
                option_idx = ord(option) - ord('A')
                if 0 <= option_idx < num_options:
                    return option
        
        # 如果所有模式都失败，查找第一个出现的 A/B/C/D
        valid_options = [chr(65 + i) for i in range(num_options)]
        for char in text:
            if char in valid_options:
                return char
        
        # 默认返回 A（避免错误）
        print(f"⚠️  Warning: Could not extract option from: {text[:50]}..., defaulting to 'A'")
        return "A"


if __name__ == "__main__":
    # 简单测试
    print("=" * 80)
    print("Testing LLaVA-NeXT-34B Wrapper")
    print("=" * 80)
    
    # 创建测试帧
    test_frames = []
    for i in range(8):
        color = (255, 0, 0) if i < 4 else (0, 0, 255)
        img = Image.new('RGB', (336, 336), color=color)
        test_frames.append(img)
    
    # 加载模型
    wrapper = LLaVANext34BWrapper()
    
    # 测试生成
    question = "What colors appear in this video?"
    options = ["Red only", "Blue only", "Red and Blue", "Green and Yellow"]
    
    print(f"\n📝 Question: {question}")
    print(f"📋 Options: {options}")
    print(f"🎬 Frames: {len(test_frames)}")
    print("\n🔮 Generating...")
    
    answer = wrapper.generate(test_frames, question, options)
    
    print(f"\n✅ Answer: {answer}")
    print("=" * 80)
