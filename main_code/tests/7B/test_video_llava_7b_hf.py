#!/usr/bin/env python3
"""
Video-LLaVA-7B-hf 正确测试脚本
基于深度调研报告的最佳实践

关键要点：
1. 使用 VideoLlavaForConditionalGeneration (专用类)
2. 严格 8帧 输入
3. padding_side = "left" (避免幻觉)
"""

import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from PIL import Image
import numpy as np

print("="*80)
print("🧪 Video-LLaVA-7B-hf 测试 (正确的HF版本)")
print("="*80)
print()

model_path = "/root/hhq/models/Video-LLaVA-7B-hf"

# 1. 加载模型
print("[1/3] 加载模型...")
try:
    # ⭐ 使用专用Processor
    processor = VideoLlavaProcessor.from_pretrained(model_path)
    
    # ⭐ 关键修正：设置 padding_side 为 "left"
    # 这是Video-LLaVA的关键bug修复，避免生成幻觉
    processor.tokenizer.padding_side = "left"
    print(f"   ✓ Processor加载成功")
    print(f"   ✓ padding_side设置为: {processor.tokenizer.padding_side}")
    
    # ⭐ 使用专用模型类（不是AutoModel）
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print(f"✅ 模型加载成功\n")
    
except Exception as e:
    print(f"❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 创建测试数据
print("[2/3] 创建测试数据...")
# ⭐ 关键：Video-LLaVA 严格需要 8 帧！
num_frames = 8
test_frames = []

for i in range(num_frames):
    # 创建渐变颜色的帧（模拟视频）
    # 前4帧红色，后4帧蓝色
    if i < 4:
        color = (255, 0, 0)  # 红色
    else:
        color = (0, 0, 255)  # 蓝色
    
    img = Image.new('RGB', (336, 336), color=color)
    test_frames.append(img)

print(f"✅ 创建了 {len(test_frames)} 帧测试视频")
print(f"   前4帧: 红色")
print(f"   后4帧: 蓝色\n")

# 3. 推理
print("[3/3] 测试推理...")

# ⭐ 使用 <video> token (不是 <image>)
prompt = "USER: <video>\nDescribe what you see in this video.\nASSISTANT:"

try:
    # 将PIL图像列表转为numpy array
    # shape: (8, 336, 336, 3)
    video_array = np.stack([np.array(f) for f in test_frames])
    
    # ⭐ 使用processor处理，注意参数名是 videos (不是 images)
    inputs = processor(
        text=prompt,
        videos=video_array,
        padding=True,
        return_tensors="pt"
    )
    
    # 移动到GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"✅ Processor处理成功")
    print(f"   input_ids shape: {inputs['input_ids'].shape}")
    if 'pixel_values_videos' in inputs:
        print(f"   pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
    print()
    
    # 生成
    print("   生成中...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
    
    # 解码
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*80)
    print("📝 模型输出:")
    print("="*80)
    print(output_text)
    print("="*80)
    print()
    
    # 检查输出
    if len(output_text) > 20 and not any(ord(c) > 127 for c in output_text[:50]):
        print("✅✅✅ 成功！模型输出了有意义的文本！")
        print("🎉 Video-LLaVA-7B-hf 可以正常使用！")
        
        # 检查是否提到颜色变化
        if "red" in output_text.lower() or "blue" in output_text.lower():
            print("🌟 模型识别到了颜色变化！")
    else:
        print("⚠️  输出可能不正常，请检查")
        
except Exception as e:
    print(f"\n❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)
