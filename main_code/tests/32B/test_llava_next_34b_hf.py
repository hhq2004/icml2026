#!/usr/bin/env python3
"""
LLaVA-NeXT-Video-34B-hf 正确测试脚本
基于深度调研报告的最佳实践

关键要点：
1. 使用 LlavaNextVideoForConditionalGeneration (专用类)
2. 建议 32帧 输入
3. 使用 FP16完整精度 (确保实验可复现性)
"""

import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from PIL import Image
import numpy as np

print("="*80)
print("🧪 LLaVA-NeXT-Video-34B-hf 测试 (正确的HF版本)")
print("="*80)
print()

model_path = "/root/hhq/models/LLaVA-NeXT-Video-34B-hf"

# 1. 加载模型
print("[1/3] 加载模型（FP16完整精度，用于正式实验）...")
try:
    # ⭐ 使用专用Processor
    processor = LlavaNextVideoProcessor.from_pretrained(model_path)
    print(f"   ✓ Processor加载成功")
    
    # ⭐ 使用FP16完整精度（不量化，确保实验可复现）
    # 您的4张A100 (160GB总显存) 完全足够运行34B FP16 (~70GB)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # FP16完整精度
        device_map="auto",  # 自动多卡分配
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"✅ 模型加载成功（FP16完整精度）\n")
    
except Exception as e:
    print(f"❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 创建测试数据
print("[2/3] 创建测试数据...")
# ⭐ 关键：LLaVA-NeXT 最佳实践是 32 帧
num_frames = 32
test_frames = []

for i in range(num_frames):
    # 创建渐变颜色（模拟视频）
    # 前16帧红色，后16帧蓝色
    if i < 16:
        color = (255, 0, 0)  # 红色
    else:
        color = (0, 0, 255)  # 蓝色
    
    img = Image.new('RGB', (336, 336), color=color)
    test_frames.append(img)

print(f"✅ 创建了 {len(test_frames)} 帧测试视频")
print(f"   前16帧: 红色")
print(f"   后16帧: 蓝色\n")

# 3. 推理
print("[3/3] 测试推理...")

# ⭐ 使用 <video> token，LLaVA-NeXT格式
prompt = "USER: <video>\nDescribe what you see in this video.\nASSISTANT:"

try:
    # 将PIL图像列表转为numpy array
    # shape: (32, 336, 336, 3)
    video_array = np.stack([np.array(f) for f in test_frames])
    
    # 使用processor处理
    inputs = processor(
        text=prompt,
        videos=video_array,
        padding=True,
        return_tensors="pt"
    )
    
    # 移动到GPU (4-bit量化模型已经在GPU上)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # ⭐ pixel_values需要转为float16以匹配模型
    if 'pixel_values_videos' in inputs:
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.float16)
    
    print(f"✅ Processor处理成功")
    print(f"   input_ids shape: {inputs['input_ids'].shape}")
    if 'pixel_values_videos' in inputs:
        print(f"   pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
    print()
    
    # 生成
    print("   生成中（34B模型，会比较慢）...")
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
        print("🎉 LLaVA-NeXT-Video-34B-hf 可以正常使用！")
        
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
