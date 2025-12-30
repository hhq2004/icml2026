#!/usr/bin/env python3
"""
最简单的Video-LLaVA测试
用于诊断问题
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np

print("🔍 诊断测试开始...\n")

# 1. 加载模型
print("[1] 加载模型...")
model_path = "/root/hhq/models/Video-LLaVA-7B"

try:
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    
    # ⭐ 修复processor配置（和wrapper中一样）
    if hasattr(processor, 'image_processor'):
        processor.image_processor.size = {"shortest_edge": 336}
        processor.image_processor.crop_size = {"height": 336, "width": 336}
    if not hasattr(processor, 'patch_size') or processor.patch_size is None:
        processor.patch_size = 14
    if not hasattr(processor, 'vision_feature_select_strategy'):
        processor.vision_feature_select_strategy = "default"
    print("✓ Processor配置已修复")
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    print("✓ 模型加载成功\n")
except Exception as e:
    print(f"✗ 模型加载失败: {e}\n")
    exit(1)

# 2. 创建测试图像（单张）
print("[2] 创建测试图像...")
# 创建一个纯红色的336x336图像
test_image = Image.new('RGB', (336, 336), color='red')
print("✓ 创建了一张纯红色图像\n")

# 3. 简单推理
print("[3] 测试推理...")
prompt = "USER: <image>\nWhat color is this image?\nASSISTANT:"

try:
    # 对于单张图像，使用images参数
    inputs = processor(
        text=prompt,
        images=test_image,
        return_tensors="pt"
    )
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    print(f"✓ Processor成功")
    print(f"  - input_ids shape: {inputs['input_ids'].shape}")
    if 'pixel_values' in inputs:
        print(f"  - pixel_values shape: {inputs['pixel_values'].shape}")
    print()
    
    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
    
    # 解码
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    print("=" * 60)
    print("📝 模型输出:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)
    
    # 检查输出
    if "red" in output_text.lower():
        print("\n✅ 成功！模型正确识别了红色")
    elif len(output_text) < 10 or any(ord(c) > 127 for c in output_text[:20]):
        print("\n❌ 输出是乱码！模型可能没有正确加载权重")
    else:
        print("\n⚠️  输出不包含'red'，但不是乱码")
    
except Exception as e:
    print(f"✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n诊断完成")
