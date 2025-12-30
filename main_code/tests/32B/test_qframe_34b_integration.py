#!/usr/bin/env python3
"""
Q-Frame + 34B 模型集成测试

测试完整流程：
1. 加载 34B 模型
2. 运行 Q-Frame 选择帧
3. 模型推理
4. 验证输出格式
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.llava_next_34b import LLaVANext34BWrapper
from methods.q_frame_clean import QFrameClean


def create_test_video_frames(num_frames=16):
    """创建测试视频帧（模拟视频）"""
    frames = []
    for i in range(num_frames):
        # 前一半红色，后一半蓝色
        if i < num_frames // 2:
            color = (255, 0, 0)  # 红色
        else:
            color = (0, 0, 255)  # 蓝色
        
        img = Image.new('RGB', (336, 336), color=color)
        frames.append(img)
    
    return frames


def test_model_wrapper():
    """测试 34B 模型封装"""
    print("\n" + "="*80)
    print("测试1: LLaVA-NeXT-34B 模型封装")
    print("="*80)
    
    try:
        # 加载模型
        print("\n[1/3] 加载 34B 模型...")
        model = LLaVANext34BWrapper()
        print("✅ 模型加载成功")
        
        # 创建测试帧
        print("\n[2/3] 创建测试帧...")
        test_frames = create_test_video_frames(8)
        print(f"✅ 创建了 {len(test_frames)} 帧测试视频")
        
        # 测试推理
        print("\n[3/3] 测试推理...")
        question = "What colors appear in this video?"
        options = ["Red only", "Blue only", "Red and Blue", "Green and Yellow"]
        
        answer = model.generate(test_frames, question, options)
        
        print(f"\n📝 Question: {question}")
        print(f"📋 Options: {options}")
        print(f"✅ Answer: {answer}")
        
        # 验证答案格式
        if answer in ["A", "B", "C", "D"]:
            print("✅ PASS: 答案格式正确")
            return True
        else:
            print(f"❌ FAIL: 答案格式错误 ({answer})")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qframe_integration():
    """测试 Q-Frame + 34B 完整集成"""
    print("\n" + "="*80)
    print("测试2: Q-Frame + 34B 完整集成")
    print("="*80)
    
    try:
        # 创建临时 args
        class Args:
            temperature = 1.0
            token_budget = 2048
        
        # 加载模型
        print("\n[1/4] 加载 34B 模型...")
        model = LLaVANext34BWrapper()
        print("✅ 模型加载成功")
        
        # 初始化 Q-Frame
        print("\n[2/4] 初始化 Q-Frame...")
        qframe = QFrameClean(Args(), model)
        print("✅ Q-Frame 初始化成功")
        
        # 测试帧采样和相似度计算
        print("\n[3/4] 测试 CLIP 相似度计算...")
        test_frames = create_test_video_frames(10)
        question = "What is happening in the video?"
        
        similarities = qframe._cross_modal_query_retrieval(test_frames, question)
        print(f"✅ 相似度计算完成: shape={similarities.shape}")
        
        # 测试 Gumbel-Max 采样
        print("\n[4/4] 测试 Gumbel-Max 采样...")
        selected_indices = qframe._gumbel_max_sampling(similarities, k=4, tau=1.0)
        print(f"✅ 采样完成: 选中帧索引={selected_indices.tolist()}")
        
        # 验证采样结果
        if len(selected_indices) == 4:
            print("✅ PASS: 采样数量正确")
        else:
            print(f"❌ FAIL: 采样数量错误 ({len(selected_indices)} != 4)")
            return False
        
        # 验证时间顺序
        is_sorted = all(selected_indices[i] <= selected_indices[i+1] for i in range(len(selected_indices)-1))
        if is_sorted:
            print("✅ PASS: 时间顺序正确")
            return True
        else:
            print("❌ FAIL: 时间顺序错误")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("Q-Frame + 34B 集成测试")
    print("="*80)
    
    results = []
    
    # 测试1: 模型封装
    result1 = test_model_wrapper()
    results.append(("模型封装", result1))
    
    # 测试2: 完整集成
    result2 = test_qframe_integration()
    results.append(("Q-Frame集成", result2))
    
    # 总结
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✅✅✅ 所有测试通过！可以开始运行冒烟测试。")
    else:
        print("\n❌ 存在失败的测试，请检查。")
    
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
