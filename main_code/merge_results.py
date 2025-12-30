#!/usr/bin/env python3
"""
合并4卡并行运行的结果文件

用法:
python merge_results.py \
    --input_dir ./results_4gpu \
    --pattern VideoMME_Q-Frame_all_Video-LLaVA-7B_chunk \
    --output merged_results.json
"""

import argparse
import json
import glob
import os


def merge_results(input_dir, pattern, output_file=None):
    """
    合并多个chunk的结果文件
    
    Args:
        input_dir: 结果文件所在目录
        pattern: 文件名模式（不包含chunk索引）
        output_file: 输出文件名（可选）
    """
    # 查找所有匹配的文件
    search_pattern = os.path.join(input_dir, f"{pattern}*.json")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"❌ No files found matching: {search_pattern}")
        return
    
    print(f"📁 Found {len(files)} chunk files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # 合并所有结果
    all_results = []
    for file_path in files:
        with open(file_path, 'r') as f:
            chunk_results = json.load(f)
            all_results.extend(chunk_results)
            print(f"  ✓ Loaded {len(chunk_results)} samples from {os.path.basename(file_path)}")
    
    # 去重（based on id）
    seen_ids = set()
    unique_results = []
    for result in all_results:
        if result['id'] not in seen_ids:
            unique_results.append(result)
            seen_ids.add(result['id'])
    
    if len(unique_results) < len(all_results):
        print(f"⚠️ Warning: Removed {len(all_results) - len(unique_results)} duplicate samples")
    
    # 统计
    total = len(unique_results)
    errors = sum(1 for r in unique_results if r.get('pred') == 'ERROR')
    valid = total - errors
    correct = sum(1 for r in unique_results if r.get('pred') == r.get('gt') and r.get('pred') != 'ERROR')
    accuracy = correct / valid * 100 if valid > 0 else 0
    
    print(f"\n📊 Merged Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Valid predictions: {valid}")
    print(f"  Errors: {errors}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # 保存合并结果
    if output_file is None:
        # 自动生成输出文件名
        output_file = os.path.join(input_dir, f"{pattern}_merged.json")
    
    with open(output_file, 'w') as f:
        json.dump(unique_results, f, indent=4)
    
    print(f"\n✅ Merged results saved to: {output_file}")
    return unique_results


def main():
    parser = argparse.ArgumentParser(description="Merge parallel inference results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing chunk files")
    parser.add_argument("--pattern", type=str, required=True, help="File pattern (without chunk index)")
    parser.add_argument("--output", type=str, default=None, help="Output filename (optional)")
    
    args = parser.parse_args()
    
    merge_results(args.input_dir, args.pattern, args.output)


if __name__ == "__main__":
    main()
