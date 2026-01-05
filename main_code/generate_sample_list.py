#!/usr/bin/env python3
"""
生成样本索引列表文件，用于GNU Parallel批处理

Usage:
    python generate_sample_list.py --dataset VideoMME --output sample_list.txt
"""

import argparse
import json
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import DATASET_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Generate sample index list for GNU Parallel")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["VideoMME", "LongVideoBench", "MLUV"],
                       help="Dataset name")
    parser.add_argument("--data_root", type=str, default="/root/hhq/dataset",
                       help="Dataset root directory")
    parser.add_argument("--duration_mode", type=str, default="all",
                       choices=["all", "ideal", "long", "short"],
                       help="Duration filter mode")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit total number of samples")
    parser.add_argument("--output", type=str, default="sample_list.txt",
                       help="Output file path")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Number of samples per batch (for grouping)")
    
    args = parser.parse_args()
    
    # 加载数据集
    print(f"📂 Loading dataset: {args.dataset}")
    dataset_class = DATASET_REGISTRY[args.dataset]
    
    if args.dataset == "VideoMME" and args.duration_mode != "all":
        dataset = dataset_class(root_dir=args.data_root, duration_mode=args.duration_mode)
    else:
        dataset = dataset_class(root_dir=args.data_root)
    
    # 限制样本数
    total_samples = len(dataset)
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
    
    print(f"📊 Total samples: {total_samples}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📝 Output: {args.output}")
    
    # 生成批次
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    
    with open(args.output, 'w') as f:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, total_samples)
            
            # 生成该batch的索引列表
            indices = list(range(start_idx, end_idx))
            indices_str = ','.join(map(str, indices))
            f.write(f"{indices_str}\n")
    
    print(f"✅ Generated {num_batches} batches")
    print(f"   Average {args.batch_size} samples/batch")
    print(f"   Last batch: {total_samples % args.batch_size or args.batch_size} samples")


if __name__ == "__main__":
    main()
