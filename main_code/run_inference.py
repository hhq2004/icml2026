# /root/hhq/main_code/run_inference.py
import argparse
import os
import json
from datasets import DATASET_REGISTRY
from methods import METHOD_REGISTRY

# 引入我们刚才写的 VideoLLaVAWrapper
from models.video_llava_7b import VideoLLaVAWrapper

def load_model(backbone_name):
    """
    模型加载工厂函数
    """
    if backbone_name == "Video-LLaVA-7B":
        from models.video_llava_7b import VideoLLaVAWrapper
        return VideoLLaVAWrapper()
    elif backbone_name == "LLaVA-NeXT-Video-34B":
        # ⭐ 新增：34B模型支持
        from models.llava_next_34b import LLaVANext34BWrapper
        return LLaVANext34BWrapper()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="EventGraph-LLM Experiments")
    
    # 核心选择参数
    parser.add_argument("--dataset", type=str, required=True, choices=["VideoMME", "LongVideoBench", "MLUV"])
    parser.add_argument("--method", type=str, required=True, choices=["FastV", "ToMe", "DyCoke", "MovieChat", "SceneGraph-Cap", "Q-Frame", "Q-Frame-Clean", "EventGraph-LMM"])
    parser.add_argument("--backbone", type=str, default="Video-LLaVA-7B", choices=["Video-LLaVA-7B", "LLaVA-NeXT-Video-34B"])
    
    # --- [新增] 调试模式参数 ---
    parser.add_argument("--duration_mode", type=str, default="ideal", 
                        choices=["all", "ideal", "long", "short"], 
                        help="ideal: 10-30min (用于验证 Ours 优势), all: 跑顶会全量数据")
    
    # --- [新增] 并行分片参数 (用于多卡并行) ---
    parser.add_argument("--num_chunks", type=int, default=1, help="把数据集分成几份")
    parser.add_argument("--chunk_idx", type=int, default=0, help="当前跑第几份 (0 到 num_chunks-1)")
    # ---------------------------------------
    
    # 路径与超参
    parser.add_argument("--data_root", type=str, default="/root/hhq/dataset")
    parser.add_argument("--token_budget", type=int, default=2048)
    
    # Q-Frame等方法的温度参数
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for Gumbel-Max sampling in Q-Frame (default: 1.0)")
    
    # ⭐ 新增：限制样本数（用于快速测试）
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for quick testing (default: None for all)")
    
    # === DyCoke专用参数 (独立section) ===
    parser.add_argument("--dycoke_K", type=float, default=0.5,
                        help="DyCoke Stage 1 pruning rate (default: 0.5, paper Table 1)")
    parser.add_argument("--dycoke_L", type=int, default=3,
                        help="DyCoke Stage 2 attention evaluation layer (default: 3)")
    parser.add_argument("--dycoke_P", type=float, default=0.7,
                        help="DyCoke Stage 2 retention rate (default: 0.7, keep top 70%)")
    # =====================================
    
    # === ToMe专用参数 (独立section) ===
    parser.add_argument("--use_full_tome", action="store_true",
                        help="Use FULL ToMe (ViT-layer token merging) instead of simplified version")
    # =====================================
    
    # === GNU Parallel批处理模式参数 (可选，向后兼容) ===
    parser.add_argument("--batch_mode", action="store_true",
                        help="Enable batch processing mode for GNU Parallel dynamic load balancing")
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated sample indices for batch processing (e.g., '0,1,2,3,4')")
    # =====================================================
    
    # 输出目录
    parser.add_argument("--output_dir", type=str, default="./result")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载模型 (Backbone)
    # ⚠️ 特殊处理：FastV、DyCoke、ToMe(完整版)使用独立的model wrapper
    if args.method in ["FastV", "DyCoke"]:
        print(f"🚀 [1/4] Skipping main model load ({args.method} uses isolated model)...")
        model = None  # FastV/DyCoke会在__init__中加载自己的model
    elif args.method == "ToMe" and args.use_full_tome:
        # ⭐ ToMe完整版：使用VideoLLaVATomeWrapper
        print(f"🚀 [1/4] Loading ToMe-patched Backbone: {args.backbone}...")
        from models.video_llava_7b_tome import VideoLLaVATomeWrapper
        model = VideoLLaVATomeWrapper(
            model_path="/root/hhq/models/Video-LLaVA-7B-hf",
            token_budget=args.token_budget,
            num_frames=32
        )
        print(f"   ✅ VideoLLaVATomeWrapper loaded with token budget={args.token_budget}")
    else:
        print(f"🚀 [1/4] Loading Backbone: {args.backbone}...")
        model = load_model(args.backbone)
    
    # 2. 初始化方法 (Method)
    print(f"🛠️ [2/4] Initializing Method: {args.method}...")
    method_class = METHOD_REGISTRY[args.method]
    processor = method_class(args, model)
    
    # 3. 加载数据集
    print(f"📂 [3/4] Loading Dataset: {args.dataset} (Mode: {args.duration_mode})...")
    dataset_class = DATASET_REGISTRY[args.dataset]
    
    # --- [修改] 针对 VideoMME 传入 duration_mode ---
    if args.dataset == "VideoMME":
        # VideoMME 类支持 duration_mode 参数
        dataset = dataset_class(root_dir=args.data_root, duration_mode=args.duration_mode)
    else:
        # 其他数据集暂未实现该筛选功能，按默认加载
        dataset = dataset_class(root_dir=args.data_root)
    
    # === [新增] 限制样本数 (用于快速测试) ===
    # ⭐ 注意：必须先限制总样本数，再分片！
    if args.max_samples is not None:
        if hasattr(dataset, 'samples'):
            original_count = len(dataset.samples)
            dataset.samples = dataset.samples[:args.max_samples]
            print(f"🔬 [Testing Mode] Limited to {args.max_samples} samples (Original: {original_count})")
        else:
            print("⚠️ Warning: Dataset does not support max_samples limiting.")
    # ==========================================
    
    # === [新增] GNU Parallel批处理模式 (优先级高于chunk模式) ===
    if args.batch_mode and args.sample_indices:
        # 批处理模式：处理指定的样本索引列表
        indices = [int(i.strip()) for i in args.sample_indices.split(',') if i.strip()]
        if hasattr(dataset, 'samples'):
            total_samples = len(dataset.samples)
            # 过滤无效索引
            valid_indices = [i for i in indices if 0 <= i < total_samples]
            if len(valid_indices) < len(indices):
                print(f"⚠️ Warning: {len(indices) - len(valid_indices)} invalid indices filtered")
            
            dataset.samples = [dataset.samples[i] for i in valid_indices]
            print(f"🔋 [Batch Mode] Processing {len(valid_indices)} samples: {valid_indices[:5]}...")
        else:
            print("⚠️ Warning: Dataset does not support batch mode. Falling back to standard mode.")
    # === [原有] 数据分片逻辑 (用于多卡并行) ===
    elif args.num_chunks > 1:
        total_samples = len(dataset)
        chunk_size = total_samples // args.num_chunks
        start_idx = args.chunk_idx * chunk_size
        
        # 处理最后一份，确保包含所有剩余数据
        if args.chunk_idx == args.num_chunks - 1:
            end_idx = total_samples
        else:
            end_idx = (args.chunk_idx + 1) * chunk_size
        
        # 执行切片
        if hasattr(dataset, 'samples'):
            dataset.samples = dataset.samples[start_idx:end_idx]
            print(f"🔄 [Parallel] Running Chunk {args.chunk_idx}/{args.num_chunks}: Samples {start_idx} to {end_idx} (Total {len(dataset)})")
        else:
            print("⚠️ Warning: Dataset does not support list slicing. Sharding skipped.")
    # ==========================================

    # 4. 循环推理
    print(f"▶️ [4/4] Start Inference...")
    print(f"  📊 Total samples: {len(dataset)}")
    print(f"  🎯 Method: {args.method}")
    print(f"  💾 Results will be saved to: {args.output_dir}")
    print()
    
    results = []
    
    # 使用tqdm添加进度条
    from tqdm import tqdm
    
    for idx, sample in enumerate(tqdm(dataset, desc="Processing", unit="sample")):
        try:
            # sample 包含: {'video_path':..., 'question':..., 'options':...}
            pred_answer = processor.process_and_inference(
                sample['video_path'], 
                sample['question'], 
                sample.get('options', [])
            )
            
            results.append({
                "id": sample['id'],
                "pred": pred_answer,
                "gt": sample['answer']
            })
            
            # 打印结果（只打印前10个样本，避免输出过多）
            if idx < 10:
                tqdm.write(f"  ✓ Sample {sample['id']}: Pred={pred_answer} | GT={sample['answer']}")
            
        except Exception as e:
            tqdm.write(f"  ❌ Error in sample {sample['id']}: {e}")
            # 添加到结果中标记为错误
            results.append({
                "id": sample['id'],
                "pred": "ERROR",
                "gt": sample['answer'],
                "error": str(e)
            })
            # import traceback
            # traceback.print_exc() # 调试时可以取消注释查看详细报错
    
    print()        
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 文件名加上 duration_mode 和 chunk_idx (如果有分片) 方便区分
    suffix = ""
    if args.num_chunks > 1:
        suffix = f"_chunk{args.chunk_idx}"
        
    save_file = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_{args.duration_mode}_{args.backbone}{suffix}.json")
    
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 统计正确率
    correct = sum(1 for r in results if r.get('pred') == r.get('gt') and r.get('pred') != "ERROR")
    total = len([r for r in results if r.get('pred') != "ERROR"])
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"✅ Done! Results saved to {save_file}")
    print(f"📈 Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"❌ Errors: {len([r for r in results if r.get('pred') == 'ERROR'])}")

if __name__ == "__main__":
    main()