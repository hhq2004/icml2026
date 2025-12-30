#!/bin/bash

# Q-Frame 7B模型 - 4卡数据并行测试 (50样本)
# 每张GPU处理12-13个样本，预计6-7分钟完成

echo "========================================================================"
echo "Q-Frame Baseline - 7B模型 4卡数据并行测试"
echo "========================================================================"
echo "配置："
echo "  - 模型: Video-LLaVA-7B"
echo "  - 方法: Q-Frame-Clean"
echo "  - 数据集: VideoMME (50样本)"
echo "  - Token Budget: 2048"
echo "  - 帧数K: 8 (7B模型: 256 tokens/frame → 2048/256=8)"
echo "  - 并行: 4×A100 数据并行"
echo "  - 预计时间: 6-7分钟"
echo "========================================================================"

# 设置通用环境变量
export TOKENIZERS_PARALLELISM=false

# 创建输出目录
mkdir -p ./result/test_50samples_7b_parallel

# 并行启动4个进程，每个使用不同GPU处理不同数据块
echo "🚀 启动4卡并行..."

# GPU 0: chunk 0/4 (样本1-13)
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --dataset VideoMME \
    --method Q-Frame-Clean \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --temperature 1.0 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 0 \
    --output_dir ./result/test_50samples_7b_parallel \
    > ./result/test_50samples_7b_parallel/gpu0.log 2>&1 &

# GPU 1: chunk 1/4 (样本14-26)
CUDA_VISIBLE_DEVICES=1 python run_inference.py \
    --dataset VideoMME \
    --method Q-Frame-Clean \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --temperature 1.0 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 1 \
    --output_dir ./result/test_50samples_7b_parallel \
    > ./result/test_50samples_7b_parallel/gpu1.log 2>&1 &

# GPU 2: chunk 2/4 (样本27-39)
CUDA_VISIBLE_DEVICES=2 python run_inference.py \
    --dataset VideoMME \
    --method Q-Frame-Clean \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --temperature 1.0 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 2 \
    --output_dir ./result/test_50samples_7b_parallel \
    > ./result/test_50samples_7b_parallel/gpu2.log 2>&1 &

# GPU 3: chunk 3/4 (样本40-50)
CUDA_VISIBLE_DEVICES=3 python run_inference.py \
    --dataset VideoMME \
    --method Q-Frame-Clean \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --temperature 1.0 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 3 \
    --output_dir ./result/test_50samples_7b_parallel \
    > ./result/test_50samples_7b_parallel/gpu3.log 2>&1 &

echo "✅ 4个进程已启动，后台运行中..."
echo "📝 日志文件："
echo "  - GPU 0: ./result/test_50samples_7b_parallel/gpu0.log"
echo "  - GPU 1: ./result/test_50samples_7b_parallel/gpu1.log"
echo "  - GPU 2: ./result/test_50samples_7b_parallel/gpu2.log"
echo "  - GPU 3: ./result/test_50samples_7b_parallel/gpu3.log"
echo ""
echo "⏳ 等待所有进程完成..."

# 等待所有后台进程完成
wait

echo ""
echo "========================================================================"
echo "✅ 所有进程完成！正在合并结果..."
echo "========================================================================"

# 合并4个chunk的结果
python -c "
import json
import glob

# 读取所有chunk文件
chunks = []
for i in range(4):
    chunk_file = f'./result/test_50samples_7b_parallel/VideoMME_Q-Frame-Clean_all_Video-LLaVA-7B_chunk{i}.json'
    try:
        with open(chunk_file, 'r') as f:
            chunks.extend(json.load(f))
    except FileNotFoundError:
        print(f'Warning: {chunk_file} not found')

# 保存合并结果
with open('./result/test_50samples_7b_parallel/VideoMME_Q-Frame-Clean_all_Video-LLaVA-7B_merged.json', 'w') as f:
    json.dump(chunks, f, indent=4)

# 计算准确率
correct = sum(1 for r in chunks if r.get('pred') == r.get('gt'))
total = len(chunks)
print(f'\n📊 合并结果统计:')
print(f'  - 总样本数: {total}')
print(f'  - 正确数: {correct}')
print(f'  - 准确率: {100*correct/total:.2f}%')
"

echo ""
echo "结果文件: ./result/test_50samples_7b_parallel/VideoMME_Q-Frame-Clean_all_Video-LLaVA-7B_merged.json"
echo "========================================================================"
