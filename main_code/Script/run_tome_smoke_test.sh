#!/bin/bash

# ToMe FULL版 (完整ViT层token merging) - 7B模型 4卡数据并行测试 (50样本)
# 🆕 使用 --use_full_tome 启用真实的ViT层token merging

echo "========================================================================"
echo "ToMe FULL版 - 7B模型 4卡数据并行测试"
echo "========================================================================"
echo "配置："
echo "  -模型: Video-LLaVA-7B (with ToMe injection)"
echo "  - 方法: ToMe FULL (ViT-layer token merging, ICLR 2023)"
echo "  - 数据集: VideoMME (50样本)"
echo "  - Token Budget: 2048"
echo "  - 初始帧数: 32 frames"
echo "  - Initial tokens: 32×256 = 8192"
echo "  - Vision layers: 24"
echo "  - Tokens merged per layer: 256"
echo "  - Final tokens: 2048"
echo "  - 并行: 4×A100 数据并行"
echo "  - 预计时间: 8-10分钟 (略慢于简化版，因为layer-wise merging)"
echo "========================================================================"

# 设置通用环境变量
export TOKENIZERS_PARALLELISM=false

# 创建输出目录
mkdir -p ../result/tome_full_50samples_7b_parallel

# 并行启动4个进程
echo "🚀 启动4卡并行（FULL ToMe模式）..."

# GPU 0
CUDA_VISIBLE_DEVICES=0 python ../run_inference.py \
    --dataset VideoMME \
    --method ToMe \
    --backbone Video-LLaVA-7B \
    --use_full_tome \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 0 \
    --output_dir ../result/tome_full_50samples_7b_parallel \
    > ../result/tome_full_50samples_7b_parallel/gpu0.log 2>&1 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python ../run_inference.py \
    --dataset VideoMME \
    --method ToMe \
    --backbone Video-LLaVA-7B \
    --use_full_tome \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 1 \
    --output_dir ../result/tome_full_50samples_7b_parallel \
    > ../result/tome_full_50samples_7b_parallel/gpu1.log 2>&1 &

# GPU 2
CUDA_VISIBLE_DEVICES=2 python ../run_inference.py \
    --dataset VideoMME \
    --method ToMe \
    --backbone Video-LLaVA-7B \
    --use_full_tome \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 2 \
    --output_dir ../result/tome_full_50samples_7b_parallel \
    > ../result/tome_full_50samples_7b_parallel/gpu2.log 2>&1 &

# GPU 3
CUDA_VISIBLE_DEVICES=3 python ../run_inference.py \
    --dataset VideoMME \
    --method ToMe \
    --backbone Video-LLaVA-7B \
    --use_full_tome \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 3 \
    --output_dir ../result/tome_full_50samples_7b_parallel \
    > ../result/tome_full_50samples_7b_parallel/gpu3.log 2>&1 &

echo "✅ 4个进程已启动，后台运行中..."
echo "📝 日志文件："
echo "  - GPU 0: ../result/tome_full_50samples_7b_parallel/gpu0.log"
echo "  - GPU 1: ../result/tome_full_50samples_7b_parallel/gpu1.log"
echo "  - GPU 2: ../result/tome_full_50samples_7b_parallel/gpu2.log"
echo "  - GPU 3: ../result/tome_full_50samples_7b_parallel/gpu3.log"
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

# 读取所有chunk文件
chunks = []
for i in range(4):
    chunk_file = f'../result/tome_full_50samples_7b_parallel/VideoMME_ToMe_all_Video-LLaVA-7B_chunk{i}.json'
    try:
        with open(chunk_file, 'r') as f:
            chunks.extend(json.load(f))
    except FileNotFoundError:
        print(f'Warning: {chunk_file} not found')

# 保存合并结果
with open('../result/tome_full_50samples_7b_parallel/VideoMME_ToMe_FULL_all_Video-LLaVA-7B_merged.json', 'w') as f:
    json.dump(chunks, f, indent=4)

# 计算准确率
correct = sum(1 for r in chunks if r.get('pred') == r.get('gt'))
total = len(chunks)
print(f'\n📊 合并结果统计:')
print(f'  - 总样本数: {total}')
print(f'  - 正确数: {correct}')
print(f'  - 准确率: {100*correct/total:.2f}%')
print(f'\n🎯 预期：准确率应≥38% (简化版32% + 6%提升)')
"

echo ""
echo "结果文件: ../result/tome_full_50samples_7b_parallel/VideoMME_ToMe_FULL_all_Video-LLaVA-7B_merged.json"

echo ""
echo "========================================================================"
echo "🔍 检查日志中的错误..."
echo "========================================================================"

# 检查错误
if grep -q "Error" ../result/tome_full_50samples_7b_parallel/*.log 2>/dev/null; then
    echo "⚠️  检测到错误，请查看日志文件"
else
    echo "✅ 未检测到错误"
fi

echo "========================================================================"
