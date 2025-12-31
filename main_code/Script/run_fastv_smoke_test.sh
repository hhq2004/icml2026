#!/bin/bash

# FastV 7B模型 - 4卡数据并行测试 (50样本)
# FastV (ECCV 2024) baseline

echo "========================================================================"
echo "FastV Baseline - 7B模型 4卡数据并行测试"
echo "========================================================================"
echo "配置："
echo "  - 模型: Video-LLaVA-7B"
echo "  - 方法: FastV (ECCV 2024)"
echo "  - 数据集: VideoMME (50样本)"
echo "  - Token Budget: 2048"
echo "  - Filtering Layer K: 2 (论文推荐)"
echo "  - Filtering Ratio R: 50% (论文推荐)"
echo "  - 初始帧数: ~8 frames"
echo "  - Tokens per frame: 256 (7B模型)"
echo "  - 剪枝后tokens: ~1024 (50% pruning)"
echo "  - 并行: 4×A100 数据并行"
echo "  - 预计时间: 6-7分钟"
echo "========================================================================"

# 设置通用环境变量
export TOKENIZERS_PARALLELISM=false

# 创建输出目录
mkdir -p ../result/fastv_50samples_7b_parallel

# 并行启动4个进程，每个使用不同GPU处理不同数据块
echo "🚀 启动4卡并行..."

# GPU 0: chunk 0/4 (样本1-13)
CUDA_VISIBLE_DEVICES=0 python ../run_inference.py \
    --dataset VideoMME \
    --method FastV \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 0 \
    --output_dir ../result/fastv_50samples_7b_parallel \
    > ../result/fastv_50samples_7b_parallel/gpu0.log 2>&1 &

# GPU 1: chunk 1/4 (样本14-26)
CUDA_VISIBLE_DEVICES=1 python ../run_inference.py \
    --dataset VideoMME \
    --method FastV \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 1 \
    --output_dir ../result/fastv_50samples_7b_parallel \
    > ../result/fastv_50samples_7b_parallel/gpu1.log 2>&1 &

# GPU 2: chunk 2/4 (样本27-39)
CUDA_VISIBLE_DEVICES=2 python ../run_inference.py \
    --dataset VideoMME \
    --method FastV \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 2 \
    --output_dir ../result/fastv_50samples_7b_parallel \
    > ../result/fastv_50samples_7b_parallel/gpu2.log 2>&1 &

# GPU 3: chunk 3/4 (样本40-50)
CUDA_VISIBLE_DEVICES=3 python ../run_inference.py \
    --dataset VideoMME \
    --method FastV \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 3 \
    --output_dir ../result/fastv_50samples_7b_parallel \
    > ../result/fastv_50samples_7b_parallel/gpu3.log 2>&1 &

echo "✅ 4个进程已启动，后台运行中..."
echo "📝 日志文件："
echo "  - GPU 0: ../result/fastv_50samples_7b_parallel/gpu0.log"
echo "  - GPU 1: ../result/fastv_50samples_7b_parallel/gpu1.log"
echo "  - GPU 2: ../result/fastv_50samples_7b_parallel/gpu2.log"
echo "  - GPU 3: ../result/fastv_50samples_7b_parallel/gpu3.log"
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
    chunk_file = f'../result/fastv_50samples_7b_parallel/VideoMME_FastV_all_Video-LLaVA-7B_chunk{i}.json'
    try:
        with open(chunk_file, 'r') as f:
            chunks.extend(json.load(f))
    except FileNotFoundError:
        print(f'Warning: {chunk_file} not found')

# 保存合并结果
with open('../result/fastv_50samples_7b_parallel/VideoMME_FastV_all_Video-LLaVA-7B_merged.json', 'w') as f:
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
echo "结果文件: ../result/fastv_50samples_7b_parallel/VideoMME_FastV_all_Video-LLaVA-7B_merged.json"

# 错误检测与显示
echo ""
echo "========================================================================"
echo "🔍 检查日志中的错误..."
echo "========================================================================"

# 检查是否有ValueError (attn_implementation错误)
if grep -q "ValueError.*attn_implementation" ../result/fastv_50samples_7b_parallel/*.log 2>/dev/null; then
    echo "❌ 检测到 attn_implementation 错误！"
    echo ""
    echo "错误详情:"
    grep -A 2 "ValueError.*attn_implementation" ../result/fastv_50samples_7b_parallel/gpu0.log | head -10
    echo ""
    echo "⚠️  Model需要使用 attn_implementation='eager'"
    echo "   请检查 video_llava_7b_fastv.py 是否已添加该参数"
    ERROR_FOUND=1
fi

# 检查是否有其他general错误
if grep -q "ERROR in FastV" ../result/fastv_50samples_7b_parallel/*.log 2>/dev/null; then
    echo "❌ 检测到 FastV处理错误！"
    echo ""
    echo "错误详情 (前20行):"
    grep -A 10 "ERROR in FastV" ../result/fastv_50samples_7b_parallel/gpu0.log | head -20
    echo ""
    ERROR_FOUND=1
fi

# 检查是否有generate()错误
if grep -q "attribute 'generate'" ../result/fastv_50samples_7b_parallel/*.log 2>/dev/null; then
    echo "❌ 检测到 generate() 方法错误！"
    echo ""
    echo "详细错误信息:"
    grep -n "AttributeError.*generate" ../result/fastv_50samples_7b_parallel/*.log 2>/dev/null | head -5
    echo ""
    echo "⚠️  模型推理失败！准确率可能是随机猜测的结果（~25%）"
    echo "   请检查模型加载是否使用了正确的类（VideoLlavaForConditionalGeneration）"
    ERROR_FOUND=1
fi

# 检查是否有Traceback
if grep -q "Traceback (most recent call last)" ../result/fastv_50samples_7b_parallel/*.log 2>/dev/null; then
    echo "⚠️  检测到Python异常追踪 (Traceback)"
    echo ""
    echo "异常类型统计:"
    grep -oP "(?<=Error type: ).*" ../result/fastv_50samples_7b_parallel/gpu0.log | sort | uniq -c || echo "  (无法提取)"
    echo ""
    echo "查看完整错误，请运行:"
    echo "  cat ../result/fastv_50samples_7b_parallel/gpu0.log | grep -A 15 'ERROR in'"
    ERROR_FOUND=1
fi

# 如果没有错误
if [ -z "$ERROR_FOUND" ]; then
    echo "✅ 未检测到明显错误"
    echo "✅ FastV执行正常"
else
    echo ""
    echo "========================================================================" 
    echo "⚠️  发现错误！请查看上方详情"
    echo "完整日志文件:"
    echo "  - ../result/fastv_50samples_7b_parallel/gpu0.log"
    echo "  - ../result/fastv_50samples_7b_parallel/gpu1.log"
    echo "  - ../result/fastv_50samples_7b_parallel/gpu2.log"
    echo "  - ../result/fastv_50samples_7b_parallel/gpu3.log"
fi

echo "========================================================================"
