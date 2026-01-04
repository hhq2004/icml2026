#!/bin/bash

# DyCoke 7B模型 - 4卡数据并行冒烟测试 (50样本)
# ⭐ Stage 1 TTM 实现 - 使用 Monkey Patching 方案
# ⚠️ 注意: 仅实现 Stage 1 TTM, Stage 2 ATM 未实现

echo "========================================================================"
echo "DyCoke Stage 1 TTM - 7B模型 4卡数据并行冒烟测试"
echo "========================================================================"
echo "配置："
echo "  - 模型: Video-LLaVA-7B (VideoLlavaForConditionalGeneration)"
echo "  - 方法: DyCoke (Monkey Patching multi_modal_projector)"
echo "  - 数据集: VideoMME (50样本)"
echo "  - Stage 1 TTM: K=0.5 (100% 官方算法复现)"
echo "  - ✅ Stage 1: Temporal Token Merging (已实现)"
echo "  - ❌ Stage 2: Dynamic KV Pruning (需修改transformers,未实现)"
echo "  - 并行: 4×A100 数据并行"
echo "  - 预期token压缩: 8192 → ~4096 (50%)"
echo "  - 预计时间: ~15-20分钟"
echo "========================================================================"

# 设置通用环境变量
export TOKENIZERS_PARALLELISM=false

# 创建输出目录
mkdir -p ../result/dycoke_smoke_test

# 并行启动4个进程，每个使用不同GPU处理不同数据块
echo "🚀 启动4卡并行..."

# GPU 0: chunk 0/4 (样本1-13)
CUDA_VISIBLE_DEVICES=0 python ../run_inference.py \
    --dataset VideoMME \
    --method DyCoke \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --dycoke_K 0.5 \
    --dycoke_L 3 \
    --dycoke_P 0.7 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 0 \
    --output_dir ../result/dycoke_smoke_test \
    > ../result/dycoke_smoke_test/gpu0.log 2>&1 &

# GPU 1: chunk 1/4 (样本14-26)
CUDA_VISIBLE_DEVICES=1 python ../run_inference.py \
    --dataset VideoMME \
    --method DyCoke \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --dycoke_K 0.5 \
    --dycoke_L 3 \
    --dycoke_P 0.7 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 1 \
    --output_dir ../result/dycoke_smoke_test \
    > ../result/dycoke_smoke_test/gpu1.log 2>&1 &

# GPU 2: chunk 2/4 (样本27-39)
CUDA_VISIBLE_DEVICES=2 python ../run_inference.py \
    --dataset VideoMME \
    --method DyCoke \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --dycoke_K 0.5 \
    --dycoke_L 3 \
    --dycoke_P 0.7 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 2 \
    --output_dir ../result/dycoke_smoke_test \
    > ../result/dycoke_smoke_test/gpu2.log 2>&1 &

# GPU 3: chunk 3/4 (样本40-50)
CUDA_VISIBLE_DEVICES=3 python ../run_inference.py \
    --dataset VideoMME \
    --method DyCoke \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --dycoke_K 0.5 \
    --dycoke_L 3 \
    --dycoke_P 0.7 \
    --duration_mode all \
    --max_samples 50 \
    --num_chunks 4 \
    --chunk_idx 3 \
    --output_dir ../result/dycoke_smoke_test \
    > ../result/dycoke_smoke_test/gpu3.log 2>&1 &

echo "✅ 4个进程已启动，后台运行中..."
echo "📝 日志文件："
echo "  - GPU 0: ../result/dycoke_smoke_test/gpu0.log"
echo "  - GPU 1: ../result/dycoke_smoke_test/gpu1.log"
echo "  - GPU 2: ../result/dycoke_smoke_test/gpu2.log"
echo "  - GPU 3: ../result/dycoke_smoke_test/gpu3.log"
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
    chunk_file = f'../result/dycoke_smoke_test/VideoMME_DyCoke_all_Video-LLaVA-7B_chunk{i}.json'
    try:
        with open(chunk_file, 'r') as f:
            chunks.extend(json.load(f))
    except FileNotFoundError:
        print(f'Warning: {chunk_file} not found')

# 保存合并结果
with open('../result/dycoke_smoke_test/VideoMME_DyCoke_all_Video-LLaVA-7B_merged.json', 'w') as f:
    json.dump(chunks, f, indent=4)

# 计算准确率
correct = sum(1 for r in chunks if r.get('pred') == r.get('gt'))
total = len(chunks)
if total > 0:
    print(f'\\n📊 合并结果统计:')
    print(f'  - 总样本数: {total}')
    print(f'  - 正确数: {correct}')
    print(f'  - 准确率: {100*correct/total:.2f}%')
    print(f'\\n📌 论文参考值 (VideoMME, 7B, K=0.5):')
    print(f'  - 论文Table 1: 61.4%')
else:
    print('\\n⚠️  没有找到任何结果文件！')
"

echo ""
echo "结果文件: ../result/dycoke_smoke_test/VideoMME_DyCoke_all_Video-LLaVA-7B_merged.json"

# 错误检测
echo ""
echo "========================================================================"
echo "🔍 检查日志中的错误..."
echo "========================================================================"

# 检查是否有关键错误
if grep -q "ERROR\|Error\|Failed" ../result/dycoke_smoke_test/*.log 2>/dev/null; then
    echo "⚠️  检测到可能的错误，请查看日志："
    echo ""
    grep -n "ERROR\|Error\|Failed" ../result/dycoke_smoke_test/*.log 2>/dev/null | head -10
    echo ""
    echo "详细日志: cat ../result/dycoke_smoke_test/gpu0.log"
else
    echo "✅ 未检测到明显错误"
fi

echo "========================================================================"
echo "✅ DyCoke冒烟测试完成！"
echo "========================================================================"
echo ""
echo "📋 下一步："
echo "  1. 检查准确率是否合理（参考论文Table 1: VideoMME 61.4%）"
echo "  2. 创建测试报告: log/DyCoke_7B_smoke_test_20260101.md"
echo "  3. 对比其他baselines的结果"
echo "  4. 如果测试通过，准备跑全量数据"
echo ""
