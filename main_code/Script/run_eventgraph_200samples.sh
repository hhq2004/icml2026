#!/bin/bash

# EventGraph-LMM (Ours) - 7B模型 4卡数据并行测试
# ICML 2026 - EventGraph-LMM: Submodular Information Maximization for Efficient Long-Video Understanding
# 支持动态样本数扩展：50 → 200 → 500 → 1000+

# 配置样本数（修改这里即可扩展测试）
# SAMPLE_COUNT=700  # 注释掉表示运行全部样本
SAMPLE_COUNT="ALL"  # 完整VideoMME数据集 (~2697 samples)

echo "========================================================================"
echo "EventGraph-LMM (Ours) - 7B模型 4卡数据并行测试 (完整VideoMME)"
echo "========================================================================"
echo "配置："
echo "  - 模型: Video-LLaVA-7B + CLIP-ViT-L/14"
echo "  - 方法: EventGraph-LMM (Graph-based Submodular Optimization)"
echo "  - 数据集: VideoMME (完整数据集 ~2697样本)"
echo "  - Token Budget: 2048"
echo "  - 算法设置:"
echo "      * τ (temporal threshold) = 30s"
echo "      * δ (similarity threshold) = 0.65"
echo "      * α (PageRank restart) = 0.15"
echo "      * λ (trade-off) = 1.0"
echo "  - 并行: 4×A100 数据并行 (~675样本/卡)"
echo "  - 机制: Graph Construction + CELF Selection + Graph-CoT"
echo "  - 预计时间: ~10-12小时"
echo "========================================================================"

# 设置通用环境变量
export TOKENIZERS_PARALLELISM=false

# 统一输出目录（所有测试覆盖写入）
OUTPUT_DIR="../result/eventgraph_test"
mkdir -p ${OUTPUT_DIR}

# 并行启动4个进程
echo "🚀 启动4卡并行（EventGraph-LMM模式）..."

# GPU 0 - 处理chunk 0/4
CUDA_VISIBLE_DEVICES=0 python ../run_inference.py \
    --dataset VideoMME \
    --method EventGraph-LMM \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --num_chunks 4 \
    --chunk_idx 0 \
    --output_dir ${OUTPUT_DIR} \
    > ${OUTPUT_DIR}/gpu0.log 2>&1 &

# GPU 1 - 处理chunk 1/4
CUDA_VISIBLE_DEVICES=1 python ../run_inference.py \
    --dataset VideoMME \
    --method EventGraph-LMM \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --num_chunks 4 \
    --chunk_idx 1 \
    --output_dir ${OUTPUT_DIR} \
    > ${OUTPUT_DIR}/gpu1.log 2>&1 &

# GPU 2 - 处理chunk 2/4
CUDA_VISIBLE_DEVICES=2 python ../run_inference.py \
    --dataset VideoMME \
    --method EventGraph-LMM \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --num_chunks 4 \
    --chunk_idx 2 \
    --output_dir ${OUTPUT_DIR} \
    > ${OUTPUT_DIR}/gpu2.log 2>&1 &

# GPU 3 - 处理chunk 3/4
CUDA_VISIBLE_DEVICES=3 python ../run_inference.py \
    --dataset VideoMME \
    --method EventGraph-LMM \
    --backbone Video-LLaVA-7B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --duration_mode all \
    --num_chunks 4 \
    --chunk_idx 3 \
    --output_dir ${OUTPUT_DIR} \
    > ${OUTPUT_DIR}/gpu3.log 2>&1 &

echo "✅ 4个进程已启动，后台运行中..."
echo "📝 日志文件："
echo "  - GPU 0: ${OUTPUT_DIR}/gpu0.log"
echo "  - GPU 1: ${OUTPUT_DIR}/gpu1.log"
echo "  - GPU 2: ${OUTPUT_DIR}/gpu2.log"
echo "  - GPU 3: ${OUTPUT_DIR}/gpu3.log"
echo ""
echo "⏳ 实时监控进度 (每10秒更新)..."
echo ""

# 实时进度监控
sleep 5  # 等待进程启动

while true; do
    # 检查所有后台进程是否还在运行
    running_count=0
    for pid in $(jobs -p); do
        if kill -0 $pid 2>/dev/null; then
            ((running_count++))
        fi
    done
    
    # 如果所有进程都结束了，退出循环
    if [ $running_count -eq 0 ]; then
        break
    fi
    
    # 统计各GPU的进度
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 进度更新 ($(date '+%H:%M:%S'))"
    
    for i in {0..3}; do
        logfile="${OUTPUT_DIR}/gpu$i.log"
        if [ -f "$logfile" ]; then
            # 提取处理进度
            progress=$(tail -20 "$logfile" | grep -oP 'Processing:\s+\K[0-9]+%' | tail -1)
            if [ -z "$progress" ]; then
                progress="启动中..."
            fi
            
            # 提取准确率 (如果有)
            accuracy=$(tail -5 "$logfile" | grep -oP 'Accuracy: \K[0-9.]+%' | tail -1)
            if [ -n "$accuracy" ]; then
                echo "  GPU $i: ✅ 完成 (准确率: $accuracy)"
            else
                echo "  GPU $i: $progress"
            fi
        else
            echo "  GPU $i: 等待启动..."
        fi
    done
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # 等待10秒后再次检查
    sleep 10
done

echo ""
echo "✅ 所有进程已完成！"

echo ""
echo "========================================================================"
echo "✅ 所有进程完成！正在合并结果..."
echo "========================================================================"

# 合并4个chunk的结果
python -c "
import json
import os

output_dir = '${OUTPUT_DIR}'

# 读取所有chunk文件
chunks = []
for i in range(4):
    chunk_file = f'{output_dir}/VideoMME_EventGraph-LMM_all_Video-LLaVA-7B_chunk{i}.json'
    try:
        with open(chunk_file, 'r') as f:
            chunks.extend(json.load(f))
    except FileNotFoundError:
        print(f'Warning: {chunk_file} not found')

# 保存合并结果
with open(f'{output_dir}/VideoMME_EventGraph-LMM_all_Video-LLaVA-7B_merged.json', 'w') as f:
    json.dump(chunks, f, indent=4)

# 计算准确率
correct = sum(1 for r in chunks if r.get('pred') == r.get('gt'))
total = len(chunks)
if total > 0:
    accuracy = 100 * correct / total
    print(f'\n📊 EventGraph-LMM 完整VideoMME测试结果:')
    print(f'  - 样本数: {total}')
    print(f'  - 正确数: {correct}')
    print(f'  - 准确率: {accuracy:.2f}%')
    
    # 显示历史对比
    print(f'\n📈 准确率趋势 (样本数扩展):')
    print(f'  - 50样本:   36.00%')
    print(f'  - 200样本:  30.50%')
    print(f'  - 700样本:  33.86%')
    print(f'  - {total}样本: {accuracy:.2f}%  ← 完整数据集')
else:
    print('\n⚠️  没有找到任何结果文件！')
"

echo ""
echo "结果文件: ${OUTPUT_DIR}/VideoMME_EventGraph-LMM_all_Video-LLaVA-7B_merged.json"

echo ""
echo "========================================================================"
# 分别检测warning和真实error
has_real_error=false
has_warning=false

for logfile in ${OUTPUT_DIR}/gpu*.log; do
    if [ -f "$logfile" ]; then
        # 检查真正的Traceback错误
        if grep -q "Traceback (most recent call last)" "$logfile" 2>/dev/null; then
            has_real_error=true
            break
        fi
        
        # 检查致命错误 (但排除统计行和warning)
        if grep -E "(RuntimeError|CUDA error|ImportError|AttributeError|ValueError)" "$logfile" 2>/dev/null | \
           grep -v "Vision tower not found" | \
           grep -v "torch_dtype is deprecated" | \
           grep -v "⚠️" | \
           grep -v "❌ Errors:" | \
           grep -v "Sample.*Error" | \
           grep -q "Error"; then
            has_real_error=true
            break
        fi
        
        # 检测warning
        if grep -q "⚠️" "$logfile" 2>/dev/null; then
            has_warning=true
        fi
    fi
done

# 输出检测结果
if [ "$has_real_error" = true ]; then
    echo "❌ 检测到真实错误！推理失败"
    echo "  查看详细: tail -100 ${OUTPUT_DIR}/gpu0.log"
elif [ "$has_warning" = true ]; then
    echo "ℹ️  推理完成，有warning但可忽略"
    echo "  常见warning: Vision tower not found, torch_dtype deprecated"
else
    echo "✅ 推理完成，无错误无warning"
fi
echo "========================================================================"
