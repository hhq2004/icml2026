#!/bin/bash

# EventGraph-LMM - GNU Parallel动态负载均衡版本
# 基于成功的 run_eventgraph_200samples.sh，添加真正的动态调度

echo "========================================================================"
echo "EventGraph-LMM - GNU Parallel动态负载均衡 (全量VideoMME)"
echo "========================================================================"
echo "配置："
echo "  - 模型: Video-LLaVA-7B + CLIP-ViT-L/14"
echo "  - 方法: EventGraph-LMM (Graph-based Submodular Optimization)"
echo "  - 数据集: VideoMME (全量 ~2697样本)"
echo "  - Token Budget: 2048"
echo "  - 动态调度: GNU Parallel (谁完成谁接新chunk)"
echo "  - 预计时间: ~10-12小时"
echo "========================================================================"

# ============================================================================
# 配置区域
# ============================================================================
OUTPUT_DIR="../result/eventgraph_full"
DATASET="VideoMME"
DATA_ROOT="/root/hhq/dataset"
TOKEN_BUDGET=2048
METHOD="EventGraph-LMM"
BACKBONE="Video-LLaVA-7B"
MAX_SAMPLES=99999  # 不限制样本数，运行全量数据集
NUM_CHUNKS=108  # 将~2697样本分成108个chunk，每个chunk ~25样本 (108=4×27)

export TOKENIZERS_PARALLELISM=false
mkdir -p ${OUTPUT_DIR}

echo ""
echo "🔧 分片策略："
echo "  - 总样本: ${MAX_SAMPLES}"
echo "  - 分片数: ${NUM_CHUNKS}"
echo "  - 每片约: $((MAX_SAMPLES / NUM_CHUNKS)) 样本"
echo "  - 并行度: 4 GPUs (动态调度)"
echo ""

# 清理旧文件
rm -f ${OUTPUT_DIR}/*.log
rm -f ${OUTPUT_DIR}/*_chunk*.json

# ============================================================================
# 检查GNU Parallel
# ============================================================================
if ! command -v parallel &> /dev/null; then
    echo "⚠️  GNU Parallel未安装，使用静态分配模式"
    echo "    安装方法: sudo apt-get install parallel"
    echo ""
    echo "🚀 启动${NUM_CHUNKS}个chunk（静态模式）..."
    
    # 降级方案：静态分配
    for chunk_idx in $(seq 0 $((NUM_CHUNKS-1))); do
        gpu_id=$((chunk_idx % 4))
        CUDA_VISIBLE_DEVICES=$gpu_id python ../run_inference.py \
            --dataset ${DATASET} --method ${METHOD} --backbone ${BACKBONE} \
            --data_root ${DATA_ROOT} --token_budget ${TOKEN_BUDGET} \
            --duration_mode all \
            --max_samples ${MAX_SAMPLES} \
            --num_chunks ${NUM_CHUNKS} \
            --chunk_idx ${chunk_idx} \
            --output_dir ${OUTPUT_DIR} \
            > ${OUTPUT_DIR}/chunk${chunk_idx}.log 2>&1 &
    done
    
    echo "✅ ${NUM_CHUNKS}个chunk已启动，等待完成..."
    wait
    
else
    # ============================================================================
    # GNU Parallel 动态调度 (推荐)
    # ============================================================================
    echo "✅ 检测到GNU Parallel，使用动态调度模式"
    echo ""
    echo "🚀 启动GNU Parallel动态调度..."
    echo "  特性：谁完成谁接新chunk，确保4个GPU持续满载"
    echo ""
    
    # 生成chunk索引列表 (0到NUM_CHUNKS-1)
    seq 0 $((NUM_CHUNKS-1)) | parallel \
        --jobs 4 \
        --joblog ${OUTPUT_DIR}/parallel_joblog.txt \
        --bar \
        --halt soon,fail=1 \
        --line-buffer \
        '
        chunk_idx={}
        gpu_id=$((({%} - 1)))
        
        echo "🔄 [Chunk {}/'${NUM_CHUNKS}'] GPU ${gpu_id} 开始处理..."
        
        CUDA_VISIBLE_DEVICES=${gpu_id} python ../run_inference.py \
            --dataset '${DATASET}' \
            --method '${METHOD}' \
            --backbone '${BACKBONE}' \
            --data_root '${DATA_ROOT}' \
            --token_budget '${TOKEN_BUDGET}' \
            --duration_mode all \
            --max_samples '${MAX_SAMPLES}' \
            --num_chunks '${NUM_CHUNKS}' \
            --chunk_idx ${chunk_idx} \
            --output_dir '${OUTPUT_DIR}' \
            > '${OUTPUT_DIR}'/chunk${chunk_idx}.log 2>&1
        
        exit_code=$?
        if [ ${exit_code} -eq 0 ]; then
            echo "✅ [Chunk {}/'${NUM_CHUNKS}'] GPU ${gpu_id} 完成"
        else
            echo "❌ [Chunk {}/'${NUM_CHUNKS}'] GPU ${gpu_id} 失败 (exit code: ${exit_code})"
            exit 1
        fi
        '
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ GNU Parallel执行失败！"
        echo "请查看日志: ls -lh ${OUTPUT_DIR}/chunk*.log"
        exit 1
    fi
    
    echo ""
    echo "✅ GNU Parallel动态调度完成！"
    echo ""
    
    # 显示任务统计
    if [ -f "${OUTPUT_DIR}/parallel_joblog.txt" ]; then
        echo "📊 任务统计："
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 最慢的5个chunk
        echo "⏱️  最慢的5个chunk："
        tail -n +2 ${OUTPUT_DIR}/parallel_joblog.txt | \
            awk '{print $4"\t"$1}' | \
            sort -n -r | head -5 | \
            awk '{printf "  %.1fs\tChunk %d\n", $1, $2}'
        
        # 总体统计
        echo ""
        echo "📌 执行统计："
        total_chunks=$(tail -n +2 ${OUTPUT_DIR}/parallel_joblog.txt | wc -l)
        avg_time=$(tail -n +2 ${OUTPUT_DIR}/parallel_joblog.txt | awk '{sum+=$4; count++} END {printf "%.1f", sum/count}')
        echo "  总chunk数: ${total_chunks}"
        echo "  平均耗时: ${avg_time}s"
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
fi

echo ""
echo "✅ 所有chunk已完成！"

# ============================================================================
# 合并结果 (完全复制成功的run_eventgraph_200samples.sh逻辑)
# ============================================================================
echo ""
echo "========================================================================"
echo "✅ 正在合并结果..."
echo "========================================================================"

# 检查是否有chunk文件
CHUNK_JSON_COUNT=$(ls ${OUTPUT_DIR}/*_chunk*.json 2>/dev/null | wc -l)

if [ ${CHUNK_JSON_COUNT} -eq 0 ]; then
    echo "⚠️  警告：未找到任何chunk JSON文件！"
    echo ""
    echo "🔍 诊断信息："
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 检查log文件
    LOG_COUNT=$(ls ${OUTPUT_DIR}/chunk*.log 2>/dev/null | wc -l)
    echo "📝 Chunk日志文件数: ${LOG_COUNT}"
    
    if [ ${LOG_COUNT} -gt 0 ]; then
        echo ""
        echo "📋 检查前3个chunk的日志错误..."
        for log_file in $(ls ${OUTPUT_DIR}/chunk*.log 2>/dev/null | head -3); do
            chunk_name=$(basename ${log_file})
            echo ""
            echo "--- ${chunk_name} ---"
            
            # 检查错误
            if grep -q "Error\|Traceback\|Failed" ${log_file} 2>/dev/null; then
                echo "❌ 发现错误："
                grep -A 5 "Error\|Traceback\|Failed" ${log_file} | head -15
            else
                echo "ℹ️  显示最后10行："
                tail -10 ${log_file}
            fi
        done
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "💡 建议操作："
    echo "  1. 查看完整日志: cat ${OUTPUT_DIR}/chunk0.log"
    echo "  2. 检查run_inference.py是否正常运行"
    echo ""
    exit 1
fi

# 合并所有chunk (使用成功的合并逻辑)
python -c "
import json
import os

output_dir = '${OUTPUT_DIR}'

# 读取所有chunk文件
chunks = []
for i in range(${NUM_CHUNKS}):
    chunk_file = f'{output_dir}/VideoMME_${METHOD}_all_${BACKBONE}_chunk{i}.json'
    try:
        with open(chunk_file, 'r') as f:
            chunks.extend(json.load(f))
    except FileNotFoundError:
        print(f'Warning: {chunk_file} not found')

# 保存合并结果
with open(f'{output_dir}/VideoMME_${METHOD}_all_${BACKBONE}_merged.json', 'w') as f:
    json.dump(chunks, f, indent=4)

# 计算准确率
correct = sum(1 for r in chunks if r.get('pred') == r.get('gt'))
total = len(chunks)
if total > 0:
    accuracy = 100 * correct / total
    print(f'\n📊 EventGraph-LMM (GNU Parallel动态调度) 结果:')
    print(f'  - 样本数: {total}')
    print(f'  - 正确数: {correct}')
    print(f'  - 准确率: {accuracy:.2f}%')
else:
    print('\n⚠️  没有找到任何结果！')
"

echo ""
echo "结果文件: ${OUTPUT_DIR}/VideoMME_${METHOD}_all_${BACKBONE}_merged.json"
echo "========================================================================"
