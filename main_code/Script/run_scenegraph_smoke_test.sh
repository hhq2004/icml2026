#!/bin/bash
# SceneGraph-Cap 冒烟测试脚本
# 使用 Video-LLaVA-7B 模型测试 SceneGraph-Cap 方法（4卡并行）

echo "======================================================================"
echo "SceneGraph-Cap Baseline 冒烟测试 (50 samples, 4-GPU parallel)"
echo "======================================================================"
echo "数据集: VideoMME (50 samples)"
echo "方法: SceneGraph-Cap"
echo "模型: Video-LLaVA-7B"
echo "Token Budget: 2048"
echo "======================================================================"
echo ""

cd /root/hhq/main_code

# 创建输出目录（带baseline名称）
mkdir -p ./result/scenegraph_cap_smoke_test

# 4卡并行运行
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python run_inference.py \
        --dataset VideoMME \
        --method SceneGraph-Cap \
        --backbone Video-LLaVA-7B \
        --data_root /root/hhq/dataset \
        --token_budget 2048 \
        --duration_mode all \
        --max_samples 50 \
        --num_chunks 4 \
        --chunk_idx $i \
        --output_dir ./result/scenegraph_cap_smoke_test &
done

echo "等待所有进程完成..."
wait

echo ""
echo "======================================================================"
echo "合并结果..."
echo "======================================================================"

# 合并4个chunk的结果（使用现有merge_results.py的参数格式）
python merge_results.py \
    --input_dir ./result/scenegraph_cap_smoke_test \
    --pattern "VideoMME_SceneGraph-Cap_all_Video-LLaVA-7B_chunk" \
    --output "./result/scenegraph_cap_smoke_test/VideoMME_SceneGraph-Cap_all_Video-LLaVA-7B_merged.json"

echo ""
echo "======================================================================"
echo "冒烟测试完成！"
echo "合并结果: ./result/scenegraph_cap_smoke_test/VideoMME_SceneGraph-Cap_all_Video-LLaVA-7B_merged.json"
echo "======================================================================"

# ⭐ 添加错误检测
echo ""
echo "======================================================================"
echo "🔍 检查日志中的错误..."
echo "======================================================================"

# 检查是否有generate()错误
if grep -q "attribute 'generate'" ./result/scenegraph_cap_smoke_test/*.log 2>/dev/null; then
    echo "❌ 警告: 检测到 generate() 方法错误！"
    echo ""
    echo "详细错误信息:"
    grep -n "AttributeError.*generate" ./result/scenegraph_cap_smoke_test/*.log 2>/dev/null | head -5
    echo ""
    echo "⚠️  模型推理失败！准确率可能是随机猜测的结果（~25%）"
    echo "   请检查模型加载是否使用了正确的类（VideoLlavaForConditionalGeneration）"
    echo ""
    echo "建议操作:"
    echo "  1. 检查 video_llava_7b.py 的模型加载部分"
    echo "  2. 查看完整日志: cat ./result/scenegraph_cap_smoke_test/gpu0.log | head -30"
else
    echo "✅ 未检测到 generate() 错误"
    echo "✅ 模型推理正常"
fi

echo "======================================================================"
