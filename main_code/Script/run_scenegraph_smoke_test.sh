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
