#!/bin/bash
# Q-Frame 冒烟测试脚本
# 使用 LLaVA-NeXT-Video-34B 模型测试 Q-Frame-Clean 方法

echo "======================================================================"
echo "Q-Frame Baseline 冒烟测试"
echo "======================================================================"
echo "数据集: VideoMME (8 samples)"
echo "方法: Q-Frame-Clean"
echo "模型: LLaVA-NeXT-Video-34B"
echo "======================================================================"
echo ""

cd /root/hhq/main_code

python run_inference.py \
    --dataset VideoMME \
    --method Q-Frame-Clean \
    --backbone LLaVA-NeXT-Video-34B \
    --data_root /root/hhq/dataset \
    --token_budget 2048 \
    --temperature 1.0 \
    --duration_mode all \
    --output_dir ./result/smoke_test

echo ""
echo "======================================================================"
echo "冒烟测试完成！"
echo "结果文件: ./result/smoke_test/VideoMME_Q-Frame-Clean_all_LLaVA-NeXT-Video-34B.json"
echo "======================================================================"
