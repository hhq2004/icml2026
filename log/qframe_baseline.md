# Q-Frame Baseline完成记录
## 配置
- 实现文件: methods/q_frame_clean.py
- 模型: Video-LLaVA-7B-hf
- K值: 7B=8帧 (2048 tokens), 34B=3帧 (1728 tokens)
- CLIP: clip-vit-large-patch14
- 温度: τ=1.0
## 测试结果
- 50样本测试: 13/50 = 26%准确率
- 测试时间: 2025-12-30
- 4卡并行: 正常
## 依赖
- run_inference.py
- models/video_llava.py (7B)
- models/llava_next_34b.py (34B)
- datasets/videomme.py