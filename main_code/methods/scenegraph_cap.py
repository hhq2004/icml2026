# /root/hhq/main_code/methods/scenegraph_cap.py
import torch
import numpy as np
from decord import VideoReader, cpu
from .base_method import BaseMethod

class SceneGraphCap(BaseMethod):
    def __init__(self, args, model):
        super().__init__(args, model)
        # Captioning Paradigm: Sacrifices fine-grained details for text efficiency.
        # We sample frames and convert them to dense captions.
        self.sample_num = 16 # 典型的 captioning 方法采样数
        
    def process_and_inference(self, video_path, question, options):
        # 1. Uniform Sampling
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames-1, self.sample_num, dtype=int)
        
        # 2. Dense Captioning (Visual -> Text)
        captions = []
        for idx in indices:
            # 提取单帧
            frame = vr[idx].asnumpy()
            timestamp = idx / vr.get_avg_fps()
            
            # 使用 VLLM 生成详细描述
            # Prompt 设计参考 SceneGraph-Cap: "Describe the scene, objects, and actions in detail."
            # 注意: 这里需要调用 model 的纯视觉生成能力
            cap_prompt = "Describe the image in detail."
            
            # 临时把单帧封装成 tensor
            frame_tensor = self.model.processor(videos=list(frame[None, ...]), return_tensors="pt")['pixel_values']
            frame_tensor = frame_tensor.to(self.model.device)
            
            # 生成
            desc = self.model.generate(frame_tensor, cap_prompt)
            captions.append(f"[Time: {timestamp:.1f}s] {desc}")
            
        # 3. Text-only Inference
        # 拼接所有 Caption
        context = "\n".join(captions)
        
        # 构造纯文本 Prompt
        text_prompt = (
            f"Here is a textual description of a video:\n{context}\n\n"
            f"Based on this, answer the question: {question}"
        )
        
        # 4. 调用 LLM (注意: video_tensor=None)
        # 这样模型就真的只看 Text，符合 "Caption-based" paradigm
        return self.model.generate(None, text_prompt, options)