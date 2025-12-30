# /root/hhq/main_code/methods/eventgraph.py
import torch
import numpy as np
import os
import cv2
from transformers import CLIPProcessor, CLIPModel
from .base_method import BaseMethod
from utils.graph_builder import compute_similarity_matrix, compute_pagerank_matrix
from utils.celf_solver import CELFSelector

class EventGraphLLM(BaseMethod):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.semantic_threshold = 0.65
        self.temporal_threshold = 30
        self.trade_off_lambda = 1.0
        
        # === 核心修改：加载本地 CLIP (用于计算 Query Relevance) ===
        local_clip = "/root/hhq/models/clip-vit-large-patch14"
        if os.path.exists(local_clip):
            print(f"[EventGraph] Loading CLIP from LOCAL: {local_clip}")
            model_name = local_clip
        else:
            print(f"[EventGraph] Local CLIP not found, verifying network...")
            model_name = "openai/clip-vit-large-patch14"
            
        self.clip_device = self.model.device
        try:
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.clip_device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"❌ [EventGraph] CLIP load failed: {e}")
            raise e
        # ========================================================
        
    def _detect_events(self, video_path):
        # 简单模拟：每 2 秒一个 event (后续可换成 PySceneDetect)
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / (fps + 1e-6)
            cap.release()
        except:
            duration = 0
        
        if duration <= 0: return []
        
        events = []
        seg = 2.0
        for t in np.arange(0, duration, seg):
            events.append((t, min(t+seg, duration)))
        return events

    def _get_clip_features(self, frames, text):
        """ 使用独立 CLIP 提取特征用于建图 """
        # Text
        inputs_text = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs_text = {k: v.to(self.clip_device) for k, v in inputs_text.items()}
        
        # Images (Batch processing frames)
        # 为了防爆显存，简单取每一段的中间帧作为代表
        inputs_img = self.clip_processor(images=frames, return_tensors="pt", padding=True)
        inputs_img = {k: v.to(self.clip_device) for k, v in inputs_img.items()}
        
        with torch.no_grad():
            text_feat = self.clip_model.get_text_features(**inputs_text)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            
            img_feats = self.clip_model.get_image_features(**inputs_img)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
        return text_feat, img_feats

    def process_and_inference(self, video_path, question, options):
        # 1. 切分 Events
        events = self._detect_events(video_path)
        if not events: return "C" # 兜底
        
        # 2. 准备 Event 代表帧 (用于 CLIP 计算相似度)
        # 这里的逻辑是：为了算图的权重，我们用 CLIP；为了最后的推理，我们用 VideoLLaVA
        representative_frames = []
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        
        for (start, end) in events:
            mid = (start + end) / 2
            idx = min(len(vr)-1, int(mid * fps))
            representative_frames.append(vr[idx].asnumpy())
            
        # 3. 计算特征 (CLIP)
        query_feat, event_feats = self._get_clip_features(representative_frames, question)
        
        # 4. 计算相关性 (Relevance)
        # (N, D) * (D, 1) -> (N, 1)
        rel = torch.mm(event_feats, query_feat.t()).squeeze()
        rel = torch.clamp(rel, min=0)
        
        # 5. 建图 (Similarity)
        # EventGraph 需要两部分特征：Global (CLS) 和 Local (Patch)
        # 这里为了跑通，我们暂时只用 Global Sim
        # (N, D) * (D, N) -> (N, N)
        sim_matrix = torch.mm(event_feats, event_feats.t())
        # Apply temporal constraint (masking)
        sim_matrix = compute_similarity_matrix(
            event_feats, # Global
            None,        # Local (暂时设为None)
            tau=self.temporal_threshold,
            event_times=events
        )
        
        # 6. PageRank
        N = len(events)
        adj = sim_matrix.clone()
        # 添加时间边
        for i in range(N-1):
            adj[i, i+1] = 1.0 
            
        Pi = compute_pagerank_matrix(adj)
        
        # 7. CELF 选择
        # 假设每个 event 的开销是固定的 (例如 1)
        costs = torch.ones(N, device=self.clip_device)
        # 预算换算成 event 数量 (假设 budget=2048, 1 event ≈ 256 tokens => budget=8 events)
        budget_in_events = max(4, self.token_budget // 256)
        
        solver = CELFSelector(Pi, rel, costs, lambda_param=self.trade_off_lambda)
        selected_indices = solver.select(budget_in_events)
        
        selected_timestamps = [events[i] for i in selected_indices]
        
        # 排序 (按时间顺序喂给 LLM)
        selected_timestamps.sort(key=lambda x: x[0])
        
        # 8. 推理 (VideoLLaVA)
        return self.model.generate_from_segments(video_path, selected_timestamps, question, options)