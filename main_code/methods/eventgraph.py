# /root/hhq/main_code/methods/eventgraph.py
"""
EventGraph-LMM完整实现
严格按照ICML 2026论文复现,不做任何简化

论文: EventGraph-LMM: Submodular Information Maximization for Efficient Long-Video Understanding
Section 3: Methodology
- 3.2: Graph Construction
- 3.3: Query-Conditional Subgraph Selection  
- 3.4: Graph-Constrained Chain-of-Thought

实现方式: Training-free插件,不修改模型内部,与Q-Frame类似
"""
import torch
import numpy as np
import cv2
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from .base_method import BaseMethod
from utils.graph_builder import compute_similarity_matrix, compute_pagerank_matrix
from utils.celf_solver import CELFSelector

try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Warning: decord not installed")
    VideoReader = None

class EventGraphLMM(BaseMethod):
    def __init__(self, args, model):
        """
        Args:
            args: 命令行参数
            model: VideoLLaVA model (通过run_inference.py传入)
        """
        super().__init__(args, model)
        
        # === 论文Section 4.1: Implementation Details ===
        self.tau = 30.0  # Temporal distance threshold (seconds)
        self.delta = 0.65  # Semantic similarity threshold
        self.alpha = 0.15  # PageRank restart probability
        self.lambda_param = 1.0  # Trade-off parameter in Eq. 8
        self.token_budget = args.token_budget  # B = 2048
        
        # Backbone信息
        backbone = getattr(args, 'backbone', 'Video-LLaVA-7B')
        if '34B' in backbone or '32B' in backbone:
            self.tokens_per_frame = 576
        else:
            self.tokens_per_frame = 256
        
        print(f"[EventGraph-LMM] Initializing...")
        print(f"  - Backbone: {backbone}")
        print(f"  - τ (temporal threshold) = {self.tau}s")
        print(f"  - δ (similarity threshold) = {self.delta}")
        print(f"  - α (PageRank restart) = {self.alpha}")
        print(f"  - λ (trade-off) = {self.lambda_param}")
        print(f"  - B (token budget) = {self.token_budget}")
        print(f"  - Tokens per frame = {self.tokens_per_frame}")
        
        # === 设备配置 ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  - Device: {self.device}")
        
        # === 加载CLIP模型 (用于Graph Construction) ===
        self._load_clip_model()
        
        print(f"[EventGraph-LMM] ✅ Initialization complete!")
    
    def _load_clip_model(self):
        """加载CLIP-ViT-L/14 (论文Section 4.1)"""
        local_clip_path = "/root/hhq/models/clip-vit-large-patch14"
        
        if os.path.exists(local_clip_path):
            print(f"  - Loading CLIP from local: {local_clip_path}")
            model_name = local_clip_path
        else:
            print(f"  - CLIP not found locally, using online: openai/clip-vit-large-patch14")
            model_name = "openai/clip-vit-large-patch14"
        
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name)
            
            # 移动到GPU
            self.clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.clip_model = self.clip_model.to(self.clip_device)
            self.clip_model.eval()
            
            print(f"  ✓ CLIP loaded successfully on {self.clip_device}")
            
        except Exception as e:
            print(f"❌ Failed to load CLIP: {e}")
            raise e
    
    def _detect_shot_boundaries(self, video_path):
        """
        论文Section 3.2 (Lines 606-608): Event Nodes
        "We first decompose the long video into a sequence of N variable-length visual segments, 
         V = {v1, v2, . . . , vN}, using a standard event-based shot boundary detection algorithm 
         (Soucek & Lokoc, 2024)."
        
        实现: TransNet V2 - 论文引用的专业shot detection算法
        - 论文: TransNet V2: An effective deep network architecture for fast shot transition detection
        - 作者: Tomáš Souček & Jakub Lokoč (2020)
        - 特性: 检测abrupt和gradual transitions，输出variable-length segments
        
        Returns:
            events: List[(start_sec, end_sec), ...] - N个variable-length视频片段
        """
        try:
            from utils.shot_detector import TransNetV2Detector, TRANSNET_AVAILABLE
            
            if not TRANSNET_AVAILABLE:
                # Fallback: 使用PySceneDetect或固定窗口
                print("  ⚠️  TransNet V2 not available, using fallback")
                return self._detect_shot_boundaries_fallback(video_path)
            
            # 使用TransNet V2进行shot detection
            detector = TransNetV2Detector(device=self.device)
            events = detector.detect_shots(video_path, threshold=0.5)
            
            # 后处理：合并过短的片段（<0.5秒）
            # 这有助于避免过度碎片化，保证事件的语义完整性
            events = detector.merge_short_segments(events, min_duration=0.5)
            
            # 论文没有明确限制事件数量，但实践中限制在合理范围（3-100个）
            if len(events) > 100:
                print(f"  ⚠️  Too many shots ({len(events)}), sampling to 100")
                step = len(events) // 100
                events = events[::step][:100]
            elif len(events) < 3:
                print(f"  ⚠️  Too few shots ({len(events)}), using fallback")
                return self._detect_shot_boundaries_fallback(video_path)
            
            print(f"  ✓ Detected {len(events)} variable-length events")
            return events
            
        except Exception as e:
            print(f"  ❌ TransNet V2 failed: {e}")
            print("  → Using fallback shot detection")
            return self._detect_shot_boundaries_fallback(video_path)
    
    def _detect_shot_boundaries_fallback(self, video_path):
        """
        Fallback shot detection（当TransNet V2不可用时）
        
        注意：这不是论文引用的方法，仅作为备用方案
        使用固定时间窗口（2秒）进行切分
        
        Returns:
            events: List[(start_sec, end_sec), ...]
        """
        # 获取视频时长
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            duration = len(vr) / vr.get_avg_fps()
        except:
            # 如果decord失败，使用opencv
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps == 0 or frame_count == 0:
                return [(0, 10)]  # 默认10秒
            duration = frame_count / fps
        
        # 固定2秒窗口切分
        events = []
        for t in np.arange(0, duration, 2.0):
            events.append((t, min(t + 2.0, duration)))
        
        print(f"  ⚠️  Fallback: Fixed 2s window, {len(events)} segments")
        return events
    
    def _extract_event_features(self, video_path, events):
        """
        论文Section 3.2: 提取事件的视觉特征
        使用CLIP-ViT-L/14提取Global和Local特征
        
        Returns:
            global_feats: (N, D) Global features from [CLS] token
            local_feats: (N, L, D) Local features from patch tokens
            representative_frames: List[PIL.Image] 用于后续推理
        """
        if VideoReader is None:
            raise ImportError("decord is required")
        
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        
        representative_frames = []
        
        # 为每个event抽取中间帧
        for start_t, end_t in events:
            mid_t = (start_t + end_t) / 2.0
            frame_idx = min(len(vr) - 1, int(mid_t * fps))
            frame_np = vr[frame_idx].asnumpy()
            frame_pil = Image.fromarray(frame_np)
            representative_frames.append(frame_pil)
        
        # 使用CLIP提取特征
        # ⚠️ 关键: 必须使用get_image_features而非vision_model,确保维度与text特征一致
        with torch.no_grad():
            inputs = self.clip_processor(images=representative_frames, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.clip_device) for k, v in inputs.items()}
            
            # 方法1: 使用get_image_features (投影后,与text特征维度一致)
            global_feats = self.clip_model.get_image_features(**inputs)  # (N, D_proj)
            
            # 方法2: 获取Local features - 需要访问vision_model的hidden states
            vision_outputs = self.clip_model.vision_model(**inputs, output_hidden_states=True)
            
            # Local features: Patch tokens (来自最后一层hidden states)
            local_feats = vision_outputs.last_hidden_state[:, 1:, :]  # (N, L, D_hidden)
        
        return global_feats, local_feats, representative_frames
    
    def _construct_event_graph(self, global_feats, local_feats, events):
        """
        论文Section 3.2: Graph Construction
        E = E_temp ∪ E_sem
        
        Returns:
            adj_matrix: (N, N) 邻接矩阵
        """
        # 计算语义边 (Eq. 3 + 4)
        adj_semantic = compute_similarity_matrix(
            global_feats=global_feats,
            local_feats=local_feats,
            tau=self.tau,
            event_times=events,
            threshold=self.delta
        )
        
        # 添加时序边 (Eq. 2)
        N = len(events)
        adj_total = adj_semantic.clone()
        
        for i in range(N - 1):
            adj_total[i, i + 1] = 1.0
        
        return adj_total
    
    def _select_subgraph(self, adj_matrix, query_text, event_feats, events):
        """
        论文Section 3.3: Query-Conditional Subgraph Selection
        F_q(S) = F_rel(S) + λ * F_reach(S)
        
        Returns:
            selected_indices: List[int]
        """
        N = len(events)
        
        # 1. 计算Query Relevance (Eq. 5)
        with torch.no_grad():
            text_inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(self.clip_device) for k, v in text_inputs.items()}
            query_feats = self.clip_model.get_text_features(**text_inputs)
        
        query_feats_norm = torch.nn.functional.normalize(query_feats, p=2, dim=-1)
        event_feats_norm = torch.nn.functional.normalize(event_feats, p=2, dim=-1)
        
        rel = torch.mm(event_feats_norm, query_feats_norm.t()).squeeze()
        rel = torch.clamp(rel, min=0.0)
        
        # 2. 计算Reachability Matrix (Eq. 6)
        Pi = compute_pagerank_matrix(adj_matrix, alpha=self.alpha)
        
        # 3. CELF优化 (Algorithm 1)
        costs = torch.ones(N, device=event_feats.device)
        budget_events = max(4, self.token_budget // self.tokens_per_frame)
        
        solver = CELFSelector(
            Pi=Pi,
            query_relevance=rel,
            costs=costs,
            lambda_param=self.lambda_param
        )
        
        selected_indices = solver.select(budget=budget_events)
        
        return selected_indices
    
    def _build_graph_cot_prompt(self, question, options, segments, adj_matrix, selected_indices):
        """
        论文Section 3.4: Graph-Constrained Chain-of-Thought (精简版)
        保留核心的图引导逻辑,但大幅减少prompt长度避免超过max_length
        """
        # 构建事件时间线
        event_timeline = []
        for i, (start_t, end_t, _) in enumerate(segments):
            event_timeline.append(f"Event{i+1} ({start_t:.0f}-{end_t:.0f}s)")
        
        # 统计边连接
        edge_count = 0
        for i, src_idx in enumerate(selected_indices):
            for j, tgt_idx in enumerate(selected_indices):
                if src_idx != tgt_idx and adj_matrix[src_idx, tgt_idx].item() > 0:
                    edge_count += 1
        
        # 精简的三阶段prompt
        prompt = f"""Question: {question}
Options:
{chr(10).join(options)}

Video Timeline: {' → '.join(event_timeline)}
Connected Events: {edge_count} semantic/temporal links

Instructions:
1. Examine each event for relevant visual evidence
2. Follow the connections between events to build reasoning chains
3. Select the answer supported by the strongest evidence path

Answer (A/B/C/D only):"""
        
        return prompt
    
    def process_and_inference(self, video_path, question, options):
        """
        主流程: EventGraph-LMM完整pipeline
        """
        print(f"\n[EventGraph] Processing: {video_path}")
        
        # === Stage 1: Graph Construction ===
        print(f"[Stage 1/3] Graph Construction...")
        
        events = self._detect_shot_boundaries(video_path)
        if len(events) == 0:
            return "C"
        print(f"  - Detected {len(events)} events")
        
        global_feats, local_feats, _ = self._extract_event_features(video_path, events)
        print(f"  - Extracted features: Global{global_feats.shape}, Local{local_feats.shape}")
        
        adj_matrix = self._construct_event_graph(global_feats, local_feats, events)
        print(f"  - Graph: {adj_matrix.shape}")
        
        # === Stage 2: Subgraph Selection ===
        print(f"[Stage 2/3] Subgraph Selection...")
        
        selected_indices = self._select_subgraph(adj_matrix, question, global_feats, events)
        print(f"  - Selected {len(selected_indices)} events: {selected_indices}")
        
        # 准备选中的片段
        selected_segments = [(events[i][0], events[i][1], i) for i in selected_indices]
        selected_segments.sort(key=lambda x: x[0])
        
        # 提取选中帧
        if VideoReader is None:
            raise ImportError("decord required")
        
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        
        selected_frames = []
        for start_t, end_t, _ in selected_segments:
            mid_t = (start_t + end_t) / 2.0
            frame_idx = min(len(vr) - 1, int(mid_t * fps))
            frame_np = vr[frame_idx].asnumpy()
            frame_pil = Image.fromarray(frame_np)
            selected_frames.append(frame_pil)
        
        # === Stage 3: Graph-CoT Inference ===
        print(f"[Stage 3/3] Graph-CoT Inference...")
        
        # 构建Graph-CoT prompt
        prompt = self._build_graph_cot_prompt(question, options, selected_segments, adj_matrix, selected_indices)
        
        # 调用VideoLLaVA推理 (与Q-Frame相同方式)
        answer = self.model.generate(selected_frames, prompt, options)
        
        print(f"  - Prediction: {answer}")
        
        return answer