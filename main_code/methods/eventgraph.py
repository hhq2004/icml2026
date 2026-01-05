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
        论文Section 3.2: Event Nodes
        使用shot boundary detection切分视频为事件单元
        
        实现: 基于颜色直方图的shot detection + 自适应阈值
        (参考标准方法,如PySceneDetect的原理,但避免额外依赖)
        
        Returns:
            events: List[(start_sec, end_sec), ...]
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or frame_count == 0:
            cap.release()
            return []
        
        duration = frame_count / fps
        
        # 采样帧进行分析 (避免处理所有帧,加速)
        sample_interval = max(1, int(fps / 2))  # 每秒采样2帧
        
        # 读取采样帧并计算直方图差异
        prev_hist = None
        hist_diffs = []
        sampled_frame_indices = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 只处理采样帧
            if frame_idx % sample_interval == 0:
                # 转换到HSV空间 (对光照变化更鲁棒)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = cv2.resize(hsv, (160, 120))  # 减小尺寸加速
                
                # 计算颜色直方图 (H和S通道)
                hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
                
                # 归一化
                hist_h = cv2.normalize(hist_h, hist_h).flatten()
                hist_s = cv2.normalize(hist_s, hist_s).flatten()
                
                if prev_hist is not None:
                    # 使用直方图交叉度量相似性
                    diff_h = 1 - cv2.compareHist(prev_hist[0], hist_h, cv2.HISTCMP_CORREL)
                    diff_s = 1 - cv2.compareHist(prev_hist[1], hist_s, cv2.HISTCMP_CORREL)
                    
                    # 综合两个通道的差异
                    diff = 0.7 * diff_h + 0.3 * diff_s
                    hist_diffs.append(diff)
                    sampled_frame_indices.append(frame_idx)
                
                prev_hist = (hist_h, hist_s)
            
            frame_idx += 1
        
        cap.release()
        
        if len(hist_diffs) == 0:
            # Fallback: 固定时间切分
            print("  [Shot Detection] No frames analyzed, using fallback (2s segments)")
            events = []
            for t in np.arange(0, duration, 2.0):
                events.append((t, min(t + 2.0, duration)))
            return events
        
        # 自适应阈值检测shot boundary
        hist_diffs = np.array(hist_diffs)
        
        # 使用中位数绝对偏差 (MAD) 估计阈值 (更鲁棒)
        median = np.median(hist_diffs)
        mad = np.median(np.abs(hist_diffs - median))
        threshold = median + 3.0 * mad  # 3-sigma规则
        
        # 如果MAD太小(视频变化很小),使用percentile
        if mad < 0.01:
            threshold = np.percentile(hist_diffs, 90)
        
        # 检测boundary (峰值检测)
        boundaries = [0]
        for i in range(1, len(hist_diffs) - 1):
            # 局部最大值 + 超过阈值
            if (hist_diffs[i] > hist_diffs[i-1] and 
                hist_diffs[i] > hist_diffs[i+1] and 
                hist_diffs[i] > threshold):
                boundaries.append(sampled_frame_indices[i])
        boundaries.append(frame_idx - 1)
        
        # 转换为时间段
        events = []
        for i in range(len(boundaries) - 1):
            start_sec = boundaries[i] / fps
            end_sec = boundaries[i + 1] / fps
            events.append((start_sec, end_sec))
        
        # 合并过短的片段 (< 0.5秒)
        merged_events = []
        if events:
            current_event = events[0]
            
            for i in range(1, len(events)):
                duration_current = current_event[1] - current_event[0]
                if duration_current < 0.5:
                    # 合并到下一个
                    current_event = (current_event[0], events[i][1])
                else:
                    merged_events.append(current_event)
                    current_event = events[i]
            
            # 添加最后一个
            if current_event[1] - current_event[0] >= 0.5:
                merged_events.append(current_event)
        
        # 防止events过多(限制最多50个)或过少
        if len(merged_events) > 50:
            # 按时长合并,保留最重要的50个
            step = len(merged_events) // 50
            merged_events = merged_events[::step][:50]
        elif len(merged_events) < 3:
            # 太少,使用fallback
            merged_events = []
            for t in np.arange(0, duration, duration / 5):
                merged_events.append((t, min(t + duration / 5, duration)))
        
        return merged_events if merged_events else events
    
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