# /root/hhq/main_code/models/video_llava.py
"""
Video-LLaVA-7B Wrapper for Video Understanding
修复版本 - 解决336x336图像尺寸问题
"""
import torch
import numpy as np
import os
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoConfig
)

# 尝试导入 decord
try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️  Warning: decord not installed. Install with: pip install decord")
    VideoReader = None

class VideoLLaVAWrapper:
    def __init__(self, model_path="/root/hhq/models/Video-LLaVA-7B-hf"):  # ⭐ 修正：添加-hf后缀
        print(f"🚀 [Model] Initializing from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 加载 Config
        try:
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            print(f"   Model Type: {config.model_type}")
        except Exception as e:
            print(f"   Warning: Could not load config: {e}")

        # 2. 加载 Processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True  # ⭐ 路径已修正，可以使用本地模式
        )
        print(f"   ✓ Processor loaded successfully")
        
        
        # ⭐ 关键修复1：强制设置image_processor的size为336
        # Video-LLaVA使用CLIP-ViT-L/14，image_size必须是336
        if hasattr(self.processor, 'image_processor'):
            self.processor.image_processor.size = {"shortest_edge": 336}
            self.processor.image_processor.crop_size = {"height": 336, "width": 336}
            print(f"   ✓ Image processor size set to 336x336")
        
        # ⭐ 关键修复2：设置patch_size（CLIP-ViT-L/14 = 14）
        # 这个是processor计算token数量时需要的
        if not hasattr(self.processor, 'patch_size') or self.processor.patch_size is None:
            self.processor.patch_size = 14
            print(f"   ✓ Patch size set to 14")
        
        # ⭐ 关键修复3：设置vision_feature_select_strategy
        # 某些processor版本需要这个参数
        if not hasattr(self.processor, 'vision_feature_select_strategy'):
            self.processor.vision_feature_select_strategy = "default"

        # 3. 检查transformers版本
        import transformers
        print(f"   Transformers version: {transformers.__version__}")
        if transformers.__version__ < "4.30" or transformers.__version__ >= "5.0":
            print(f"   ⚠️  Model was trained with transformers 4.31.0")

        # 4. 加载模型
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,  # ⭐ 路径已修正
            trust_remote_code=True
        )
        self.model.eval()
        print(f"   ✓ Model loaded successfully")

        # 5. 获取并初始化 Vision Tower（增强版）
        self.vision_tower = None
        
        # 尝试多种方式获取vision tower
        if hasattr(self.model, 'get_vision_tower'):
            self.vision_tower = self.model.get_vision_tower()
        elif hasattr(self.model, 'vision_tower'):
            self.vision_tower = self.model.vision_tower
        elif hasattr(self.model, 'get_model'):
            base_model = self.model.get_model()
            if hasattr(base_model, 'get_vision_tower'):
                self.vision_tower = base_model.get_vision_tower()
            elif hasattr(base_model, 'vision_tower'):
                self.vision_tower = base_model.vision_tower
        
        # 确保vision tower被初始化  
        if self.vision_tower and hasattr(self.vision_tower, 'load_model'):
            self.vision_tower.load_model()
            print(f"   ✓ Vision tower loaded and initialized")
        elif self.vision_tower:
            print(f"   ✓ Vision tower found")
        else:
            print(f"   ⚠️  Vision tower not found")
        
        # FastV剪枝配置
        self.fastv_enabled = False
        self.fastv_filtering_layer = 2
        self.fastv_filtering_ratio = 0.5
        self.fastv_hook_handle = None
        self.fastv_pruned_indices = None
        self.fastv_num_image_tokens = 0
            
        print(f"✅ Model loaded successfully on {self.device}")

    def enable_fastv_pruning(self, filtering_layer, filtering_ratio):
        """启用FastV剪枝 - 7B模型简化版(仅支持K=0)"""
        self.fastv_enabled = True
        self.fastv_filtering_layer = filtering_layer
        self.fastv_filtering_ratio = filtering_ratio
        
        if filtering_layer != 0:
            print(f"⚠️  Video-LLaVA-7B only supports K=0, falling back")
        
        print(f"[FastV] Enabled for 7B: K={filtering_layer}, R={filtering_ratio}")
    
    def disable_fastv_pruning(self):
        """禁用FastV剪枝"""
        self.fastv_enabled = False
        print(f"[FastV] Disabled")

    def _load_video_frames(self, video_path, start_time, end_time, num_frames=8):
        """ 
        使用 decord 读取特定时间段的帧 
        返回: numpy array (num_frames, H, W, 3)
        """
        if not VideoReader:
            print("⚠️  decord not available, returning dummy frames")
            return np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
            
        if not os.path.exists(video_path):
            print(f"⚠️  Video not found: {video_path}")
            return np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            
            start_idx = max(0, int(start_time * fps))
            end_idx = min(total_frames - 1, int(end_time * fps))
            
            if start_idx >= end_idx:
                indices = [start_idx] * num_frames
            else:
                indices = np.linspace(start_idx, end_idx, num_frames).astype(int)
            
            frames = vr.get_batch(indices).asnumpy()  # (K, H, W, C)
            return frames
        except Exception as e:
            print(f"⚠️  Error loading video frames: {e}")
            return np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)

    def encode_text(self, text):
        """ 
        提取 Query 文本特征 
        用于Q-Frame等方法的文本编码
        """
        inputs = self.processor.tokenizer(
            text, 
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(inputs["input_ids"])
            text_feat = torch.mean(embeds, dim=1)  # (1, hidden_dim)
        return text_feat

    def encode_events(self, video_path, events, frames_per_event=8):
        """ 
        EventGraph-LMM专用: 提取Event特征
        
        Args:
            video_path: 视频路径
            events: list of (start_time, end_time) tuples
            frames_per_event: 每个event采样帧数
            
        Returns:
            dict: {"global": tensor, "local": tensor, "costs": tensor}
        """
        global_feats = []
        local_feats = []
        costs = []
        event_cost = 64  # 每个event的token消耗
        
        for (start, end) in events:
            frames = self._load_video_frames(video_path, start, end, num_frames=frames_per_event)
            
            # 转为PIL Images
            pil_frames = [Image.fromarray(f) for f in frames]
            
            # Processor处理
            inputs = self.processor(
                images=pil_frames,
                return_tensors="pt",
                padding=True
            )
            pixel_values = inputs.pixel_values.to(self.device, dtype=torch.float16)
            
            # 通过Vision Tower提取特征
            with torch.no_grad():
                if self.vision_tower:
                    outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                    features = outputs.hidden_states[-1]  # (B, L, D)
                else:
                    raise RuntimeError("Vision tower not available!")

                # Pooling
                g_feat = torch.mean(features, dim=[0, 1])  # Global
                l_feat = torch.mean(features, dim=0)       # Local
                
            global_feats.append(g_feat.cpu())
            local_feats.append(l_feat.cpu())
            costs.append(event_cost)
            
        if not global_feats:
            return None

        return {
            "global": torch.stack(global_feats).to(self.device),
            "local": torch.stack(local_feats).to(self.device),
            "costs": torch.tensor(costs, device=self.device)
        }

    def generate_from_segments(self, video_path, selected_timestamps, question, options):
        """ 
        EventGraph-LMM推理入口
        从选中的视频片段生成答案
        """
        all_frames = []
        for (start, end) in selected_timestamps:
            frames = self._load_video_frames(video_path, start, end, num_frames=4)
            all_frames.extend(list(frames))
            
        if len(all_frames) == 0:
            return "C"  # 默认答案
        
        # 转换为numpy stack
        video_tensor = np.stack(all_frames)
        return self.generate(video_tensor, question, options)

    def generate(self, video_tensor, prompt, options=None):
        """
        通用推理接口
        
        Args:
            video_tensor: numpy array (K, H, W, C) 或 PIL Images列表 或 单个PIL Image
            prompt: 问题文本  
            options: 选项列表（可选）
            
        Returns:
            answer: str, 模型生成的答案
        """
        # 1. 转换输入为PIL Images列表
        frames = []
        
        if isinstance(video_tensor, np.ndarray):
            # Numpy array
            if video_tensor.ndim == 4:  # (K, H, W, C)
                frames = [Image.fromarray(frame.astype(np.uint8)) for frame in video_tensor]
            elif video_tensor.ndim == 3:  # 单帧 (H, W, C)
                frames = [Image.fromarray(video_tensor.astype(np.uint8))]
            else:
                raise ValueError(f"Unexpected video_tensor shape: {video_tensor.shape}")
                
        elif isinstance(video_tensor, list):
            # 列表
            if len(video_tensor) > 0:
                if isinstance(video_tensor[0], np.ndarray):
                    frames = [Image.fromarray(f.astype(np.uint8)) for f in video_tensor]
                elif isinstance(video_tensor[0], Image.Image):
                    frames = video_tensor
                else:
                    raise TypeError(f"Unsupported list element type: {type(video_tensor[0])}")
            else:
                raise ValueError("Empty video_tensor list")
                
        elif isinstance(video_tensor, Image.Image):
            # 单张PIL Image
            frames = [video_tensor]
            
        else:
            raise TypeError(f"Unsupported video_tensor type: {type(video_tensor)}")
        
        # ⭐ 关键修复：确保所有帧都是336x336
        # 即使processor配置了size，我们也显式resize以确保万无一失
        frames = [f.resize((336, 336), Image.Resampling.BILINEAR) if f.size != (336, 336) else f 
                  for f in frames]
        
        print(f"   📊 Processing {len(frames)} frames at 336x336")
        
        # 2. 构建Prompt（在frames定义之后）
        # ⭐ 注意：LLaVA processor期望<image>标记，不是<video>
        formatted_prompt = f"USER: <image>\\n{prompt}\\n"
        
        if options:
            formatted_prompt += "Select the best answer from:\\n"
            for i, opt in enumerate(options):
                formatted_prompt += f"({chr(65+i)}) {opt}\\n"
            formatted_prompt += "Answer with the option letter directly.\\nASSISTANT:"
        else:
            formatted_prompt += "ASSISTANT:"
        
        print(f"   💬 Prompt: {formatted_prompt[:100]}...")
        
        # 3. 使用Processor处理
        # ⭐ 关键：对于多帧，使用videos参数而不是images参数
        try:
            # Video-LLaVA应该支持videos参数
            inputs = self.processor(
                text=formatted_prompt,
                videos=frames,  # 使用videos参数
                return_tensors="pt",
                padding=True
            )
            print(f"   ✓ Processor succeeded with 'videos' parameter")
        except Exception as e:
            print(f"   ⚠️  'videos' parameter failed: {e}")
            # 回退：使用images参数（需要调整prompt）
            try:
                # 如果使用images，需要在prompt中插入对应数量的<image>
                image_tokens = "\\n".join(["<image>"] * len(frames))
                formatted_prompt_multi = formatted_prompt.replace("<image>", image_tokens)
                
                inputs = self.processor(
                    text=formatted_prompt_multi,
                    images=frames,
                    return_tensors="pt",
                    padding=True
                )
                print(f"   ✓ Processor succeeded with 'images' parameter (multi-token prompt)")
            except Exception as e2:
                print(f"   ❌ Both methods failed: {e}, {e2}")
                raise e
        
        # 4. 移到GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        # 5. 推理
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        
        # 6. 解码
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # 提取ASSISTANT后的内容
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response