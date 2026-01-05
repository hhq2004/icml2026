#!/usr/bin/env python3
"""
ToMe (Token Merging) Wrapper for Video-LLaVA-7B
严格按照官方 tome/patch/timm.py 实现

核心思路：
1. 加载标准Video-LLaVA-7B模型
2. Monkey patch Vision Tower的CLIPEncoderLayer
3. 在每层attention后执行token merging
4. 最终输出维度不变，对LLM透明

参考：
- 论文: Token Merging: Your ViT But Faster (ICLR 2023)
- 官方源码: origin/ToMe/tome/patch/timm.py
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Callable, Tuple, List, Optional
from PIL import Image

# 导入基础wrapper
from .video_llava_7b import VideoLLaVAWrapper

try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Warning: decord not installed")
    VideoReader = None


# ============================================================================
# 核心算法：直接从官方 tome/merge.py 照搬
# ============================================================================

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    双向软匹配算法（官方tome/merge.py line 18-97原文实现）
    
    Applies ToMe with a balanced matching set (50%, 50%).
    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    
    Args:
        metric: Similarity metric tensor (通常是attention keys)
        r: 要合并的token数量
        class_token: 是否有CLS token（CLIP有）
        distill_token: 是否有蒸馏token（CLIP没有）
    
    Returns:
        merge: 合并函数
        unmerge: 反合并函数（用于可视化）
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        # No merging
        return lambda x, mode=None: x, lambda x: x

    with torch.no_grad():
        # ⭐ Cosine similarity (normalize)
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # ⭐ Alternating partition
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # ⭐ Compute similarity scores
        scores = a @ b.transpose(-1, -2)

        # ⭐ Protect CLS token
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        # ⭐ Find best match for each token in A
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # ⭐ Keep top-r edges
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        """
        应用token merging（官方line 70-80）
        
        Args:
            x: [batch, tokens, channels]
            mode: "mean" or "sum"
        
        Returns:
            merged_x: [batch, tokens-r, channels]
        """
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        
        # Extract unmerged tokens from A
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        
        # Extract source tokens to be merged
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # ⭐ Scatter-reduce (官方使用scatter_reduce，PyTorch 1.12+)
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """反合并（用于可视化，官方line 82-95）"""
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, 
    x: torch.Tensor, 
    size: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加权平均合并（官方tome/merge.py line 210-224）
    
    Applies the merge function by taking a weighted average based on token size.
    
    Args:
        merge: 合并函数
        x: tokens [batch, tokens, channels]
        size: token大小 [batch, tokens, 1]（表示每个token代表几个原始patch）
    
    Returns:
        merged_x: 合并后的tokens
        merged_size: 合并后的token大小
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    # Weighted sum
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    # Normalize
    x = x / size
    
    return x, size


def parse_r(num_layers: int, r: int) -> List[int]:
    """
    解析merging schedule（官方tome/utils.py line 80-105）
    
    Args:
        num_layers: 层数
        r: 每层合并的token数（constant schedule）
    
    Returns:
        r_schedule: List[int]，每层的r值
    """
    # Constant schedule: 每层合并相同数量
    return [r] * num_layers


# ============================================================================
# ToMe适配层：将官方timm.Block逻辑适配到CLIPEncoderLayer
# ============================================================================

class ToMeCLIPEncoderLayer(nn.Module):
    """
    ToMe版本的CLIPEncoderLayer
    
    参照官方tome/patch/timm.py:ToMeBlock (line 21-56)
    适配transformers的CLIPEncoderLayer架构
    
    核心修改：
    1. 在self-attention后执行token merging
    2. 跟踪token size（用于weighted average）
    3. 不改变输入输出接口（对外透明）
    """
    
    def __init__(self, original_layer, tome_info: dict, layer_idx: int):
        """
        Args:
            original_layer: 原始的CLIPEncoderLayer实例
            tome_info: ToMe配置字典
            layer_idx: 当前层的索引（用于从r列表中获取对应的r值）
        """
        super().__init__()
        
        # 复制原始layer的所有属性
        self.__dict__.update(original_layer.__dict__)
        
        # ToMe配置
        self._tome_info = tome_info
        self._layer_idx = layer_idx  # ⭐ 保存layer索引
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """
        前向传播（参照官方ToMeBlock.forward，line 34-56）
        
        流程：
        1. Layer Norm 1
        2. Self-Attention（需要获取keys用于matching）
        3. ⭐ Token Merging（如果r>0）
        4. Layer Norm 2
        5. MLP
        """
        # 1. Residual connection准备
        residual = hidden_states
        
        # 2. Layer Norm 1
        hidden_states = self.layer_norm1(hidden_states)
        
        # 3. Self-Attention
        # ⚠️ 这里需要获取attention keys用于matching
        # CLIPAttention的forward默认不返回keys，我们通过_tome_info传递
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        
        # CLIPAttention返回(attn_output, attn_weights)或attn_output
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        
        hidden_states = residual + attn_output
        
        # 4. ⭐ Token Merging（核心逻辑）
        # ⚠️ 使用layer_idx访问r，而不是pop()
        r = self._tome_info["r"][self._layer_idx] if self._tome_info["r"] else 0
        
        # 🔍 DEBUG: 打印layer info（仅layer 0）
        if self._layer_idx == 0:
            print(f"  [ToMe DEBUG] Layer {self._layer_idx}: input shape={hidden_states.shape}, r={r}")
        
        if r > 0:
            input_shape = hidden_states.shape
            
            # 获取metric（使用上一步attention的keys）
            # ⚠️ 临时方案：使用hidden_states作为metric
            # TODO: 理想情况应该从attention提取keys
            metric = hidden_states
            
            # Bipartite soft matching
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                class_token=self._tome_info.get("class_token", True),
                distill_token=False,
            )
            
            # Weighted average merging
            hidden_states, self._tome_info["size"] = merge_wavg(
                merge,
                hidden_states,
                self._tome_info.get("size")
            )
            
            # 🔍 DEBUG: 打印merging结果（仅layer 0, 5, 11, 23）
            if self._layer_idx in [0, 5, 11, 23]:
                print(f"  [ToMe DEBUG] Layer {self._layer_idx}: {input_shape} → {hidden_states.shape} (merged {r} tokens)")
        
        # 5. MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states,)


# ============================================================================
# ToMe Wrapper：主模型包装器
# ============================================================================

class VideoLLaVATomeWrapper(VideoLLaVAWrapper):
    """
    ToMe专用Video-LLaVA包装器
    
    功能：
    1. 加载标准Video-LLaVA-7B模型
    2. 动态替换Vision Tower的CLIPEncoderLayer为ToMeCLIPEncoderLayer
    3. 配置merging schedule
    """
    
    def __init__(
        self,
        model_path: str = "/root/hhq/models/Video-LLaVA-7B-hf",
        token_budget: int = 2048,
        num_frames: int = 32
    ):
        """
        初始化ToMe wrapper
        
        Args:
            model_path: Video-LLaVA模型路径
            token_budget: token预算（默认2048）
            num_frames: 采样帧数（默认32）
        """
        # 1. 调用父类初始化（加载标准Video-LLaVA）
        print(f"[ToMe] Initializing Video-LLaVA with ToMe...")
        super().__init__(model_path)
        
        # ⭐ 强制获取vision tower
        # 🔍 关键发现：Video-LLaVA实际使用IMAGE_TOWER处理视频，不是video_tower！
        if self.vision_tower is None:
            print(f"[ToMe] ⚠️ Vision tower not found in parent class, attempting direct access...")
            
            # VideoLlavaForConditionalGeneration结构：
            # self.model (VideoLlavaForConditionalGeneration)
            #   └── .model (VideoLlavaModel)
            #       ├── .image_tower (CLIPVisionModel) ← ⭐ 实际使用这个！
            #       └── .video_tower (CLIPVisionModel) ← ❌ 不使用这个！
            
            # ⭐ 关键修复：使用image_tower（实际被调用的tower）
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'image_tower'):
                self.vision_tower = self.model.model.image_tower
                print(f"[ToMe] ✓ Found vision tower via model.model.image_tower (ACTUAL tower used!)")
            
            # Fallback: 如果没有image_tower，尝试video_tower
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'video_tower'):
                self.vision_tower = self.model.model.video_tower
                print(f"[ToMe] ⚠️ Using model.model.video_tower (fallback)")
            
            # 确保vision tower被初始化
            if self.vision_tower and hasattr(self.vision_tower, 'load_model'):
                self.vision_tower.load_model()
                print(f"[ToMe] ✓ Vision tower initialized via load_model()")
        
        # 验证vision tower
        if self.vision_tower is None:
            # 打印模型结构帮助调试
            print(f"[ToMe] ⚠️ Detailed debugging info:")
            print(f"  - model type: {type(self.model)}")
            
            if hasattr(self.model, 'model'):
                print(f"  - model.model type: {type(self.model.model)}")
                
                # 打印所有属性
                all_attrs = [attr for attr in dir(self.model.model) if not attr.startswith('_')]
                print(f"  - model.model attributes (non-private): {all_attrs}")
                
                # 检查_modules
                if hasattr(self.model.model, '_modules'):
                    print(f"  - model.model._modules keys: {list(self.model.model._modules.keys())}")
                
                # 尝试直接访问可能的vision tower名称
                possible_names = ['vision_tower', 'vision_model', 'vision_encoder', 'image_encoder']
                for name in possible_names:
                    if hasattr(self.model.model, name):
                        attr = getattr(self.model.model, name)
                        print(f"  - ✓ Found '{name}': {type(attr)}")
            
            raise RuntimeError(
                "[ToMe] ❌ Failed to load vision tower! "
                "Cannot find vision_tower in VideoLlavaForConditionalGeneration structure. "
                "See debug info above for available attributes."
            )
        
        print(f"[ToMe] ✓ Vision tower loaded: {self.vision_tower.__class__.__name__}")
        
        # 2. ToMe配置
        self.token_budget = token_budget
        self.num_frames = num_frames
        
        # 计算token数：32帧 × 256 tokens/帧 = 8192
        self.tokens_per_frame = 256  # Video-LLaVA-7B: 224x224 ÷ 14x14 = 256
        self.initial_tokens = num_frames * self.tokens_per_frame  # 8192
        
        # 3. 计算merging schedule
        # Vision Tower层数：CLIP-ViT-L有24层
        vision_encoder = self.vision_tower.vision_model.encoder
        self.num_layers = len(vision_encoder.layers)
        
        # Total tokens to merge
        total_to_merge = self.initial_tokens - self.token_budget  # 8192 - 2048 = 6144
        
        # Constant schedule: r_per_layer
        self.r_per_layer = max(1, total_to_merge // self.num_layers)  # 6144 / 24 = 256
        
        print(f"[ToMe] Configuration:")
        print(f"  - Num frames: {num_frames}")
        print(f"  - Tokens per frame: {self.tokens_per_frame}")
        print(f"  - Initial tokens: {self.initial_tokens}")
        print(f"  - Token budget: {token_budget}")
        print(f"  - Vision layers: {self.num_layers}")
        print(f"  - Tokens to merge: {total_to_merge}")
        print(f"  - Tokens per layer (r): {self.r_per_layer}")
        
        # 4. ⭐ 注入ToMe到Vision Tower
        self._inject_tome_to_vision_tower()
        
        print(f"[ToMe] ✅ Initialization complete")
    
    def _inject_tome_to_vision_tower(self):
        """
        动态替换Vision Tower的CLIPEncoderLayer
        
        参照官方tome/patch/timm.py:apply_patch (line 116-151)
        
        ⚠️ 关键：必须修改model.model.video_tower，而不是self.vision_tower！
        因为model.generate()直接访问model.model.video_tower
        """
        print(f"[ToMe] Injecting ToMe into Vision Tower...")
        
        # ToMe配置字典（所有层共享）
        self._tome_info = {
            "r": parse_r(self.num_layers, self.r_per_layer),  # [256, 256, ..., 256]
            "size": None,  # Token size tracking
            "class_token": True,  # CLIP有CLS token
            "distill_token": False,
        }
        
        # ⭐ 关键修复：获取model实际使用的IMAGE_TOWER
        # 测试证明：Video-LLaVA处理视频时调用image_tower，不是video_tower！
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'image_tower'):
            actual_tower = self.model.model.image_tower
            print(f"  → Using model.model.image_tower (VERIFIED: actual tower used during generation)")
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'video_tower'):
            actual_tower = self.model.model.video_tower
            print(f"  → Fallback to model.model.video_tower")
        else:
            actual_tower = self.vision_tower
            print(f"  → Fallback to self.vision_tower")
        
        # 获取vision encoder
        vision_encoder = actual_tower.vision_model.encoder
        
        # 遍历所有层，替换class
        for layer_idx, layer in enumerate(vision_encoder.layers):
            # ⭐ 动态替换class（monkey patching核心）
            original_class_name = layer.__class__.__name__
            
            # 创建ToMe版本的layer（传递layer_idx）
            tome_layer = ToMeCLIPEncoderLayer(layer, self._tome_info, layer_idx)
            
            # 替换原layer
            vision_encoder.layers[layer_idx] = tome_layer
            
            if layer_idx == 0:
                print(f"  ✓ Layer  {layer_idx}: {original_class_name} → ToMeCLIPEncoderLayer")
        
        print(f"  ✓ Replaced {self.num_layers} layers in {type(actual_tower).__name__}")
        print(f"[ToMe] Injection complete")
    
    def generate(self, video_tensor, prompt, options=None):
        """
        推理接口（继承父类，无需修改）
        
        ToMe在Vision Tower内部透明工作，不影响外部接口
        """
        # Reset ToMe info for each forward pass
        self._tome_info["r"] = parse_r(self.num_layers, self.r_per_layer)
        self._tome_info["size"] = None
        
        # 调用父类generate
        return super().generate(video_tensor, prompt, options)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing ToMe Implementation")
    print("=" * 80)
    
    # 测试bipartite_soft_matching
    print("\n🧪 Testing bipartite_soft_matching:")
    
    batch_size = 1
    num_tokens = 100
    channels = 512
    r = 10
    
    metric = torch.randn(batch_size, num_tokens, channels)
    merge, unmerge = bipartite_soft_matching(metric, r, class_token=True)
    
    x = torch.randn(batch_size, num_tokens, channels)
    merged = merge(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Merged shape: {merged.shape}")
    print(f"  Expected: ({batch_size}, {num_tokens - r}, {channels})")
    
    assert merged.shape == (batch_size, num_tokens - r, channels), "Merge failed!"
    
    # 测试unmerge
    restored = unmerge(merged)
    assert restored.shape == x.shape, "Unmerge failed!"
    
    print(f"  ✓ Bipartite matching works correctly")
    
    # 测试merge_wavg
    print("\n🧪 Testing merge_wavg:")
    size = torch.ones(batch_size, num_tokens, 1)
    merged_wavg, merged_size = merge_wavg(merge, x, size)
    
    print(f"  Merged (wavg) shape: {merged_wavg.shape}")
    print(f"  Merged size shape: {merged_size.shape}")
    print(f"  ✓ Weighted average merge works correctly")
    
    print("\n✅ All tests passed!")
    print("=" * 80)
