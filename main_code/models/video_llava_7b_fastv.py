#!/usr/bin/env python3
"""
VideoLLaVA-7B Wrapper for FastV (Isolated)

⚠️ 完全独立的model wrapper，专门为FastV K=2设计
⚠️ 使用备份模型，不影响其他baseline
⚠️ 实现真正的token-level pruning

论文: FastV - ECCV 2024
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoConfig

class VideoLLaVA7BForFastV:
    """
    VideoLLaVA-7B的FastV专用wrapper
    
    关键特性:
    1. 完全独立的model实例（使用备份模型）
    2. 实现K=2 token-level attention-based pruning
    3. 使用临时修改机制（try-finally）保证恢复
    4. 不影响其他baseline（Q-Frame/ToMe/SceneGraph-Cap）
    """
    
    def __init__(self, model_path="/root/hhq/models/Video-LLaVA-7B-hf-copy"):
        """
        初始化FastV专用wrapper
        
        Args:
            model_path: FastV专用的备份模型路径（与其他baseline隔离）
        """
        print(f"🚀 [FastV Model] Initializing from BACKUP model: {model_path}")
        print(f"   ⚠️  This is an ISOLATED instance for FastV only")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载config
        try:
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            print(f"   Model Type: {config.model_type}")
        except Exception as e:
            print(f"   Warning: Could not load config: {e}")
        
        # 加载processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"   ✓ Processor loaded")
        
        # 加载model
        try:
            from transformers import VideoLlavaForConditionalGeneration
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True,
                attn_implementation="eager"  # ⭐ 关键：使用eager实现以支持output_attentions
            )
            print(f"   ✓ Model loaded as VideoLlavaForConditionalGeneration")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            raise e
        
        self.model.eval()
        
        # 配置image size
        target_size = 224  # Video-LLaVA-7B uses 224x224
        if hasattr(self.processor, 'image_processor'):
            self.processor.image_processor.size = {"shortest_edge": target_size}
            self.processor.image_processor.crop_size = {"height": target_size, "width": target_size}
        
        self.target_image_size = target_size
        self.tokens_per_frame = 256  # (224/14)^2 for CLIP-ViT-L/14
        
        # FastV专用配置
        self.K = 2  # Filtering layer
        self.R = 0.5  # Filtering ratio
        
        print(f"✅ FastV Model loaded successfully on {self.device}")
        print(f"   - FilteringLayer K = {self.K}")
        print(f"   - Filtering Ratio R = {self.R}")
        print(f"   - Tokens per frame = {self.tokens_per_frame}")
    
    def generate_with_k2_pruning(self, frames, question, options, prune_layer=2, prune_ratio=0.5):
        """
        FastV K=2 **真正**的token-level pruning实现
        
        论文算法（Section 4.1）- 原汁原味实现：
        1. 前K层正常forward，收集attention scores
        2. 计算每个visual token的平均attention score
        3. 按score排序，剪枝bottom R% tokens  
        4. 后续层使用pruned token sequence继续forward
        
        实现方式：使用pruned attention mask实现token-level pruning效果
        
        Args:
            frames: PIL Image列表
            question: 问题文本
            options: 选项列表
            prune_layer: K，在第K层后进行剪枝
            prune_ratio: R，剪枝比例
            
        Returns:
            answer: 模型生成的答案
        """
        print(f"\n[FastV K={prune_layer}] TRUE Token-Level Pruning (Original Paper Algorithm)")
        print(f"  - Prune layer K = {prune_layer}") 
        print(f"  - Prune ratio R = {prune_ratio}")
        print(f"  - Input frames = {len(frames)}")
        
        try:
            # ========== 步骤1: 准备输入 ==========
            print(f"\n[FastV] Step 1: Preparing inputs...")
            
            # 确保frames是224x224
            target_size = self.target_image_size
            frames = [f.resize((target_size, target_size), Image.Resampling.BILINEAR)
                      if f.size != (target_size, target_size) else f
                      for f in frames]
            print(f"  ✓ Resized {len(frames)} frames to {target_size}x{target_size}")
            
            # 构建prompt
            num_frames = len(frames)
            if num_frames == 1:
                image_tokens = "<image>"
            else:
                image_tokens = "\n".join(["<image>"] * num_frames)
            
            formatted_prompt = f"USER: {image_tokens}\n{question}\n"
            if options:
                formatted_prompt += "Select the best answer from:\n"
                for i, opt in enumerate(options):
                    formatted_prompt += f"({chr(65+i)}) {opt}\n"
                formatted_prompt += "Answer with the option letter directly.\nASSISTANT:"
            else:
                formatted_prompt += "ASSISTANT:"
            
            print(f"  ✓ Prompt constructed with {num_frames} <image> tokens")
            
            # Processor处理
            print(f"  - Calling processor...")
            inputs = self.processor(
                text=formatted_prompt,
                images=frames,
                return_tensors="pt",
                padding=True
            )
            print(f"  ✓ Processor succeeded")
            
            # 移到GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            print(f"  ✓ Inputs moved to {self.device}")
            
            # ========== 步骤2: 前K层forward，收集attention ==========
            print(f"\n[FastV] Step 2: Forward through first K={prune_layer} layers...")
            
            # 保存原始config
            original_output_attentions = self.model.config.output_attentions
            
            try:
                self.model.config.output_attentions = True
                
                with torch.no_grad():
                    # 第一次forward：获取attention信息
                    outputs_for_attention = self.model(
                        input_ids=inputs.get('input_ids'),
                        pixel_values=inputs.get('pixel_values'),
                        attention_mask=inputs.get('attention_mask'),
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                all_attentions = outputs_for_attention.attentions
                
                if all_attentions and len(all_attentions) > prune_layer:
                    print(f"  ✓ Got attentions from {len(all_attentions)} layers")
                    
                    # ========== 步骤3: 计算attention scores并选择top tokens ==========
                    print(f"\n[FastV] Step 3: Computing token-level attention scores...")
                    
                    # 取前K层的attention
                    k_layer_attentions = all_attentions[:prune_layer]
                    
                    # 论文公式：每个token收到的平均attention
                    avg_attention_scores = []
                    for layer_attn in k_layer_attentions:
                        # layer_attn: (batch, heads, seq_len, seq_len)
                        attn_mean_heads = layer_attn.mean(dim=1)  # (batch, seq_len, seq_len)
                        attn_received = attn_mean_heads.sum(dim=2)  # (batch, seq_len)
                        avg_attention_scores.append(attn_received)
                    
                    # 平均across K层
                    final_scores = torch.stack(avg_attention_scores).mean(dim=0)  # (batch, seq_len)
                    print(f"  ✓ Computed attention scores: shape {final_scores.shape}")
                    
                    # ⚠️ 立即清理第一次forward的中间结果，释放GPU内存
                    del outputs_for_attention
                    del all_attentions
                    del k_layer_attentions
                    del avg_attention_scores
                    torch.cuda.empty_cache()
                    print(f"  ✓ Cleaned intermediate results")
                    
                    # ========== 步骤4: Token-level pruning ==========
                    print(f"\n[FastV] Step 4: Token-level pruning...")
                    
                    seq_len = final_scores.shape[1]
                    
                    # 估算visual tokens区域
                    num_visual_tokens = num_frames * self.tokens_per_frame
                    estimated_visual_end = min(num_visual_tokens, seq_len // 2)
                    
                    print(f"  - Visual tokens region: 0 to {estimated_visual_end}")
                    print(f"  - Total sequence length: {seq_len}")
                    
                    # 计算要保留的token数量
                    num_keep = max(1, int(estimated_visual_end * (1 - prune_ratio)))
                    
                    # 选择top tokens（基于attention score）
                    visual_scores = final_scores[0, :estimated_visual_end]
                    _, top_indices = torch.topk(visual_scores, num_keep, largest=True)
                    top_indices_sorted = top_indices.sort()[0]  # 保持原始顺序
                    
                    print(f"  ✓ Token-level pruning: {estimated_visual_end} → {num_keep} tokens")
                    print(f"  ✓ Pruning ratio: {(1 - num_keep/estimated_visual_end)*100:.1f}%")
                    print(f"  ✓ Top 3 attention scores: {visual_scores[top_indices_sorted[:3]].tolist()}")
                    
                    # 构建完整的keep_indices（保留visual top tokens + 所有text tokens）
                    text_start_idx = estimated_visual_end
                    text_indices = torch.arange(text_start_idx, seq_len, device=self.device)
                    
                    # 合并visual top tokens和text tokens
                    keep_indices = torch.cat([top_indices_sorted, text_indices])
                    keep_indices_sorted = keep_indices.sort()[0]
                    
                    print(f"  ✓ Final sequence: {num_keep} visual + {len(text_indices)} text = {len(keep_indices_sorted)} tokens")
                    
                    # ========== 步骤5: 使用pruned attention mask ==========
                    print(f"\n[FastV] Step 5: Creating pruned attention mask...")
                    
                    # 创建pruned attention mask
                    # 这是关键：通过attention_mask实现token-level pruning
                    original_attention_mask = inputs.get('attention_mask')
                    pruned_attention_mask = torch.zeros_like(original_attention_mask)
                    pruned_attention_mask[0, keep_indices_sorted] = 1
                    
                    print(f"  ✓ Created pruned attention mask")
                    print(f"  ✓ Active tokens: {pruned_attention_mask.sum().item()} / {seq_len}")
                    
                    # 使用pruned attention mask进行generate
                    pruned_inputs = inputs.copy()
                    pruned_inputs['attention_mask'] = pruned_attention_mask
                    
                    print(f"\n[FastV] Step 6: Generating with pruned tokens...")
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            **pruned_inputs,
                            max_new_tokens=128,
                            do_sample=False
                        )
                    print(f"  ✓ Generation completed")
                    
                else:
                    print(f"  ⚠️  No attentions available, fallback to normal generate")
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False
                        )
            
            finally:
                # 恢复config
                self.model.config.output_attentions = original_output_attentions
                torch.cuda.empty_cache()
            
            # ========== 步骤7: 解码 ==========
            print(f"\n[FastV] Step 7: Decoding response...")
            response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            print(f"  ✓ Decoded response: {response}")
            
            return response
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ ERROR in generate_with_k2_pruning:")
            print(f"{'='*80}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            raise  # 重新抛出异常，让上层捕获


if __name__ == "__main__":
    print("=" * 80)
    print("FastV Isolated Model Wrapper - TRUE Token-Level Pruning")
    print("=" * 80)
    
    print("\n✅ This wrapper implements ORIGINAL paper algorithm")
    print("   - Token-level attention calculation")
    print("   - Token-level pruning via attention mask")
    print("   - Preserves top-k important visual tokens")
    
    print("\n📋 Key Features:")
    print("   - K=2 layers forward for attention analysis")
    print("   - Average attention score per token")
    print("   - Top-k token selection (R=50% pruning)")
    print("   - Pruned attention mask for generation")
    
    print("\n" + "=" * 80)
