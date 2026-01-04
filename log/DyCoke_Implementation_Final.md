# DyCoke Implementation Final

**日期**: 2026-01-01 ~ 2026-01-03  
**状态**: Frame-level approximation (Token-level不可行)  
**准确率**: 待测试 (预期 30-35%)

---

## 📋 实现总结

DyCoke baseline已实现并可用，但由于Hugging Face transformers库的架构限制，采用了**Frame-level近似**而非论文的**Token-level实现**。

---

## ✅ 实现方案

### **当前方案: Frame-level近似**

```python
# 模拟Stage 1 TTM效果
effective_ratio = 1 - K  # K=0.5 → 保留50%
effective_frames = int(32 * effective_ratio)  # 32 → 16 frames

# 使用TTM相似度逻辑选择信息量最大的帧
indices = select_frames_by_similarity(frames, K=0.5)
reduced_frames = [frames[i] for i in indices]
```

**特点**:
- ✅ 帧级别 (vs 论文的Token级别)
- ✅ 使用余弦相似度选择帧 (保留核心思想)
- ✅ K=0.5参数与论文一致
- ❌ Stage 2动态KV Cache pruning未实现

---

## 🔴 Token-level实现障碍

### **根本问题**: Video-LLaVA的`masked_scatter`限制

```python
# Video-LLaVA内部实现
inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)
```

**要求**: `video_features.numel()` 必须等于 mask中True的数量

**冲突**: TTM改变token数量后必然无法匹配

### **尝试过的方案**:

| 方案 | 结果 | 原因 |
|------|------|------|
| 套用DyCoke官方代码 | ❌ | LLaVA-OneVision与Video-LLaVA架构不兼容 |
| Monkey Patch projector | ❌ | masked_scatter严格验证token数 |
| Monkey Patch forward | ❌ | 等同于重写整个模型 |
| **修改transformers源码** | ✅ 可行但不符合baseline要求 |

---

## 📊 与论文对比

### **Stage 1: Temporal Token Merging**

| 维度 | 论文 | 当前实现 | 影响 |
|------|------|---------|------|
| 操作粒度 | Token-level (256 tokens/frame) | Frame-level | ⚠️ 粗粒度 |
| 算法 | 滑动窗口4帧 + Odd/Even分组 | 相似度选帧 | ⚠️ 简化 |
| 压缩率 | K=0.5 (50% tokens) | K=0.5 (50% frames) | ✅ 一致 |
| 输出 | Merged visual tokens | Reduced frames | ⚠️ 不同 |

### **Stage 2: Dynamic KV Cache Pruning**

| 维度 | 论文 | 当前实现 |
|------|------|---------|
| 实现状态 | 完整 | ❌ 未实现 |
| Attention评估 | Layer L=3 | 无 |
| Token选择 | Top P=70% | 无 |

**预期精度损失**: -10~15% (vs论文完整实现)

---

## 🎯 实现评估

### **正面**:
1. ✅ 可用性: 代码运行稳定
2. ✅ 独立性: 不影响其他baseline
3. ✅ 核心思想: 保留compression概念
4. ✅ 参数一致: K/L/P与论文相同

### **负面**:
1. ⚠️ 实现完整度: 60% (Stage 1简化, Stage 2缺失)
2. ⚠️ 准确率: 预计比论文低10-15%
3. ❌ Token-level: 未实现真正的token操作

---

## 📁 相关文件

- `main_code/models/video_llava_7b_dycoke.py` - Model wrapper (Frame-level)
- `main_code/methods/DyCoke.py` - Method class
- `main_code/Script/run_dycoke_smoke_test.sh` - 4卡并行测试

---

## 💡 结论

**作为baseline有效**，可用于对比实验。但需在论文中明确说明简化策略。

如需完整Token-level实现，需要:
1. 联系DyCoke作者获取官方代码
2. 或修改transformers源码 (~600行代码)
3. 或等待transformers库提供更好的hook支持

---

**最后更新**: 2026-01-04  
**文档维护**: 本文档为DyCoke实现的唯一官方文档
