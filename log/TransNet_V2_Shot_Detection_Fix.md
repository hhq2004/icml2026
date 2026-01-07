# TransNet V2 Shot Detection 修复完整指南

## 📋 问题诊断

**当前问题**：EventGraph-LMM准确率只有 **30.48%** (822/2697)

**根本原因**：Shot Detection使用了fallback方法（固定2秒窗口），导致：
- 事件切分不准确
- 关键推理事件被错误分割或合并
- CELF无法选择到正确的reasoning-critical events

**论文要求**：TransNet V2专业shot detection算法

---

## 🔧 修复内容

### 1. TransNet V2是什么？

**TransNet V2** = PyPI库 + 预训练深度学习模型

- **官方库**：`transnetv2-pytorch`
- **功能**：专业的shot boundary detection（检测镜头切换）
- **优势**：
  - 检测abrupt transitions（硬切）
  - 检测gradual transitions（淡入淡出等）
  - 输出variable-length segments（符合论文要求）

**为什么需要 `shot_detector.py`？**
1. ✅ TransNet V2需要复杂的预处理（resize、归一化、tensor转换）
2. ✅ 需要NMS后处理避免过度碎片化
3. ✅ 提供fallback机制保证向后兼容
4. ✅ 模块化设计，未来其他方法也可复用

---

### 2. 修复的文件

#### 文件1: `utils/shot_detector.py`

**修复A：TransNet V2 API调用**
```python
# ❌ 修复前（错误API）
predictions_batch = self.model.predict_frames(frames_batch)

# ✅ 修复后（正确API）
predictions_batch = self.model(frames_batch)[0]  # 返回(predictions, indices)
```

**修复B：添加NMS（非极大值抑制）**
```python
# ❌ 修复前：直接阈值化，导致过度碎片化
boundary_frames = np.where(frame_predictions > threshold)[0]

# ✅ 修复后：使用NMS过滤密集boundaries
boundary_frames = self._find_boundaries_with_nms(
    frame_predictions, threshold, min_distance=15  # 两个boundary至少相距15帧
)
```

**NMS效果示例**：
- 输入：234个候选boundaries（几乎每帧都是boundary）
- 输出：28个精选boundaries（真正的镜头切换点）

**修复C：输入预处理**
```python
# ✅ 使用TransNet V2论文标准尺寸
pil_img = pil_img.resize((48, 27), Image.BILINEAR)  # W x H
frames_batch = frames_batch.astype(np.float32) / 255.0  # 归一化到[0,1]
frames_batch = torch.from_numpy(frames_batch).permute(0, 3, 1, 2)  # (B,3,27,48)
```

#### 文件2: `methods/eventgraph.py`

**修复：添加device设置**
```python
# 在__init__中添加
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 📁 需要同步到远端的文件

### 核心修復文件（2个）

```bash
# 1. Shot detector修复
main_code/utils/shot_detector.py

# 2. EventGraph device设置
main_code/methods/eventgraph.py
```

### 测试文件（可选，建议同步）

```bash
# 3. TransNet V2功能测试
main_code/test_transnet_v2.py

# 4. 完整pipeline测试脚本
main_code/Script/test_eventgraph_fixed.sh
```

---

## 🚀 远端执行流程

### Step 1: 同步文件到百度云

```bash
# 将上述4个文件从本地复制到百度云服务器
# 路径：/root/hhq/main_code/...
```

### Step 2: 安装TransNet V2

```bash
# SSH到百度云服务器
cd /root/hhq

# 安装transnetv2-pytorch库（包含预训练模型）
pip install transnetv2-pytorch

# 验证安装
python -c "from transnetv2_pytorch import TransNetV2; print('✅ Installed')"
```

**预期输出**：
```
✅ Installed
```

### Step 3: 测试TransNet V2功能

```bash
cd /root/hhq/main_code

# 运行单视频测试
python test_transnet_v2.py
```

**这个测试的逻辑**：
1. ✅ 检查TransNet V2是否安装
2. ✅ 从VideoMME数据集自动找一个视频
3. ✅ 运行shot detection
4. ✅ 展示检测到的events数量和时间段
5. ✅ 验证NMS是否工作

**预期输出示例**：
```
================================================================================
TransNet V2 Shot Detector - Functional Test
================================================================================

✅ TransNet V2 is available

[0/3] Finding test video...
  ✓ Using: /root/hhq/dataset/Video-MME/videos/xxxxx.mp4

[1/3] Initializing TransNet V2...
[TransNet V2] Loading pretrained model...
  ✓ TransNet V2 loaded on cuda

[2/3] Detecting shots in: /root/hhq/dataset/Video-MME/videos/xxxxx.mp4
[TransNet V2] Detecting shots in: xxxxx.mp4
  Video info: 12000 frames, 30.00 fps
  Processing in batches of 100 frames...
    Processed 500/12000 frames
    ...
    Processed 12000/12000 frames
  Prediction stats: min=0.001, max=0.987, mean=0.123
  NMS: 234 candidates → 28 boundaries  # ← NMS工作正常！
  ✓ Detected 28 shots

[3/3] Postprocessing...
  ✓ After merging short segments: 26 events

================================================================================
✅ Detection Complete!
================================================================================
Total Events: 26

First 10 Events:
  Event  1:   0.00s -  15.23s  (duration: 15.23s)
  Event  2:  15.23s -  32.45s  (duration: 17.22s)
  Event  3:  32.45s -  48.67s  (duration: 16.22s)
  ...
  Event 10: 135.20s - 151.43s  (duration: 16.23s)
  ... and 16 more events

================================================================================
✅ TEST PASSED
================================================================================
```

**❌ 如果失败**：
```
❌ TransNet V2 not available!
Install with: pip install transnetv2-pytorch
```

### Step 4: 运行完整EventGraph-LMM测试（2697样本）

```bash
cd /root/hhq/main_code/Script

# Option A: 使用修复后的测试脚本
bash test_eventgraph_fixed.sh

# Option B: 使用原有的parallel脚本
bash run_eventgraph_parallel.sh
```

**完整测试的逻辑**：
1. ✅ 对2697个VideoMME视频运行EventGraph-LMM
2. ✅ 每个视频使用TransNet V2进行shot detection
3. ✅ 构建事件图
4. ✅ CELF选择关键事件
5. ✅ Graph-CoT推理
6. ✅ 计算准确率并与修复前对比

**预期输出**：
```
================================================================================
EventGraph-LMM - TransNet V2修复后完整测试
================================================================================

🧪 Step 1: 测试TransNet V2功能...
✅ TransNet V2测试通过！

🚀 Step 2: 运行EventGraph-LMM (2697样本，4卡并行)...
[GPU 0] Processing samples 0-674...
[GPU 1] Processing samples 675-1349...
[GPU 2] Processing samples 1350-2024...
[GPU 3] Processing samples 2025-2696...

✅ 所有进程完成！正在合并结果...

================================================================================
📈 EventGraph-LMM 性能报告 (TransNet V2修复后)
================================================================================
  总样本数: 2697
  正确数:   XXX
  准确率:   XX.XX%
================================================================================

🔍 对比修复前:
  - 修复前准确率: 30.48% (822/2697)
  - 修复后准确率: XX.XX% (XXX/2697)
  - 改进幅度:     +X.XX 百分点

✅ 修复成功！准确率显著提升！  # ← 如果改进>=3个百分点
================================================================================
```

---

## 🔍 如何判断修复是否成功

### 查看日志中的关键指标

**成功标志**：
```
[TransNet V2] Detecting shots in: xxx.mp4
  ✓ Detected 28 shots  # ← 3-100之间为合理
  NMS: 234 candidates → 28 boundaries  # ← NMS有效工作
```

**失败标志**：
```
⚠️ TransNet V2 not available, using fallback  # ← 还在用fallback
⚠️ Fallback: Fixed 2s window, 150 segments   # ← 没用TransNet V2
```

### 准确率判断

| 准确率范围 | 判断 | 下一步 |
|-----------|------|--------|
| **≥ 35%** | ✅ 修复非常成功 | 继续优化CELF和超参数 |
| **33-35%** | ✅ 修复成功 | Shot Detection已解决 |
| **31-33%** | ⚠️ 有改进但不明显 | 检查CELF选择逻辑 |
| **< 31%** | ❌ 改进不明显 | 确认TransNet V2是否真正运行 |

---

## 📊 预期改进

### Before（使用Fallback）
```
固定2秒窗口切分
→ 忽略真实镜头转换
→ 关键事件被错误分割
→ CELF选择错误片段
→ 准确率: 30.48%
```

### After（使用TransNet V2 + NMS）
```
TransNet V2专业检测
→ 精确识别镜头切换（abrupt/gradual）
→ NMS避免过度碎片化
→ 语义完整的事件段
→ CELF选择correct reasoning-critical events
→ 预期准确率: 33.5-40%
```

**预期提升**：**+3至9.5个百分点**

---

## ⚠️ 注意事项

### 1. 向后兼容性
- ✅ 修改仅影响EventGraph-LMM
- ✅ 其他baseline（Q-Frame, SceneGraph-Cap, ToMe, FastV, DyCoke）不受影响
- ✅ TransNet V2失败会自动fallback

### 2. GPU内存
- TransNet V2在GPU上运行
- 如果GPU内存不足，会自动使用CPU（较慢）

### 3. 首次运行会下载模型
- TransNet V2首次运行会自动下载预训练模型（~100MB）
- 需要网络连接

---

## 🐛 故障排查

### 问题1: TransNet V2安装失败
```bash
# 尝试升级pip
pip install --upgrade pip

# 重新安装
pip install transnetv2-pytorch --no-cache-dir
```

### 问题2: CUDA不可用
```bash
# 检查CUDA
nvidia-smi

# 检查PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题3: 准确率提升不明显
**排查步骤**：
1. 查看日志确认TransNet V2是否真正运行
2. 检查shot数量是否合理（3-100个）
3. 查看NMS是否工作（candidates → boundaries）
4. 如果以上都正常，问题可能在CELF或PageRank

---

## 📝 修改总结

| 文件 | 修改内容 | 代码行数 | 影响 |
|------|---------|---------|------|
| `utils/shot_detector.py` | 修复API + 添加NMS | +47行 | ✅ 仅EventGraph |
| `methods/eventgraph.py` | 添加device设置 | +4行 | ✅ 仅EventGraph |
| **其他baseline** | 无修改 | 0 | ✅ 不受影响 |

---

## 🎯 成功标准

修复成功的三个指标：
1. ✅ TransNet V2正常加载和运行（日志中无"fallback"字样）
2. ✅ shot数量合理（3-100个events）
3. ✅ 准确率提升至少3个百分点（30.48% → 33.5%+）

---

**修复完成时间**: 2026-01-07  
**预期提升**: 30.48% → 33.5-40.0%  
**状态**: ✅ 代码修复完成，等待远程测试验证
