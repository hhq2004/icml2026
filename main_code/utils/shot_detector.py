#!/usr/bin/env python3
"""
Professional Shot Boundary Detection using TransNet V2
论文引用: EventGraph-LMM Section 3.2 (Lines 606-608)
"using a standard event-based shot boundary detection algorithm (Soucek & Lokoc, 2024)"

TransNet V2论文:
- Title: TransNet V2: An effective deep network architecture for fast shot transition detection
- Authors: Tomáš Souček & Jakub Lokoč  
- Published: 2020
- GitHub: https://github.com/soCzech/TransNetV2
"""

import torch
import numpy as np
from pathlib import Path

try:
    from transnetv2_pytorch import TransNetV2
    TRANSNET_AVAILABLE = True
except ImportError:
    TRANSNET_AVAILABLE = False
    print("⚠️  TransNet V2 not available. Install with: pip install transnetv2-pytorch")


class TransNetV2Detector:
    """
    TransNet V2 Shot Boundary Detector
    
    完全复现论文Section 3.2的event decomposition:
    "We first decompose the long video into a sequence of N variable-length visual segments"
    
    特性:
    - 使用预训练的TransNet V2模型
    - 检测abrupt和gradual transitions
    - 输出variable-length segments（符合论文描述）
    """
    
    def __init__(self, device='cuda'):
        """
        初始化TransNet V2检测器
        
        Args:
            device: 'cuda' 或 'cpu'
        """
        if not TRANSNET_AVAILABLE:
            raise ImportError(
                "TransNet V2 is required for shot detection. "
                "Install with: pip install transnetv2-pytorch"
            )
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 加载预训练的TransNet V2模型
        print("[TransNet V2] Loading pretrained model...")
        self.model = TransNetV2()
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ TransNet V2 loaded on {self.device}")
    
    def detect_shots(self, video_path, threshold=0.5):
        """
        检测视频中的shot boundaries
        
        Args:
            video_path: 视频文件路径
            threshold: shot detection阈值 (默认0.5，论文标准)
        
        Returns:
            events: list of (start_time, end_time) tuples
                   表示N个variable-length视频片段
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"[TransNet V2] Detecting shots in: {Path(video_path).name}")
        
        # 使用decord读取视频帧（避免ffmpeg依赖，与项目其他部分一致）
        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("decord is required. Install with: pip install decord")
        
        # 读取视频
        vr = VideoReader(str(video_path), ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        print(f"  Video info: {total_frames} frames, {fps:.2f} fps")
        
        # TransNet V2 API要求:
        # - 输入shape: [1, num_frames, 27, 48, 3]（视频序列格式，不是batch）
        # - 数据类型: torch.uint8
        # - 值范围: [0, 255]
        
        # 由于GPU内存限制，我们分批处理，每次最多500帧
        batch_size = 500
        all_predictions = []
        
        print(f"  Processing video in chunks of {batch_size} frames...")
        
        for start_idx in range(0, total_frames, batch_size):
            end_idx = min(start_idx + batch_size, total_frames)
            chunk_size = end_idx - start_idx
            
            # 读取当前chunk的帧
            indices = list(range(start_idx, end_idx))
            frames_chunk = vr.get_batch(indices).asnumpy()  # (chunk_size, H, W, 3)
            
            # Resize到TransNet V2标准尺寸
            from PIL import Image
            resized_frames = []
            for frame in frames_chunk:
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((48, 27), Image.BILINEAR)  # W x H
                resized_frames.append(np.array(pil_img))
            
            # 转换为视频序列格式: [1, num_frames, 27, 48, 3]
            video_tensor = np.stack(resized_frames, axis=0)  # (chunk_size, 27, 48, 3)
            video_tensor = np.expand_dims(video_tensor, axis=0)  # (1, chunk_size, 27, 48, 3)
            video_tensor = torch.from_numpy(video_tensor).to(torch.uint8)
            video_tensor = video_tensor.to(self.device)
            
            # TransNet V2推理
            with torch.no_grad():
                predictions_chunk = self.model(video_tensor)[0]  # 返回(predictions, indices)
            
            # 转换为numpy并立即squeeze成1D数组
            if torch.is_tensor(predictions_chunk):
                predictions_chunk = predictions_chunk.cpu().numpy()
            
            # 确保是1D数组（移除所有大小为1的维度）
            predictions_chunk = np.squeeze(predictions_chunk)
            
            all_predictions.append(predictions_chunk)
            
            # 清理显存
            del video_tensor
            torch.cuda.empty_cache()
            
            if end_idx % 1000 == 0 or end_idx == total_frames:
                print(f"    Processed {end_idx}/{total_frames} frames")
        
        # 合并所有chunk的预测结果
        predictions = np.concatenate(all_predictions, axis=0)
        
        # 确保是1D数组
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
        
        print(f"  Prediction stats: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")
        
        # 使用NMS找到shot boundaries
        boundary_frames = self._find_boundaries_with_nms(predictions, threshold, min_distance=15)
        
        # 转换为事件段
        events = self._boundaries_to_events(
            boundary_frames,
            total_frames=total_frames,
            fps=fps
        )
        
        print(f"  ✓ Detected {len(events)} shots")
        return events
    
    def _find_boundaries_with_nms(self, predictions, threshold=0.5, min_distance=15):
        """
        使用NMS（非极大值抑制）找到shot boundaries
        避免过于密集的检测导致过度碎片化
        
        Args:
            predictions: (num_frames,) 每帧的transition概率
            threshold: shot detection阈值
            min_distance: 两个boundary之间的最小帧数间隔
        
        Returns:
            boundary_frames: boundary帧索引数组
        """
        # 找到所有超过阈值的候选边界
        candidates = np.where(predictions > threshold)[0]
        
        if len(candidates) == 0:
            return np.array([], dtype=np.int64)
        
        # 非极大值抑制：只保留局部极大值
        boundaries = []
        i = 0
        while i < len(candidates):
            # 取当前候选作为起点
            current_frame = candidates[i]
            current_score = predictions[current_frame]
            
            # 查找min_distance范围内的最大值
            j = i
            while j < len(candidates) and candidates[j] - current_frame < min_distance:
                if predictions[candidates[j]] > current_score:
                    current_frame = candidates[j]
                    current_score = predictions[current_frame]
                j += 1
            
            # 保存这个局部最大值
            boundaries.append(current_frame)
            
            # 跳过已处理的候选
            i = j
        
        boundary_frames = np.array(boundaries, dtype=np.int64)
        print(f"  NMS: {len(candidates)} candidates → {len(boundary_frames)} boundaries")
        
        return boundary_frames
    
    def _boundaries_to_events(self, boundary_frames, total_frames, fps):
        """
        将boundary帧索引转换为时间段
        
        Args:
            boundary_frames: boundary帧的索引数组
            total_frames: 视频总帧数
            fps: 视频帧率
        
        Returns:
            events: list of (start_time, end_time)
        """
        if len(boundary_frames) == 0:
            # 没有检测到边界，返回整个视频作为一个事件
            duration = total_frames / fps
            return [(0.0, duration)]
        
        events = []
        
        # 第一个事件：从0到第一个boundary
        if boundary_frames[0] > 0:
            start_time = 0.0
            end_time = boundary_frames[0] / fps
            events.append((start_time, end_time))
        
        # 中间事件：从一个boundary到下一个boundary
        for i in range(len(boundary_frames) - 1):
            start_time = boundary_frames[i] / fps
            end_time = boundary_frames[i + 1] / fps
            events.append((start_time, end_time))
        
        # 最后一个事件：从最后一个boundary到视频结尾
        if boundary_frames[-1] < total_frames - 1:
            start_time = boundary_frames[-1] / fps
            end_time = total_frames / fps
            events.append((start_time, end_time))
        
        return events
    
    def _get_fps(self, video_path):
        """
        获取视频帧率
        
        Args:
            video_path: 视频路径
        
        Returns:
            fps: 帧率
        """
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(video_path), ctx=cpu(0))
            fps = vr.get_avg_fps()
            return fps
        except Exception as e:
            print(f"  ⚠️  Could not get FPS, defaulting to 30: {e}")
            return 30.0  # 默认30fps
    
    def merge_short_segments(self, events, min_duration=0.5):
        """
        合并过短的事件段（保证语义完整性）
        
        论文没有明确要求，但作为后处理有助于避免过度碎片化
        
        Args:
            events: list of (start, end)
            min_duration: 最小持续时间（秒），默认0.5秒
        
        Returns:
            merged_events: 合并后的事件列表
        """
        if len(events) == 0:
            return events
        
        merged = []
        current_start, current_end = events[0]
        
        for start, end in events[1:]:
            duration = current_end - current_start
            
            if duration < min_duration:
                # 当前段太短，合并到下一段
                current_end = end
            else:
                # 当前段足够长，保存并开始新段
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # 添加最后一段
        merged.append((current_start, current_end))
        
        return merged


# Fallback: PySceneDetect（如果TransNet V2不可用）
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False


class PySceneDetectFallback:
    """
    备用方案：使用PySceneDetect进行shot detection
    
    注意：这不是论文引用的方法，仅作为fallback
    """
    
    def __init__(self):
        if not PYSCENEDETECT_AVAILABLE:
            raise ImportError("PySceneDetect not available")
        print("[Fallback] Using PySceneDetect instead of TransNet V2")
    
    def detect_shots(self, video_path, threshold=30.0):
        """
        使用PySceneDetect检测shot boundaries
        
        Args:
            video_path: 视频路径
            threshold: content detector阈值
        
        Returns:
            events: list of (start_time, end_time)
        """
        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        scene_list = scene_manager.get_scene_list()
        
        events = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            events.append((start_time, end_time))
        
        video_manager.release()
        
        print(f"  ✓ Detected {len(events)} shots (PySceneDetect)")
        return events


if __name__ == "__main__":
    # 简单测试
    print("=" * 80)
    print("Testing TransNet V2 Shot Detector")
    print("=" * 80)
    
    # 测试视频路径（需要替换为实际路径）
    test_video = "/root/hhq/dataset/VideoMME/video/001.mp4"
    
    if TRANSNET_AVAILABLE:
        detector = TransNetV2Detector()
        events = detector.detect_shots(test_video)
        
        print(f"\n检测结果:")
        print(f"  - 事件数: {len(events)}")
        print(f"  - 前5个事件:")
        for i, (start, end) in enumerate(events[:5], 1):
            duration = end - start
            print(f"    Event {i}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
    else:
        print("\n⚠️  TransNet V2 not installed")
        print("Install with: pip install transnetv2-pytorch")
    
    print("=" * 80)
