# /root/hhq/main_code/datasets/videomme.py
import os
import json
import re
from .base_dataset import BaseDataset

class VideoMME(BaseDataset):
    def __init__(self, root_dir, duration_mode="all"):
        super().__init__(root_dir)
        
        self.json_path = os.path.join(root_dir, "Video-MME", "video_mme.json")
        self.video_root = os.path.join(root_dir, "Video-MME", "videos") 
        
        print(f"[VideoMME] Loading json from {self.json_path}...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 1. 建立视频索引 (这步必须保留，否则找不到文件)
        self.video_map = self._build_video_index(self.video_root)
        
        self.samples = []
        video_count = 0
        
        print("🚀 [VideoMME] FORCE LOAD MODE: Loading EVERYTHING found on disk...")
        
        for item in data:
            # 2. 查找视频路径
            video_id = str(item.get('video_id', '')).strip()
            url = item.get('url', '')
            
            fpath = self.video_map.get(video_id)
            if not fpath and url:
                yt_id = self._extract_youtube_id(url)
                if yt_id: fpath = self.video_map.get(yt_id)
            if not fpath and video_id.startswith("_"):
                fpath = self.video_map.get(video_id[1:])

            # 如果找不到文件，物理上没法跑，跳过
            if not fpath: 
                continue
            
            video_count += 1

            # 3. 构造样本 (不做任何筛选)
            # 给一个默认时长防止除零报错
            dummy_duration = 3600 

            for q in item['questions']:
                self.samples.append({
                    "id": f"{video_id}_{q['question_id']}",
                    "video_path": fpath,
                    "duration": dummy_duration,
                    "question": q['question'],
                    "options": q['options'],
                    "answer": q['answer'],
                    "task_type": q['task_type']
                })
        
        print(f"✅ Successfully loaded {len(self.samples)} samples from {video_count} videos.")

        # === 冒烟测试强制截断已禁用 ===
        # 现在由 run_inference.py 的 --max_samples 参数控制样本数
        # ============================

    def _build_video_index(self, root_dir):
        print(f"    -> Indexing videos in {root_dir}...")
        idx = {}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.mkv', '.webm', '.avi')):
                    name = os.path.splitext(file)[0]
                    full_path = os.path.join(root, file)
                    idx[name] = full_path
                    if name.startswith("_"):
                        idx[name[1:]] = full_path
        print(f"    -> Indexed {len(idx)} files.")
        return idx

    def _extract_youtube_id(self, url):
        if not isinstance(url, str): return None
        patterns = [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"]
        for p in patterns:
            match = re.search(p, url)
            if match: return match.group(1)
        return None