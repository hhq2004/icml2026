# /root/hhq/main_code/datasets/longvideobench.py
import os
import json
from .base_dataset import BaseDataset

class LongVideoBench(BaseDataset):
    def __init__(self, root_dir, duration_mode="all"):
        super().__init__(root_dir)
        
        # 路径配置
        self.json_path = os.path.join(root_dir, "LongVideoBench", "lvb_val.json")
        self.video_root = os.path.join(root_dir, "LongVideoBench", "videos")
        
        # 如果文件不存在，仅打印警告（为了不阻断其他数据集运行）
        if not os.path.exists(self.json_path):
            print(f"Warning: LongVideoBench json not found at {self.json_path}")
            return

        print(f"[LongVideoBench] Loading from {self.json_path}...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.samples = []
        for item in data:
            # 这里的字段根据 LVB 实际 json 结构调整
            # 假设 item 包含 video_id, question, candidates, correct_choice
            video_id = item.get('video_id', '')
            # 尝试拼接视频路径 (.mp4)
            video_path = os.path.join(self.video_root, f"{video_id}.mp4")
            
            # 只有当文件存在时才加入(可选)
            # if not os.path.exists(video_path): continue

            self.samples.append({
                "id": item.get('id', video_id),
                "video_path": video_path,
                "question": item.get('question', ''),
                "options": item.get('candidates', []), # 选项列表
                "answer": item.get('correct_choice', 'C'), # A/B/C/D
                "duration": item.get('duration', 0)
            })
            
        print(f"[LongVideoBench] Loaded {len(self.samples)} samples.")