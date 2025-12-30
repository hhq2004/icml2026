# /root/hhq/main_code/datasets/base_dataset.py
import os
import json
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 子类不需要重写这个，只需要在 __init__ 里填充 self.samples
        return self.samples[idx]