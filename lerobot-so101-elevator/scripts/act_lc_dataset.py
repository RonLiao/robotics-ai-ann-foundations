# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class ACTLCDataset(Dataset):
    """
    包裝原生的 LeRobotDataset，使其在產出每筆資料時，
    能自動根據該筆資料的 task_index 從 meta/info.json 中提取對應的文字指令，
    並將其放入 'language_instruction' 欄位中，供 ACT-LC 模型使用。
    """
    def __init__(self, repo_id: str, *args, **kwargs):
        # 載入原生的 LeRobotDataset
        self.dataset = LeRobotDataset(repo_id, *args, **kwargs)
        
    def __getattr__(self, name):
        # 讓包裝類別能夠透傳原生 Dataset 的所有屬性 (例如 .meta, .fps, .features 等)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        task_str = ""
        
        # 嘗試取得 current item 的 task_index
        if "task_index" in item:
            task_idx = item["task_index"]
            # 如果是 Tensor，轉為 int
            if isinstance(task_idx, torch.Tensor):
                task_idx = task_idx.item()
                
            # 從 dataset.meta.tasks 中尋找對應的文字指令
            if hasattr(self.dataset, "meta") and hasattr(self.dataset.meta, "tasks"):
                tasks_info = self.dataset.meta.tasks
                
                # 相容不同的 LeRobot 版本結構 (dict 或 list)
                if isinstance(tasks_info, dict):
                    if task_idx in tasks_info:
                        val = tasks_info[task_idx]
                        task_str = val.get("task", "") if isinstance(val, dict) else str(val)
                    elif str(task_idx) in tasks_info:
                        val = tasks_info[str(task_idx)]
                        task_str = val.get("task", "") if isinstance(val, dict) else str(val)
                elif isinstance(tasks_info, list) and task_idx < len(tasks_info):
                    val = tasks_info[task_idx]
                    task_str = val.get("task", "") if isinstance(val, dict) else str(val)
        
        # 若找不到特定指令，給予空字串防呆
        item["language_instruction"] = task_str
        
        return item
