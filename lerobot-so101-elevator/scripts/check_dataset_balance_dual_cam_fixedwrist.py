import unicodedata
from collections import Counter
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pandas as pd

def get_visual_width(text):
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F', 'A') else 1 for c in text)

def pad_to_width(text, width):
    current_width = get_visual_width(text)
    return text + ' ' * max(0, width - current_width)

def check_balance(repo_id):
    print(f"正在載入本地資料集: {repo_id}...")
    try:
        dataset = LeRobotDataset(repo_id)

        tasks_list = []
        obj_tasks = getattr(dataset, "tasks", None)
        if obj_tasks is None and hasattr(dataset, "meta"):
            obj_tasks = getattr(dataset.meta, "tasks", None)
        if obj_tasks is not None:
            if isinstance(obj_tasks, (pd.Series, pd.DataFrame)):
                tasks_list = obj_tasks.index.tolist()
            elif isinstance(obj_tasks, list):
                tasks_list = obj_tasks

        episodes_tasks = []
        ep_data = None
        if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
            ep_data = dataset.meta.episodes
        elif hasattr(dataset, "episode_metadata"):
            ep_data = dataset.episode_metadata

        if ep_data is not None:
            for i in range(len(ep_data)):
                item = ep_data[i]
                val = None
                for k in ["task_index", "task_id", "label", "tasks"]:
                    val = item.get(k) if isinstance(item, dict) else getattr(item, k, None)
                    if val is not None:
                        break
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    val = val[0]
                elif hasattr(val, "tolist"):
                    val = val.tolist()[0]
                if isinstance(val, str):
                    episodes_tasks.append(val)
                elif val is not None:
                    idx = int(val)
                    episodes_tasks.append(tasks_list[idx] if idx < len(tasks_list) else f"Idx:{idx}")
                else:
                    episodes_tasks.append("Unknown Task")

        if not episodes_tasks:
            print("資料集中無 Episode 記錄。")
            return

        counter = Counter(episodes_tasks)
        print(f"\n資料集平衡檢查: {repo_id}")
        col1_w, col2_w = 40, 15
        header = f"{pad_to_width('Task 指令', col1_w)} | {pad_to_width('Episode 數量', col2_w)}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for task, count in sorted(counter.items()):
            print(f"{pad_to_width(task, col1_w)} | {pad_to_width(str(count), col2_w)}")
        print("-" * len(header))
        print(f"總計 Episode: {len(episodes_tasks)}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_balance("RonLiao/lerobot-so101-elevator-6btn-dual-cam-fixedwrist")
