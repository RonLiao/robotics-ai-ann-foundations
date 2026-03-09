# 通用電梯按鈕壓印 (General Elevator Pressing)

這是在 **robotics-ai-ann-foundations** 專案下，專注於具身智能（Embodied AI）實作的子項目。

## 專案目標

實現能跨越不同面板設計的通用電梯按鈕識別與按壓任務。

- **硬體架構**：使用 SO-101 機器人手臂。
- **軟體框架**：基於 LeRobot 框架進行開發。
- **技術路徑**：利用 ACT (Action Chunking with Transformers) 進行動作生成，並結合底層的 PID 控制確保執行精度。

## 專案目錄結構說明

- `docs/`：放置實作過程的筆記、疑難排解記錄與備忘命令集。
    - `cheatsheet.md`: 常用腳本與啟動命令備忘錄。
- `configs/`：放置機器人馬達校正檔 (`calibration/`) 與訓練參數配置。
- `scripts/`：預計放置後續用於資料收集、訓練與推論的啟動腳本。
- `data/`：預計放置收集到的 hdf5 格式 demonstration 數據。
- **Hugging Face Dataset**: [RonLiao/lerobot-so101-elevator-dataset](https://huggingface.co/datasets/RonLiao/lerobot-so101-elevator-dataset)

## 當前狀態

1. 已設定馬達 ID 和校正，可達成兩隻手臂連動。
    - **Wrist Roll 關節問題觀測紀錄 (2026-03-05)**：
        - *現象描述*：Leader Arm 的 `wrist_roll` 關節在校正時，讀取的 MIN/MAX 範圍會隨初始啟動角度不同而產生劇烈偏移，甚至出現負值或大於 4095 的情況。
        - *驗證結論*：已透過 `watch_motor_position.py` 確認，只要不執行 `lerobot-calibrate`，馬達重新上電後中心和兩極限角度皆會**保持不變**。因此不影響訓練和推論階段的使用，可利用 `read_motor_info.py` 輔助觀察。
2. 已確認相機與兩手遙控指令運作正常。
