# 通用電梯按鈕壓印 (General Elevator Pressing)

這是在 **robotics-ai-ann-foundations** 專案下，專注於具身智能（Embodied AI）實作的子項目。

## 專案目標

實現能跨越不同面板設計的通用電梯按鈕識別與按壓任務。

- **硬體架構**：使用 SO-101 機器人手臂。
- **軟體框架**：基於 LeRobot 框架進行開發。
- **技術路徑**：利用 ACT (Action Chunking with Transformers) 進行動作生成，並結合底層的 PID 控制確保執行精度。

## 專案目錄結構說明

- `docs/`：實作過程的詳細筆記、各階段的操作流程與故障排除。
    - [01-setup-and-calibration.md](docs/01-setup-and-calibration.md)：環境建置、Docker 設定與手臂校正筆記。
    - [02-practice-circle-magnet.md](docs/02-practice-circle-magnet.md)：練習任務（圓形磁鐵）的錄製、訓練與推論實作紀錄。
- `configs/`：放置機器人馬達校正檔 (`calibration/`) 與訓練參數配置。
- `scripts/`：資料收集、數據驗證與監控馬達位置的工具腳本。
- `record/`：錄製的 demonstrations 數據（.parquet 與影片）。
- **Hugging Face Dataset**: [RonLiao/lerobot-so101-elevator-dataset](https://huggingface.co/datasets/RonLiao/lerobot-so101-elevator-dataset)

## 當前狀態

1. **環境與校正**：已建立支援 GPU 加速與 32GB Shared Memory 的 Docker 容器 (`ron_so101_v2`)。裝置權限與手臂校正檔已透過 symbolic link 連結，確保錄製與訓練一致性。
2. **硬體觀測**：確認 Leader Arm 的 `wrist_roll` 關節異常可透過「不重複執行 `lerobot-calibrate`」來規避，詳見 [01-setup-and-calibration.md](docs/01-setup-and-calibration.md)。

## 開發規劃 (Roadmap)

本任務拆分為兩個階段進行：

### 階段一：練習任務 (按壓牆上圓形磁鐵)
- **目標**：熟悉 LeRobot 完整工作流程，確保從錄製、訓練到推論的軟硬體工作正常。
- **當前進度 (2026-03-12 更新)**：
  - **[已完成]** 兩隻手臂與前置相機 (640x480) 的遠端遙控連動測試。
  - **[已完成]** 註冊 [wandb.ai](https://wandb.ai) 取得 API Key 並於 Server48 登入，準備開始訓練與監控 Loss 曲線。
  - **[已完成]** 錄制 50 個 Episodes
  - **[已完成]** 重建Docker容器，增加容器的ShareMemory和GPU支援
  - **[已完成]** 整合救回的 50 個 Episodes
  - **[已完成]** 數據存放於新容器快取路徑，並與 Hugging Face 帳號完成認證連線。
  - **[已完成]** 執行 `wandb login` 登入即時監控儀表板。
  - **[進行中]** 執行 `lerobot-train` 啟動 ACT 模型訓練。
  - **[進行中]** 利用 WandB 監控 Loss 曲線 (已從 6.8 穩定收斂至 2.4 以下)。

### 階段二：正式任務 (通用電梯按壓)
- **目標**：收集具備多樣性的真實電梯數據，訓練具備泛化能力的通用模型。
- **作法**：上傳正式資料至 Hugging Face，調整超參數訓練大模型，並評估不同電梯面板的按壓成功率。
