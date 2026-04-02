# 通用電梯按鈕按壓 (General Elevator Pressing)

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/RonLiao/lerobot-so101-elevator-dataset)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/RonLiao/so101-elevator-act)
[![WandB Monitoring](https://img.shields.io/badge/Weights%20%26%20Biases-Monitoring-FF4654?logo=weightsandbiases)](https://wandb.ai/ron-liao-nuwa-robotics/lerobot-so101-elevator)

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
    - [03-lerobot-framework-anatomy.md](docs/03-lerobot-framework-anatomy.md)：LeRobot 框架深度解析、資料結構與工具鏈原理。
    - [04-practice-6-button-panel.md](docs/04-practice-6-button-panel.md)：特定電梯面板（6顆按鈕）語意語言驅動多任務模型訓練紀錄。
    - `05-vla-elevator-pressing.md`：(規劃中) 視覺-語言-動作大模型 (VLA) 導入與實作紀錄。
- `configs/`：放置機器人馬達校正檔 (`calibration/`) 與訓練參數配置。
- `scripts/`：資料收集、數據驗證與監控馬達位置的工具腳本。
- `record/`：錄製的 demonstrations 數據（.parquet 與影片）。
- **Hugging Face Dataset**: [RonLiao/lerobot-so101-elevator-dataset](https://huggingface.co/datasets/RonLiao/lerobot-so101-elevator-dataset)

## 當前狀態

1. **環境與校正**：已建立支援 GPU 加速與 32GB Shared Memory 的 Docker 容器 (`ron_so101_v2`)。裝置權限與手臂校正檔已透過 symbolic link 連結，確保錄製與訓練一致性。
2. **硬體觀測**：確認 Leader Arm 的 `wrist_roll` 關節異常可透過「不重複執行 `lerobot-calibrate`」來規避，詳見 [01-setup-and-calibration.md](docs/01-setup-and-calibration.md)。

## 開發規劃 (Roadmap)

本任務拆分為四個階段進行：

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
  - **[已完成]** 啟動 ACT 模型訓練。
  - **[已完成]** 利用 WandB 監控 Loss 曲線 (最終收斂至 2.4)。
  - **[已完成]** 將錄製的 Dataset 與訓練成果同步至 Hugging Face。
  - **[已完成]** 訓練完模型實機推論驗證測試。

### 階段二：特定面板多任務按壓 (Language-Conditioned ACT 模型)
- **目標**：由輸入的字串決定按哪個按鈕。針對特定款式（6 顆按鈕），採用「層次二」解決方案（條件式多任務 ACT 模型），透過文字指令的切換直接驅動單一模型按壓不同按鈕，驗證語言與影像特徵的區辨聯結能力。
- **當前進度**：
  - **[已完成]** 撰寫 [04-practice-6-button-panel.md](docs/04-practice-6-button-panel.md) 紀錄語意驅動模型架構與任務規劃。
  - **[已完成]** 了解 LeRobot 原生 ACT 模型實作（原始碼與資料流），並於 `modeling_act.py` 加入註解，完成前置作業。
  - **[已完成]** 第一步：修改 ACT 網路架構 (注入 Text Embeddings)。完成建立客製化的 `act_lc` 模型目錄，並透過測試腳本驗證了結合文字編碼器等跨模態網路架構的資料流。
  - **[進行中]** 第二步：多任務資料集混合 (Multi-task Data Collection)。
    - **[已完成]** 建立 `scripts/record_6btn.sh` 多任務錄製輔助腳本與平衡檢查工具。
    - **[進行中]** 錄製 6 顆按鈕的 demonstrations (目標每顆 50+ Episodes)。
      - **[已完成]** 解決 Headless 環境下的重置超時與資料集編輯崩潰問題，並實現 5 秒全自動倒數機制。
      - **[已完成]** 按鈕 1 (`press button 1`) 完成 50 次錄製。
      - **[待進行]** 完成其餘 5 顆按鈕 (按鈕 2~6) 的錄製，確保資料集數量平衡。
  - **[待進行]** 第三步：啟動條件式訓練 (Language-Conditioned Training)。
    - **[待進行]** 準備並執行 `scripts/train_act_lc.py` 啟動 6 顆按鈕的多任務混合訓練。

### 階段三：視覺-語言-動作大模型 (VLA) 實作
- **目標**：邁步「層次三」的前沿技術，直接引入 VLA (Vision-Language-Action) 多模態大模型。將驗證它強大的網路常識與 Zero-shot 推論潛力，以自然語言端到端控制複雜影像的按壓行為。
- **當前進度**：
  - **[待進行]** 未來預計將實作細節紀錄於 `docs/05-vla-elevator-pressing.md`。

### 階段四：正式任務 (通用電梯泛化按壓測試)
- **目標**：結合前述階段的架構經驗，收集具備極大多樣性的真實電梯數據，進一步評估所選之大模型框架對於現實世界完全未知的電梯面板，其泛化與按壓成功率。
- **作法**：上傳百種真實電梯之資料集至 Hugging Face，調整超參數訓練終端泛化通用模型。

### 持續性任務：框架解析與經驗累積
- **目標**：隨著各階段模型的訓練實作，持續深入剖析與記錄 LeRobot 框架、HuggingFace 以及 WandB 的底層架構與進階工具鏈使用心得。
- **紀錄文件**：[03-lerobot-framework-anatomy.md](docs/03-lerobot-framework-anatomy.md)
