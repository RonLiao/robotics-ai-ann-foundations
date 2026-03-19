# 03 - LeRobot 框架深度解析與工具鏈 (Framework Anatomy)

這篇筆記旨在從架構層面理解 LeRobot 框架如何整合資料集、模型與監控工具，並記錄這些工具在具身智能（Embodied AI）工作流中的具體角色。

本章節以 [02-practice-circle-magnet](docs/02-practice-circle-magnet.md) 的 **Dataset/Model/Monitoring** 為例，展示 LeRobot 框架如何整合資料集、模型與監控工具，並記錄這些工具在具身智能（Embodied AI）工作流中的具體角色。
  - [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/RonLiao/lerobot-so101-elevator-dataset)
  - [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/RonLiao/so101-elevator-act)
  - [![WandB Monitoring](https://img.shields.io/badge/Weights%20%26%20Biases-Monitoring-FF4654?logo=weightsandbiases)](https://wandb.ai/ron-liao-nuwa-robotics/lerobot-so101-elevator)

---

## 第一章：Hugging Face 工具介紹 (Hugging Face Tools)

Hugging Face 在專案中扮演了資料與權重的託管倉庫，以下是其核心功能解析。

### 1.1 Datasets & Models
- **Datasets**：存放「教材（訓練數據）」。由於影像數據巨大，HF 提供了版本管理（Revision）與高效分流機制。
- **Models**：存放「成果（訓練權重）」。訓練出的 `model.safetensors` 是 AI 學習後的智慧結晶。

### 1.2 Dataset Viewer 介面指南
這是在 Hugging Face 網頁上即時預覽數據的強大工具：
- **長條圖 (Histograms)**：位於表格標題下方，代表該欄位數據的 **統計分佈 (Distribution)**。
    - 例如 `episode_index` 的分佈顯示每段錄製的長度；`timestamp` 的分佈可確認採樣是否均勻。
- **"Use this dataset" 按鈕**：提供一鍵複製的程式碼片段，方便在其他環境載入數據。
- **"Edit dataset card" 按鈕**：編輯該儲存庫的說明文件 (README)。這是向外界展示實驗環境、硬體型號與經驗的重要窗口。

---

## 第二章：Weights & Biases (訓練監控工具)

- **用途**：即時監控 Loss 下降曲線、GPU 資源佔用以及學習率變化。
- **意義**：判斷模型是否收斂，作為調整超參數（Batch Size, LR）的科學依據。

---

## 第三章：LeRobot 框架深度解析 (LeRobot Framework Analysis)

本章節解析 LeRobot 如何定義數據標準，這是縮小硬體與模型間差距的關鍵。

### 3.1 核心定義檔案：meta/info.json
`meta/info.json` 是由 LeRobot 框架定義的規範，決定了 AI 模型如何「閱讀」錄製結果。
- 總量與規格 (Summary Statistics)
  - **`total_episodes: 51`**：
    - **本次實例**：總共錄製了 51 次成功的圓形磁鐵按壓動作。
  - **`total_frames: 13947`**：
    - **本次實例**：這 51 次錄製累積產生了近一萬四千個影格數據。
  - **`fps: 30`**：
    - **本次實例**：系統每秒記錄 30 次影像與座標，提供 ACT 模型所需的連續動作序列。
  - **`splits`**：
    - **本次實例**：設定 `{"train": "0:51"}`，代表目前所有 51 個錄製段落均用於模型學習。

- 特徵欄位 (Features) - 模型最在意的部分
這是 AI 訓練時的輸入與輸出定義：
  - **`action` (預測動作)**：
    - **本次實例**：AI 學習後應輸出的 SO-101 六個馬達位置（肩、肘、腕、夾爪）。模型預測下一秒手臂應移動到的目標座標。
  - **`observation.state` (目前狀態)**：
    - **本次實例**：手臂感測器回傳的當前座標。幫助模型理解目前位置與目標磁鐵的距離。
  - **`observation.images.front` (視覺輸入)**：
    - **本次實例**：前置攝像頭拍攝的 640x480 色彩影像。AI 透過此影像辨識圓形磁鐵的位置。
  - **`task_index` (任務索引)**：
    - **用途**：區分同一個資料集內的不同任務（例如「按按鈕」與「開門」）。
    - **本次實例**：由於本專案目前僅包含「圓形磁鐵按壓」一項任務，故數值均固定為 `0`。

- 資料儲存結構 (Data Path)
  - **`data_path / video_path`**：這是描述檔案是怎麼分層存放的（例如放在 data/chunk-000）。
  - **`Parquet 格式`**：您在 Hub 網頁上看到的表格（如第二張圖），其實是讀取 .parquet 檔案。這是一種非常適合大數據的高效能檔案格式，AI 訓練時讀取它的速度比讀取 CSV 快得多。

### 3.3 如何統一 Dataset 與 Model 間的差距 (Connecting the Gap)

LeRobot 框架透過以下機制實現「硬體與模型間的無縫對接」：

1. **統一特徵映射 (Feature Mapping)**：
    將不同型號機器人的數據統化為 `action` 和 `observation` 名稱，使模型架構具備跨平台重用性。
2. **設定檔自動連結 (Config Linking)**：
    訓練時讀取 `info.json` 自動配置模型輸入維度，減少手動修改程式碼的出錯率。
3. **Hub 透明化傳輸**：
    透過 `repo_id` 實現本地快取與雲端數據的透明同步，降低資料管理的難度。

---
