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
  - **[已完成]** 第二步：多任務資料集混合 (Multi-task Data Collection)。
    - **[已完成]** 建立 `scripts/record_6btn.sh` 多任務錄製輔助腳本與平衡檢查工具。
    - **[已完成]** 錄製 3 顆按鈕的 demonstrations (目標每顆 70+ Episodes)。
      - **[已完成]** 解決 Headless 環境下的重置超時與資料集編輯崩潰問題，並實現 5 秒全自動倒數機制。
      - **[已完成]** 按鈕 1 (`press button 1`) 完成 50 次錄製。
      - **[已完成]** 按鈕 2 (`press button 2`) 完成 50 次錄製。
      - **[已完成]** 按鈕 3 (`press button 3`) 完成 50 次錄製。
      - **[已完成]** 三個按鍵各補錄 20 個 Episode（50 → 70），維持資料集平衡後重新訓練。
      - **[待進行]** 完成其餘 3 顆按鈕 (按鈕 4~6) 的錄製（保留後續擴充）。
  - **[已完成]** 第三步：啟動條件式訓練 (Language-Conditioned Training)。
    - v1 模型（各 50 Episodes）：實作 `scripts/train_act_lc.py`，完成首次 100K 步訓練，Loss 收斂至 0.034。
    - **[已完成]** 實作自定義推論路由 `inference_language_act.py`。
    - **[已完成]** 實作 Deep Model Hotfix 解決 Config 殘缺問題（手動注入編碼組件）。
    - **[已完成]** 實作基於位移監控的任務完成「自動靜止停止機制」。
    - **[已完成]** 解決 `observation.state` 歸一化參數對齊問題，修復手臂亂舞行為（自動同步 `stats.json` 並實作 `(x-mean)/std` 正規化與反正規化）。
    - **[已完成]** 完成首次實機推論部署，手臂可平滑執行軌跡不亂舞，推論引擎正常運作。
    - **[已完成]** 建立 `scripts/check_train_frames.py` 與 `--save_frame` 推論首幀存檔機制，並完成訓練集與推論時相機視角比對，確認視角一致（Camera Calibration Drift 已排除）。
    - **[已完成]** 補錄資料並重訓 (v2)，驗證是否解決 Mode Collapse 導致定位失敗的問題
    - v2 模型（各 70 Episodes）：補錄資料後重訓，上傳至 `RonLiao/so101-elevator-act-lc-btn-1-to-3-v2`。
  - **[已完成]** 第四步：實機推論部署。
    - 實作 `scripts/inference_language_act.py` 推論中控台，含自動靜止停止機制。
    - 修正歸一化 (`stats.json`) 對齊問題，解決手臂亂舞問題。
    - 診斷並修正 Language Model 不一致問題（`bert` 誤用為 `distilbert`），確認 v2 模型語言條件已生效。
    - **[已確認瓶頸]** 全景相機視角在「最後 5cm」缺乏近距離視覺反饋，無法精確區分相鄰按鈕。
  - **[進行中]** 第五步：手眼相機雙相機 v3 方案。
    - **[已完成]** 確認手眼相機 device index（`/dev/video2`），建立雙相機錄製腳本 `scripts/record_6btn_dual_cam.sh`。
    - **[已完成]** 建立新 Dataset Repo (`RonLiao/lerobot-so101-elevator-6btn-dual-cam`) 並重新錄製（各 50 Episodes）。
    - **[已完成]** 訓練含手眼視角的 dualcam 模型，上傳至 `RonLiao/so101-elevator-act-lc-btn-1-to-3-dualcam`。
    - **[已完成]** 新建 `scripts/inference_language_act_dualcam.py` 支援雙相機推論，Dummy 測試通過。
    - **[進行中]** 實機推論部署除錯：3 顆按鈕均無法精準按壓。
      - **[已排除]** 雙相機視角對齊：推論首幀與訓練集截圖高度吻合（front + wrist 均正常），手眼相機無偏移。
      - **[已排除]** Stats 歸一化對齊：Dummy 測試確認 `stats_dualcam.json` 已正確載入（state mean/std 數值合理），此項排除。
      - **[已修正，重訓完成]** Checkpoint 中 `text_proj` 權重遺失（根本原因：Monkey-patch 僅替換 ACTPolicy 未替換 ACTConfig，導致語言條件從未啟用）：已修正 `train_act_lc.py` 與 `modeling_act.py`，以修正後腳本重訓 100K 步（`act_lc_btn_1_to_3_dualcam_v2`）。重訓 Log 確認 `num_total_params=118M`（vs bug 版 52M），66M 差值即為 DistilBERT，語言條件已生效。
    - **[已完成]** 驗證 v2 fixed checkpoint 含 `text_proj` key（`torch.allclose` 確認權重正確載入）。
    - **[已完成]** 實機推論部署。Dummy 測試確認語言條件有效（button 1 vs button 2 在 Step 100 位移量差達三倍），但實機三個按鈕指令仍落在同一位置（按鍵 1/3 中間偏右）。
    - **[已確認瓶頸]** 語言條件技術上生效，但語言信號強度不足以克服三顆按鈕視覺特徵高度相似的問題，模型在 100K 步 / 50 Episodes 訓練量下仍向空間均值坍塌（Mode Collapse）。
    - **[已完成]** 補錄至每顆按鈕 100 Episodes（新增 50 集全由標準起始位置錄製，強化樣本密度），上傳至 `RonLiao/lerobot-so101-elevator-6btn-dual-cam`（共 300 Episodes）。
    - **[已完成]** 以 200K 步重訓 v3 模型（`act_lc_btn_1_to_3_dualcam_v3`）：Loss 0.034，grdn ~2.4，語言條件生效但實機仍 Mode Collapse。
    - **[已完成]** VAE z 群集分析（`scripts/analyze_z_clusters.py`）：確認 z 分離比 = 0.05，z 未攜帶任務資訊，z-dropout 假設排除；**根本原因確認為語言 token 在 620-token Self-Attention 序列中被視覺 token 稀釋**。
  - **[進行中]** 第六步：架構改善——可學習 Task Embedding（以 FiLM 調製取代語言 token concat）。
    - **[已完成]** text_scale 實驗（×1、×3、×5）確認問題不在信號強度，排除信號衰減假設。
    - **[已完成]** VAE z 群集分析（`scripts/analyze_z_clusters.py`）：分離比 = 0.05，確認 z 未攜帶任務資訊，排除 z-dropout 假設；根本原因確認為 Self-Attention 中語言 token 被視覺 token 稀釋。
    - **[已完成]** 設計並實作 act_te 架構（`policies/act_te/`）：繼承 vanilla ACTConfig，新增 `num_tasks` 欄位；以 `nn.Embedding + FiLM` 取代 DistilBERT，語言條件施加於 Encoder 輸出後、Decoder 前，梯度路徑最短且無 token 競爭。
    - **[已完成]** 建立訓練腳本 `scripts/train_act_te.py` 與推論腳本 `scripts/inference_act_te_dualcam.py`。
    - **[已完成]** 更新架構圖 `docs/assets/ACT_TE_Architecture.png`：Text 走側路 → Language Encoder → FiLM，不再塞進 Encoder 序列。
    - **[已完成]** act_te v1 訓練至 200K 步（100K→200K resume），最終 Loss=0.034、grdn=1.73（本專案最低），WandB：[rf7pgq2v](https://wandb.ai/ron-liao-nuwa-robotics/lerobot-so101-elevator-te-dualcam/runs/rf7pgq2v)。
    - **[已完成]** Dummy test：100K 1.45×；200K 1.14×，Task Embedding 有效分化。
    - **[進行中]** 實機推論精度優化：Mode Collapse 已解決，成功率 30-40%，末端偏差 ~1cm（精度優化中）。
    - **[待進行]** 完成其餘 3 顆按鈕 (按鈕 4~6) 的錄製（保留後續擴充）。

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
- **學習進度**：
  - **[已完成]** LeRobot 六大模組框架解析（scripts / policies / datasets / robot_devices / config / envs）
  - **[已完成]** Hydra Config + Registry 機制：`--robot.type=so101_follower` 如何透過 Registry 查找並實例化對應驅動
  - **[已完成]** ACT + SO-101 移植案例：確認 `train.py` / `record.py` 不需修改，移植工作完全在 policies / robot_devices 層完成
  - **[已完成]** ACT-LC 客製化實作：`train_act_lc.py`（Monkey-patch）、`inference_language_act.py`（互動式推論 + 靜止偵測）
  - **[已完成]** Dataset 層三大客製化點：`lerobot-record` 參數決定 features schema、`delta_timestamps` 決定時間窗口、`__getitem__` Wrapper 注入額外欄位（ACTLCDataset）
  - **[已完成]** robot_devices 層架構：`robots/`（SO101Follower / SO101Leader）與 `cameras/`（OpenCV / RealSense）平行結構；`SO101Follower.capture_observation()` / `send_action()` vs `SO101Leader.get_action()` 的介面差異
  - **[已完成]** `lerobot-record` 錄製模式的完整 Leader→Follower→Dataset 每幀資料流
  - **[待進行]** Trace `lerobot-record` 指令的實際 script：從 `pyproject.toml` 入口出發，逐步追蹤 `record.py` 如何解析 `--robot.type` / `--teleop.type`、呼叫 `make_robot_from_config()`、進入 `robots/utils.py` Registry 查找，最終連結到 `so101_follower.py` / `so101_leader.py` 的實際實作
