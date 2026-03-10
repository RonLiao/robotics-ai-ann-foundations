# 02 - 練習任務：按壓圓形磁鐵 (Practice Run)

這篇筆記記錄正式挑戰「通用電梯按鈕」之前，為了熟悉 LeRobot 完整工作流程而設計的「按壓牆上圓形磁鐵」練習任務。主要目的是向外展示我實際走通了整套系統，並留下具體的步驟備忘與實作的踩坑經驗。

## 練習目標

1. **走通完整管線 (Pipeline)**：親手跑過**資料收集 (Data Collection)**、**模型訓練 (Model Training)** 到**實機推論 (Inference)** 的流程，確保能掌控所有的軟硬體腳本。
2. **驗證硬體配置**：透過這次小規模的資料錄製，確認攝影機視角與手臂運動範圍是否合理。
3. **快速試錯**：不追求完美的泛化能力，先求有、再求好，快速生出一個懂「按壓」的 ACT 模型。

## 第一步：資料收集 (Data Collection)

使用 `lerobot-record` 指令啟動遙控模式，直接操作 Leader Arm 來示範按壓牆上的圓形磁鐵。

- **硬體配置**：
  - Leader Arm (操作端)：`/dev/ttyACM0`
  - Follower Arm (執行端)：`/dev/ttyACM1`
  - 前置相機：`/dev/video0` (設定為 640x480, 30fps)
- **錄製計畫**：打算先錄製 50 個 demonstrations (episodes)。

*(詳細啟動指令備忘於 `cheatsheet.md`)*

**錄製過程中的實務經驗與觀察：**
- (待補充：實際錄製時遇到什麼困難？例如夾爪是否會遮擋視線、動作需不需要因為相機幀率而刻意放慢等)
- 環境會撞到東西所限，無論是leader arm還是follower arm都不能做到從任何初始角度的示範
- 承上，因此抓取的50次無法涵蓋所有初始角度，或許會在之後訓練或推論時出問題
- 對一個簡單任務如這個按圓磁鐵來說，是否有縮小錄制次數的可能，還是50次已經是讓模型能收斂的最小次數了？

## 第二步：設定視覺化監控 (WandB)

採用 LeRobot 原生支援之 **Weights & Biases (WandB)** 監控訓練收斂狀況。避開 Server 端 Port Forwarding 配置。

**配置要點：**
1. 於 [wandb.ai](https://wandb.ai) 取得 API Key。
2. 訓練指令中啟用 `--wandb.enable=True` 以自動建立專案頁面。
3. 之後即可在手機或其他筆電透過 WandB 儀表板隨時觀察即時 Loss 變化。

## 第三步：模型訓練 (Model Training)

(待補充：紀錄訓練腳本命令、預訓練模型選用、Batch Size/Epochs 設定及收斂情形)

## 第四步：實機推論 (Inference)

(待補充：紀錄如何載入 checkpoint 進行自動控制測試，以及實際按壓磁鐵的成功率與手臂行為表現)
