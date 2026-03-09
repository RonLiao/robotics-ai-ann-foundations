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

*(詳細的啟動與參數指令，已經備忘在同目錄的 `cheatsheet.md` 裡了)*

**錄製過程中的實務經驗與觀察：**
- (待補充：實際錄製時遇到什麼困難？例如夾爪是否會遮擋視線、動作需不需要因為相機幀率而刻意放慢等)

## 第二步：設定視覺化監控 (WandB)

因為是親手練模型，必須要能看到 Loss 曲線才知道模型有沒有在收斂。這裡選擇使用 LeRobot 原生支援的 **Weights & Biases (WandB)**，好處是不用去搞 Server 端的 Port Forwarding。

**設定步驟：**
1. 在 [wandb.ai](https://wandb.ai) 取得帳號的 API Key。
2. 在 Server48 的 Docker 容器中直接登入：
   ```bash
   wandb login
   ```
   (貼上 API Key 即可。之後只要開始啟動訓練，就能隨時用手機或筆電登入 WandB 看即時的 Loss 變化與收斂狀況。)

## 第三步：模型訓練 (Model Training)

(待補充：紀錄實際執行的訓練腳本命令、選用的預訓練模型、Batch Size 與 Epochs 的設定，以及在 WandB 上觀察到的收斂狀況)

## 第四步：實機推論 (Inference)

(待補充：紀錄如何載入 checkpoint 進行自動控制測試，以及實際按壓磁鐵的成功率與手臂行為表現)
