# 01 - 環境建置與手臂校正筆記

本文件記錄了 LeRobot SO-101 電梯按壓專案在環境準備、裝置校準以及一些重要參考資料的過程。

## 1. 環境建置過程

我們在 `Server48` 主機上使用 Docker 運行整個控制環境：

1. **第一次建立 Docker**：
   根據 [LeRobot 安裝指南](https://huggingface.co/docs/lerobot/installation) 安裝 ffmpeg 與 LeRobot 套件。

2. **裝置對應與權限宣告**：
   每次重新啟動容器，請務必確定宿主機（Host）上有順利抓到 USB 轉列埠 (`/dev/ttyACM0`, `/dev/ttyACM1`) 與相機 (`/dev/video0`)，並進入 docker 給予對應的權限 (`chmod 666`)。

## 2. 關於手臂校正的經驗

進行雙臂的遙控操作前，須完成 Leader Arm 與 Follower Arm 的校正。

- **硬體確認**：可以透過建立虛擬 ID 用 `lerobot-calibrate` 快速確認手臂是否處於連線狀態。
- **異常問題追蹤（重要）**：
  Leader Arm 的 `wrist_roll` 關節（手把旋轉）馬達存在異常。目前觀察到，同樣的物理角度，在每次重開機後回報的 raw data 都不同，該馬達的角度疑似在開機時會被重設為 2048。這可能會影響資料錄製時動作的精確對應，需持續追蹤。

## 3. 面臨的下一步工作

在環境建置與連動測試成功後，目前的待辦清單：
- 錄製數據 ([資料錄製官方指引](https://huggingface.co/docs/lerobot/il_robots))
- 進行模型訓練
- 執行實際推論按壓任務

## 4. 推薦參考資源

這些官方文件與技術文章對於理解及排解 LeRobot 在實體硬體上的運作非常有幫助：

- **HuggingFace 官方範例：**
  - [10_use_so100](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md) (LeRobot 範例)
  - [7_get_started_with_real_robot](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md)
- **優質技術文章與解析：**
  - [LeRobot——Hugging Face打造的机器人开源库：包含对顶层script、与dataset的源码分析(含在简易机械臂SO-ARM100上的部署)](https://blog.csdn.net/v_JULY_v/article/details/139692392)
