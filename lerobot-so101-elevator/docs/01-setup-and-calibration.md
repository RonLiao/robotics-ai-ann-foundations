# 01 - 環境建置與手臂校正筆記

本文件記錄了 LeRobot SO-101 電梯按壓專案在環境準備、裝置校準以及一些重要參考資料的過程。

## 1. 環境建置過程

在 `Server48` 主機上使用 Docker 運行整個控制環境：

1. **建立 Docker 容器（只需執行一次）**

   a. 根據 [LeRobot 安裝指南](https://huggingface.co/docs/lerobot/installation) 安裝 ffmpeg 與 LeRobot 套件。

   b.為了確保訓練與錄製順利，建立容器時需加入 `--shm-size` 參數擴充共享記憶體限制 (建議至少 4GB 甚至 8GB 以上) 以避免 `out of shared memory` 錯誤，同時加上 `--privileged -v /dev:/dev` 以解放硬體權限，並將gpu權限開放給docker。
   ```bash

   # 說明：
   # --shm-size="32g"：解決訓練時 DataLoader 共享記憶體不足的問題。
   # -v /path/to/host/folder:/workspace/shared：將 Host 端資料夾掛載到容器內 (請依實際需求修改 /path/to/host/folder)。
   # --privileged 和 -v /dev:/dev：賦予容器存取 Host 端所有硬體設備的權限，不需再手動逐一指定 --device。
   # --gpus all：讓容器能夠存取 Host 端的 NVIDIA GPU 進行加速運算。
   sudo docker run -d -it --name ron_so101_v2 --gpus all --shm-size="32g" \
     -v /home/ron/SSD4T/ron/shared:/root/shared \
     --privileged -v /dev:/dev ron_so101:latest
   ```

2. **啟動 Docker 容器（第二次以後執行這行）**
   ```bash
   sudo docker start ron_so101_v2
   ```

3. **進入 Docker 容器**
   ```bash
   sudo docker exec -it ron_so101_v2 /bin/bash
   ```

4. **裝置對應與權限宣告**

   每次重新啟動容器或重新插拔硬體，務必確定宿主機（Host）上有順利抓到 USB 轉列埠 (`/dev/ttyACM0`, `/dev/ttyACM1`) 與相機 (`/dev/videoX`)，並進入 docker 給予對應的權限：
   ```bash
   chmod 666 /dev/ttyACM0 /dev/ttyACM1 /dev/video*
   ```

5. **啟動lerobot環境**
   ```bash
   conda activate lerobot
   cd lerobot/src/lerobot/scripts/
   ```

6. **手臂校正（只需執行一次）**

   重新校正手臂：**每次執行都會重置馬達的初始角度，所以校正完成後請勿再執行第二次，否則錄制的訓練資料將會失效**

   ```bash
   lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm
   lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_awesome_follower_arm
   ```
   校正後的設定檔位置
   ```bash
   Follower Arm: `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json`
   Leader Arm: `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json`
   ```

   將校正後的檔案存進github，並用symbolic link連結
   ```bash
   ln -sf /root/shared/robotics-ai-ann-foundations/lerobot-so101-elevator/configs/calibration/my_awesome_follower_arm.json /root/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json
   ln -sf /root/shared/robotics-ai-ann-foundations/lerobot-so101-elevator/configs/calibration/my_awesome_leader_arm.json /root/.cache/huggingface/lerobot/calibration/robots/so101_leader/my_awesome_leader_arm.json
   ```
   經驗：**Leader Arm 的 `wrist_roll` 關節（手把旋轉）馬達偶爾回報負數或大於4095的值**
   
     - **現象描述**：Leader Arm 的 `wrist_roll` 關節在校正時，讀取的 MIN/MAX 範圍會隨初始啟動角度不同而產生劇烈偏移，甚至出現負值或大於 4095 的情況。
     - 目前觀察到，同樣的物理角度，在每次重開機後若**執行 `lerobot-calibrate`**，回報的 raw data 會產生嚴重偏移。
     - **實驗結論**：利用 `watch_motor_position.py` 實時讀取馬達暫存器證實，**只要不執行 `lerobot-calibrate`**，即使拔插電源（Power Cycle），馬達讀值依然會保持原有的絕對位置。這證明了`lerobot-calibrate`會重置馬達的初始角度。
     - **解決方案**：只要校正過的手臂，都應該妥善保存設定檔，只要這設定檔存在，即使重開機也不會改變馬達回到的角度和外界角度的對照。換句話說，**每個錄制的dataset其實都會和這個校正設定檔綁定，重新校正將會使得之前錄製的dataset無法使用。**


7. **手臂檢查**

   快速確認手臂是否連接 (實時讀取所有馬達)：
   ```bash
   python scripts/watch_motor_position.py --port /dev/ttyACM0
   python scripts/watch_motor_position.py --port /dev/ttyACM1
   ```

8. **相機與連動測試**

   搜尋相機索引：
   ```bash
   lerobot-find-cameras opencv
   ```

   連動測試 (不含相機)：
   ```bash
   lerobot-teleoperate \
     --teleop.type=so101_leader \
     --teleop.port=/dev/ttyACM0 \
     --teleop.id=my_awesome_leader_arm \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM1 \
     --robot.id=my_awesome_follower_arm
   ```

   連動測試 (含相機)：
   ```bash
   lerobot-teleoperate \
     --teleop.type=so101_leader \
     --teleop.port=/dev/ttyACM0 \
     --teleop.id=my_awesome_leader_arm \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM1 \
     --robot.id=my_awesome_follower_arm \
     --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
   ```

## 2. 下一步工作

在環境建置與連動測試成功後，由02-practice-circle-magnet.md記錄接續的項目：
- 錄製數據 ([資料錄製官方指引](https://huggingface.co/docs/lerobot/il_robots))
- 進行模型訓練
- 執行實際推論按壓任務

## 3. 推薦參考資源

這些官方文件與技術文章對於理解及排解 LeRobot 在實體硬體上的運作非常有幫助：

- **HuggingFace 官方範例：**
  - [10_use_so100](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md) (LeRobot 範例)
  - [7_get_started_with_real_robot](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md)
- **優質技術文章與解析：**
  - [LeRobot——Hugging Face打造的机器人开源库：包含对顶层script、与dataset的源码分析(含在简易机械臂SO-ARM100上的部署)](https://blog.csdn.net/v_JULY_v/article/details/139692392)
