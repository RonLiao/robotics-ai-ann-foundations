# 命令備忘錄 (Cheatsheet)

記錄 SO-101 機器人與 LeRobot 操作常用指令。

## Docker 環境操作

建立並背景執行 Docker 容器（初次）：
```bash
sudo docker run -d -it --name ron_so101 --device=/dev/ttyACM0 --device=/dev/ttyACM1 --device=/dev/video0 ron_so101
```

重啟 Docker 容器：
```bash
sudo docker start ron_so101
```

進入 Docker 容器並設定環境：
```bash
sudo docker exec -it -u 0 ron_so101 /bin/bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
sudo chmod 666 /dev/video0
conda activate lerobot
cd lerobot/src/lerobot/scripts/
```

## 手臂檢查與校正 (Calibration)

在 Server 48 (Docker) 上的原始校正檔路徑：
- Follower Arm: `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json`
- Leader Arm: `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json`

重新校正手臂：
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_awesome_follower_arm
```

快速確認手臂是否連接 (實時讀取所有馬達)：
  註：請勿再執行lerobot-calibrate，否則會打亂馬達的初始角度
```bash
python scripts/watch_motor_position.py --port /dev/ttyACM0
python scripts/watch_motor_position.py --port /dev/ttyACM1
```

## 相機與連動測試

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

## 資料錄製 (Recording)

### 1. 手動觸發錄製腳本
針對 Docker/Headless 環境下鍵盤監聽失效問題，建立專用腳本 `record_episodes.sh` 以規避終端機貼上長指令之錯誤。腳本依據本地路徑自動判斷啟動全新錄製或續錄。

> [!IMPORTANT]
> **發生 `RevisionNotFoundError` 或 `info.json` 遺失時：**
> 多為遠端存儲庫狀態異常。須先手動刪除 Hugging Face 上的 Dataset Repository 及其本地快取夾後重啟。
> **解決方法**：先至 [Hugging Face 網頁](https://huggingface.co/datasets/RonLiao/lerobot-so101-elevator-dataset/settings) 刪除該 Dataset，並執行 `rm -rf ~/.cache/huggingface/lerobot/RonLiao/lerobot-so101-elevator-dataset`，然後再執行腳本。

**執行方式：**
```bash
# 賦予權限並執行
chmod +x record_episodes.sh
./record_episodes.sh
```

### 2. 如何檢查錄製結果
執行以下指令確認檔案是否成功生成且大小正常：
- **查看 Parquet 數據** (軌跡數據)：`ls -l ~/.cache/huggingface/lerobot/RonLiao/lerobot-so101-elevator-dataset/data`
- **查看影片目錄** (相機影像)：`ls -l ~/.cache/huggingface/lerobot/RonLiao/lerobot-so101-elevator-dataset/videos`

### 3. 單段刪除
用於修正特定錄壞之片段 (如 Episode 4)：
*註：若資料集僅剩最後一段，受工具保護機制限制不可刪除，須直接清理整個目錄。*

```bash
lerobot-edit-dataset \
  --repo_id=RonLiao/lerobot-so101-elevator-dataset \
  --operation.type=delete_episodes \
  --operation.episode_indices="[4]"
```
*註：括號內的 4 替換成想刪除的該段編號。刪除後，再次啟動錄製迴圈，程式會自動從缺少的編號開始續錄。
*註：請將括號內的 4 替換成您想刪除的該段編號。刪除後，再次啟動錄製迴圈，程式會自動從缺少的編號開始續錄。*

### 4. 連續自動錄製
適合想一次錄完，中間不間斷的情境：
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=false \
  --play_sounds=false \
  --dataset.repo_id=RonLiao/lerobot-so101-elevator-dataset \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="Press the circular magnet on the wall" \
  --dataset.push_to_hub=true
```

> [!TIP]
> **若出現 `FileExistsError` 或 `File exists` 報錯：**
> - **想從頭錄製**：執行 `rm -rf ~/.cache/huggingface/lerobot/RonLiao/lerobot-so101-elevator-dataset` 刪除舊資料後重跑。
> - **想續錄**：在指令末尾加上 `--resume=true`。

## WandB 與訓練監控 (Training & WandB)

1. 登入 WandB (只需執行一次，登入資訊會儲存在容器中。除非刪除或重新建立 Container，否則只是 Docker 或 Host 重啟都不需重新登入)：
```bash
wandb login
```

*若需更換 API Key 或重新登入，請執行：*
```bash
wandb login --relogin
```

> [!NOTE]
> API Key 應視為密碼保護，請勿洩露。若不慎洩露，請至 [wandb.ai/settings](https://wandb.ai/settings) 重新產生。

2. 啟動訓練並開啟 WandB 監控：
```bash
lerobot-train \
  --dataset.repo_id=RonLiao/lerobot-so101-elevator-dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_elevator_test \
  --job_name=act_elevator_test \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=5000 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --wandb.enable=true \
  --wandb.project=lerobot-so101-elevator \
  --policy.repo_id=RonLiao/so101-elevator-act
```
