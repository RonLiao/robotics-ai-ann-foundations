# 命令備忘錄 (Cheatsheet)

記錄在 SO-101 機器人與 LeRobot 操作上的常用指令。

## Docker 環境操作

建立並背景執行 Docker 容器（第一次）：
```bash
sudo docker run -d -it --name ron_so101 --device=/dev/ttyACM0 --device=/dev/ttyACM1 --device=/dev/video0 ron_so101
```

重開機後重啟 Docker 容器：
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

尋找相機索引：
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

```
