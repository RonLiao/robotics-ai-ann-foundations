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

重新校正手臂：
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_awesome_follower_arm
```

快速確認手臂是否連接 (測試用 ID)：
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=test_leader
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=test_follower
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

連動測試 (結合前置相機，準備數據錄製)：
```bash
lerobot-teleoperate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 160, height: 120, fps: 20}}"
```
