#!/bin/bash

# 使用方式: bash scripts/record_6btn_dual_cam.sh [按鈕編號 1-6] [錄製集數] [是否續傳 true/false]
# 範例: bash scripts/record_6btn_dual_cam.sh 3 5 true

BTN_IDX=${1:-1}
NUM_EPISODES=${2:-1}
RESUME_FLAG=${3:-true}

REPO_ID="RonLiao/lerobot-so101-elevator-6btn-dual-cam"
TASK_STRING="press button $BTN_IDX"

echo "----------------------------------------------------"
echo "正在啟動雙相機多任務錄製程序 (5秒自動倒數模式)..."
echo "目標按鈕: $BTN_IDX"
echo "指令內容: '$TASK_STRING'"
echo "錄製總集數: $NUM_EPISODES"
echo "資料集 ID: $REPO_ID"
echo "續傳模式: $RESUME_FLAG"
echo "相機: front (index 0) + wrist (index 2)"
echo "----------------------------------------------------"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --display_data=false \
  --play_sounds=false \
  --dataset.repo_id=$REPO_ID \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="$TASK_STRING" \
  --dataset.push_to_hub=false \
  --resume=$RESUME_FLAG
