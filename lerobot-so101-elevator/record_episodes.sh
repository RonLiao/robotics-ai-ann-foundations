#!/bin/bash

# record_episodes.sh
# 針對 Docker/Headless 環境下之手動觸發錄製腳本

REPO_ID="RonLiao/lerobot-so101-elevator-dataset"
LOCAL_PATH="$HOME/.cache/huggingface/lerobot/$REPO_ID"

while true; do
    if [ -d "$LOCAL_PATH" ]; then
        RESUME_FLAG="true"
        echo "========= [偵測到既有資料，準備續錄] ========="
    else
        RESUME_FLAG="false"
        echo "========= [準備建立全新資料集] ========="
    fi
    
    echo "按 [Enter] 開始錄製下一個 Episode (10秒)"
    echo "按 [Ctrl + C] 結束並退出此腳本"
    read

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
      --dataset.repo_id=$REPO_ID \
      --dataset.num_episodes=1 \
      --dataset.episode_time_s=10 \
      --dataset.single_task="Press the circular magnet on the wall" \
      --dataset.push_to_hub=false \
      --resume=$RESUME_FLAG

    echo "========= [Episode 錄製結束] ========="
done
