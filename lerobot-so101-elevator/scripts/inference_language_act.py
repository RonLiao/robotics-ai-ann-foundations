import os
import sys
import argparse
import time
import torch
import traceback

# 將 root directory 加入 sys.path 確保能 import policies
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

try:
    import numpy as np
    from policies.act_lc.modeling_act import ACTPolicy
except ImportError as e:
    print(f"❌ 無法匯入自定義模型，請確認路徑或套件：{e}")
    sys.exit(1)

# 依賴於 lerobot 的機器人控制庫
try:
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so101_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
except ImportError as e:
    print(f"⚠️ 找不到 LeRobot 機器人控制模組，若僅作離線測試請忽略。詳情: {e}")

def main():
    parser = argparse.ArgumentParser(description="Language-Conditioned ACT Inference Router")
    
    # 硬體環境與模型參數預設值
    parser.add_argument("--repo_id", type=str, default="RonLiao/so101-elevator-act-lc-btn-1-to-3", help="Hugging Face Model Repo")
    parser.add_argument("--robot_type", type=str, default="so101_follower", help="Robot identifier for LeRobot")
    parser.add_argument("--robot_id", type=str, default="my_awesome_follower_arm", help="您在錄製資料時賦予手臂的 ID")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM1", help="Serial port for the follower arm")
    # 注意: 模型對影像預期鍵值為 'observation.images.front' (根據先前訓練紀錄)
    parser.add_argument("--camera_key", type=str, default="front", help="Camera identifier in config")
    parser.add_argument("--camera_index", type=str, default="0", help="Camera index or path (字串型態，因為可能是 /dev/video0)")
    parser.add_argument("--camera_width", type=int, default=640, help="Camera width")
    parser.add_argument("--camera_height", type=int, default=480, help="Camera height")
    parser.add_argument("--camera_fps", type=int, default=30, help="Camera fps")
    parser.add_argument("--num_steps", type=int, default=200, help="下達指令後持續執行的最大步數 (Fallback)")
    parser.add_argument("--stop_threshold", type=float, default=0.001, help="判定為靜止的位移閾值 (越小越嚴格)")
    parser.add_argument("--stop_patience", type=int, default=15, help="連續靜止幾步後判定為任務完成")
    
    # 測試模式與推論裝置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理硬體裝置")
    parser.add_argument("--dummy", action="store_true", help="使用模擬資料進行測試，不連接真實手臂")
    
    args = parser.parse_args()

    print("=====================================================")
    print("🚀 Language-Conditioned ACT 推理路由系統 啟動")
    print("=====================================================")

    # ==========================
    # 1. 載入我們改造後的模型權重
    # ==========================
    print(f"\n📥 正在從 {args.repo_id} 載入預訓練模型 ({args.device})...")
    policy = ACTPolicy.from_pretrained(args.repo_id)
    policy.eval()
    policy.to(args.device)
    print("✅ 模型載入與初始化完畢！")

    # ==========================
    # 2. 初始化機械臂與攝影機連線
    # ==========================
    robot = None
    if not args.dummy:
        print(f"\n🤖 正在初始化與連線至實體手臂 ({args.robot_port})...")
        
        try:
            # 1. 建立機器人特定配置 (務必指定 id 以讀取正確的校正檔)
            robot_cfg = SO101FollowerConfig(
                id=args.robot_id,
                port=args.robot_port
            )
            
            # 2. 設定攝影機參數 (使用 OpenCVCameraConfig)
            # 參數順序: index_or_path, fps, width, height
            robot_cfg.cameras = {
                args.camera_key: OpenCVCameraConfig(
                    index_or_path=args.camera_index if not args.camera_index.isdigit() else int(args.camera_index),
                    fps=args.camera_fps,
                    width=args.camera_width,
                    height=args.camera_height,
                )
            }
            
            # 3. 建立機器人物件並連線
            robot = make_robot_from_config(robot_cfg)
            robot.connect()
            print("✅ 手臂與攝影機連線就緒！")
        except Exception as e:
            print(f"❌ 硬體連線失敗，請檢查權限或設備號: {e}")
            sys.exit(1)
    else:
        print("\n⚠️ 進入 Dummy (預演) 模式，推論引擎將使用隨機雜訊，不連接真實手臂實體。")

    # ==========================
    # 3. 互動式指令接收引擎
    # ==========================
    print("\n------------------------------")
    print("💬 推論引擎怠速中，等待指令...")
    
    try:
        while True:
            print("\n💡 可用指令參考：'press button 1', 'press button 2', 'press button 3'")
            cmd = input("⌨️  請輸入按壓目標指令 (或輸入 'q' 退出): ").strip()
            
            if cmd.lower() in ["q", "quit", "exit"]:
                print("👋 關閉推理路由引擎...")
                break
            if not cmd:
                continue

            print(f"🎯 鎖定特徵注入條件: [ {cmd} ] - 開始執行任務...")
            
            prev_action = None
            static_steps = 0

            for s in range(args.num_steps):
                start_t = time.time()
                # 抓取並建構神經網路輸入批次
                if args.dummy:
                    observation = {
                        f"observation.images.{args.camera_key}": torch.randn(1, 3, args.camera_height, args.camera_width).to(args.device),
                        "observation.state": torch.randn(1, 6).to(args.device),
                    }
                else:
                    raw_obs = robot.get_observation()
                    # 只有第 0 步印一次，幫助除錯
                    if s == 0:
                        print(f"🔍 原始鍵值: {list(raw_obs.keys())}")
                    
                    observation = {}
                    joint_values = []
                    
                    # 1. 處理影像 (強制轉換為 float32 並歸一化)
                    if "front" in raw_obs:
                        img = raw_obs["front"]
                        if not isinstance(img, torch.Tensor):
                            img = torch.from_numpy(img)
                        if img.ndim == 3 and img.shape[-1] == 3:
                            img = img.permute(2, 0, 1)
                        img = img.float() / 255.0
                        observation["observation.images.front"] = img.unsqueeze(0).to(args.device)
                    
                    # 2. 整合 6 維狀態 (依序尋找關節)
                    joint_names = [
                        'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                        'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'
                    ]
                    joint_values = []
                    for j_name in joint_names:
                        val = raw_obs.get(j_name, 0.0)
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        joint_values.append(float(val))
                    
                    if len(joint_values) == 6:
                        observation["observation.state"] = torch.tensor([joint_values], dtype=torch.float32).to(args.device)
                    
                    # 3. 補上指令 (文字)
                    observation["language_instruction"] = [cmd] 

                # 【📍 偵錯輸出】
                if s == 0:
                    print(f"📊 準備推論，Observation 鍵值: {list(observation.keys())}")
                    for k, v in observation.items():
                        kind = type(v)
                        info = v.shape if hasattr(v, "shape") else v
                        print(f"   - {k}: {kind}, info={info}")

                # 模型推論 (不寫入梯度)
                with torch.no_grad():
                    action = policy.select_action(observation)
                
                # 自動適應 2D/3D 並取最新一步動作
                current_action = action.squeeze(0).cpu() if action.dim() > 2 else action.cpu()
                if current_action.dim() > 1: current_action = current_action[0] # 取第一步

                # --- 靜止偵測自動停止邏輯 ---
                dist = 0.0
                if prev_action is not None:
                    # 計算歐幾里得距離（位移量）
                    dist = torch.norm(current_action - prev_action).item()
                    if dist < args.stop_threshold:
                        static_steps += 1
                    else:
                        static_steps = 0 # 只要有動，就重新計數
                
                if static_steps >= args.stop_patience:
                    print(f"   💡 偵測到模型動作已收斂並靜止 (位移<{args.stop_threshold})，判定任務完成。")
                    break

                # --- 動作發送 ---
                if args.dummy:
                    if s % 50 == 0:
                        print(f"   ↳ [Step {s}] 位移量: {dist:.6f}")
                else:
                    # 【📍 關鍵修正】：將 Tensor 轉回 Robot 預期的字典格式
                    action_dict = {
                        'shoulder_pan.pos': current_action[0],
                        'shoulder_lift.pos': current_action[1],
                        'elbow_flex.pos': current_action[2],
                        'wrist_flex.pos': current_action[3],
                        'wrist_roll.pos': current_action[4],
                        'gripper.pos': current_action[5]
                    }
                    robot.send_action(action_dict)
                
                prev_action = current_action
                    
                # 對齊 FPS
                # time.sleep(0.01)

            print(f"✅ 任務「{cmd}」執行完畢，回到待命狀態。")

    except KeyboardInterrupt:
        print("\n👋 收到強制中斷 (Ctrl+C)，提早退出推理引擎...")
    except Exception:
        print("\n❌ 推理過程中發生未預期異常：")
        traceback.print_exc()
    finally:
        if robot and not args.dummy:
            print("🔌 正在安全關閉硬體資源...")
            robot.disconnect()

if __name__ == "__main__":
    main()
