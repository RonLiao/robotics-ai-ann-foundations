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
    parser.add_argument("--repo_id", type=str, default="RonLiao/so101-elevator-act-lc-btn-1-to-3-v2", help="Hugging Face Model Repo or local checkpoint path")
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
    parser.add_argument("--save_frame", type=str, default="outputs/train_frames/inference_frame.png",
                        help="儲存推論首幀影像的路徑（用於和訓練集畫面對比），設為空字串可停用")
    
    args = parser.parse_args()

    print("=====================================================")
    print("🚀 Language-Conditioned ACT 推理路由系統 啟動")
    print("=====================================================")

    # ==========================
    # 1. 載入我們改造後的模型權重
    # ==========================
    print(f"\n📥 正在從 {args.repo_id} 載入預訓練模型 ({args.device})...")
    policy = ACTPolicy.from_pretrained(args.repo_id)
    
    # 【📍 終極修正】：補齊所有缺失的語言模型屬性與內部組件 (Deep Model Hotfix)
    if not hasattr(policy.config, "language_model_name") or policy.config.language_model_name is None:
        # Must match configuration_act.py: language_model_name = "distilbert-base-uncased"
        policy.config.language_model_name = "distilbert-base-uncased"
        policy.config.max_text_length = 16  # matches configuration_act.py default
        policy.config.language_dim = 768
        print("🔧 Config 修復: 補齊基礎屬性")
        
        from transformers import AutoTokenizer, AutoModel
        import torch.nn as nn
        
        # 1. 修復 Policy 層 (外層)
        policy.tokenizer = AutoTokenizer.from_pretrained(policy.config.language_model_name, clean_up_tokenization_spaces=True)
        print(f"🌍 使用 Tokenizer: {policy.config.language_model_name}")
        
        # 2. 修復 Model 層 (內層)
        if not hasattr(policy.model, "text_encoder"):
            print("🔧 模型組件修復: 注入 text_encoder 與 text_proj...")
            policy.model.text_encoder = AutoModel.from_pretrained(policy.config.language_model_name).to(args.device)
            # 凍結 text_encoder 參數 (與訓練時一致)
            for param in policy.model.text_encoder.parameters():
                param.requires_grad = False
            
            # 根據 BERT (768) 與模型維度 (通常是 512) 建立投影層
            policy.model.text_proj = nn.Linear(policy.config.language_dim, policy.model.config.dim_model).to(args.device)
            
            # 【📍 關鍵補齊】：文字位置編碼 (Positional Embedding for Text)
            policy.model.encoder_text_feat_pos_embed = nn.Embedding(policy.config.max_text_length, policy.model.config.dim_model).to(args.device)
            print("✅ 內部組件與位置編碼初始化成功！")
    
    policy.eval()
    policy.to(args.device)
    print("✅ 模型載入與初始化完畢！")

    # ==========================
    # 1.5. 【📍 全域歸一化參數預載 (Stats)】
    # ==========================
    global_stats_to_use = None
    if hasattr(policy, "stats") and policy.stats:
        global_stats_to_use = policy.stats
        print(f"📊 偵測到模型自帶歸一化規格 (Stats): {list(global_stats_to_use.keys())}")
        if "observation.state" in global_stats_to_use:
            s_mean = global_stats_to_use["observation.state"]["mean"]
            print(f"   - State Mean (前三維): {s_mean[:3]}")
    else:
        print("📥 模型缺少 Stats，嘗試從 Hugging Face Hub (RonLiao/lerobot-so101-elevator-6btn-multitask) 讀取...")
        import json
        from huggingface_hub import hf_hub_download
        import shutil
        meta_path = os.path.join(root_dir, "configs", "stats.json")
        if not os.path.exists(meta_path):
            try:
                dl_path = hf_hub_download(repo_id="RonLiao/lerobot-so101-elevator-6btn-multitask", filename="meta/stats.json", repo_type="dataset")
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                shutil.copy(dl_path, meta_path)
            except Exception as e:
                print(f"⚠️ 下載失敗: {e}")
                meta_path = None
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    raw_stats = json.load(f)
                    global_stats_to_use = {}
                    for k, v in raw_stats.items():
                        global_stats_to_use[k] = {
                            "mean": torch.tensor(v["mean"] if isinstance(v, dict) else v[0]),
                            "std": torch.tensor(v["std"] if isinstance(v, dict) else v[1]),
                        }
                print("🔧 成功全域載入歸一化 Stats！")
            except Exception as e:
                print(f"⚠️ 讀取本機 Stats 失敗: {e}")

    if hasattr(policy.config, "language_model_name"):
        print(f"🌍 語言模型名稱: {policy.config.language_model_name}")
    else:
        print("⚠️  警告: 此 Policy Config 無 language_model_name，指令將無效。")

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
                        
                        # 【📍 關鍵修正】：與訓練保持一致，使用 RGB 色域
                        img = img[[2, 1, 0], :, :]
                        img = img.float() / 255.0
                        observation["observation.images.front"] = img.unsqueeze(0).to(args.device)

                        # -- Save first frame for visual comparison --
                        if s == 0 and args.save_frame:
                            try:
                                from PIL import Image as PilImage
                                save_img = img.clamp(0.0, 1.0)
                                pil_img = PilImage.fromarray(
                                    (save_img.permute(1, 2, 0).numpy() * 255).astype("uint8")
                                )
                                os.makedirs(os.path.dirname(os.path.abspath(args.save_frame)), exist_ok=True)
                                pil_img.save(args.save_frame)
                                print(f"📸 推論首幀已儲存: {args.save_frame}")
                                print(f"   → 請與 outputs/train_frames/front/grid_press_button_*.png 並排比較")
                            except Exception as _e:
                                print(f"⚠️ 儲存首幀失敗: {_e}")
                    
                    # 2. 整合 6 維狀態
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
                    
                    # 3. 【📍 核心修復】：終極歸一化 (Hardcore Normalization)
                    # 如果 policy 沒有 stats，我們必須從資料集硬讀，否則手臂會亂飛
                    if hasattr(policy, "normalize_inputs"):
                        observation = policy.normalize_inputs(observation)
                    else:
                        if global_stats_to_use:
                            for k in ["observation.images.front", "observation.state"]:
                                if k in observation and k in global_stats_to_use:
                                    stat = global_stats_to_use[k]
                                    if "state" in k:
                                        mean = stat["mean"].to(args.device)
                                        std = stat["std"].to(args.device)
                                        observation[k] = (observation[k] - mean) / (std + 1e-8)
                        elif s == 0:
                            print("🚨 嚴重警告: 找不到歸一化參數，模型推論極可能會完全失控！")
                    
                    # 4. 補上指令 (文字)
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

                # 【📍 補齊遺漏的反正規化 (Unnormalization)】
                # 如果我們使用 global_stats_to_use 且模型內部缺少反正規化，必須手動反推物理量！
                if not hasattr(policy, "unnormalize_outputs") and global_stats_to_use and "action" in global_stats_to_use:
                    act_mean = global_stats_to_use["action"]["mean"].to(action.device)
                    act_std = global_stats_to_use["action"]["std"].to(action.device)
                    action = action * act_std + act_mean
                elif hasattr(policy, "unnormalize_outputs"):
                    action = policy.unnormalize_outputs(action)
                
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
