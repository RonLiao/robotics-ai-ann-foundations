import os
import sys
import argparse
import time
import torch
import traceback

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

try:
    import numpy as np
    from policies.act_lc.modeling_act import ACTPolicy
except ImportError as e:
    print(f"❌ 無法匯入自定義模型，請確認路徑或套件：{e}")
    sys.exit(1)

try:
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so101_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
except ImportError as e:
    print(f"⚠️ 找不到 LeRobot 機器人控制模組，若僅作離線測試請忽略。詳情: {e}")

def process_image(img_raw, device):
    """將 HWC uint8 影像轉為 CHW float32 RGB Tensor。"""
    if not isinstance(img_raw, torch.Tensor):
        img_raw = torch.from_numpy(img_raw)
    if img_raw.ndim == 3 and img_raw.shape[-1] == 3:
        img_raw = img_raw.permute(2, 0, 1)
    img_raw = img_raw[[2, 1, 0], :, :]  # BGR → RGB
    return img_raw.float().div(255.0).unsqueeze(0).to(device)

def main():
    parser = argparse.ArgumentParser(description="Dual-Camera Language-Conditioned ACT Inference Router")

    parser.add_argument("--repo_id", type=str, default="RonLiao/so101-elevator-act-lc-btn-1-to-3-dualcam")
    parser.add_argument("--robot_id", type=str, default="my_awesome_follower_arm")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--front_camera_index", type=str, default="0")
    parser.add_argument("--wrist_camera_index", type=str, default="2")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--stop_threshold", type=float, default=0.001)
    parser.add_argument("--stop_patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dummy", action="store_true", help="不連接實體手臂，使用隨機資料測試")
    parser.add_argument("--save_frame", type=str, default="outputs/train_frames/inference_dualcam_frame.png")

    args = parser.parse_args()

    print("=====================================================")
    print("🚀 Dual-Camera Language-Conditioned ACT 推理路由系統 啟動")
    print("=====================================================")

    # ==========================
    # 1. 載入模型
    # ==========================
    print(f"\n📥 正在從 {args.repo_id} 載入預訓練模型 ({args.device})...")
    policy = ACTPolicy.from_pretrained(args.repo_id)

    if not hasattr(policy.config, "language_model_name") or policy.config.language_model_name is None:
        policy.config.language_model_name = "distilbert-base-uncased"
        policy.config.max_text_length = 16
        policy.config.language_dim = 768
        print("🔧 Config 修復: 補齊基礎屬性")

        from transformers import AutoTokenizer, AutoModel
        import torch.nn as nn

        policy.tokenizer = AutoTokenizer.from_pretrained(policy.config.language_model_name, clean_up_tokenization_spaces=True)
        print(f"🌍 使用 Tokenizer: {policy.config.language_model_name}")

        if not hasattr(policy.model, "text_encoder"):
            print("🔧 模型組件修復: 注入 text_encoder 與 text_proj...")
            policy.model.text_encoder = AutoModel.from_pretrained(policy.config.language_model_name).to(args.device)
            for param in policy.model.text_encoder.parameters():
                param.requires_grad = False
            policy.model.text_proj = torch.nn.Linear(policy.config.language_dim, policy.model.config.dim_model).to(args.device)
            policy.model.encoder_text_feat_pos_embed = torch.nn.Embedding(policy.config.max_text_length, policy.model.config.dim_model).to(args.device)
            print("✅ 內部組件與位置編碼初始化成功！")

    policy.eval()
    policy.to(args.device)
    print("✅ 模型載入與初始化完畢！")

    # ==========================
    # 1.5. 載入歸一化 Stats
    # ==========================
    global_stats_to_use = None
    if hasattr(policy, "stats") and policy.stats:
        global_stats_to_use = policy.stats
        print(f"📊 偵測到模型自帶歸一化規格 (Stats): {list(global_stats_to_use.keys())}")
    else:
        print("📥 模型缺少 Stats，嘗試從 HuggingFace (lerobot-so101-elevator-6btn-dual-cam) 讀取...")
        import json
        from huggingface_hub import hf_hub_download
        import shutil
        meta_path = os.path.join(root_dir, "configs", "stats_dualcam.json")
        if not os.path.exists(meta_path):
            try:
                dl_path = hf_hub_download(
                    repo_id="RonLiao/lerobot-so101-elevator-6btn-dual-cam",
                    filename="meta/stats.json",
                    repo_type="dataset"
                )
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                shutil.copy(dl_path, meta_path)
            except Exception as e:
                print(f"⚠️ 下載失敗: {e}")
                meta_path = None
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    raw_stats = json.load(f)
                global_stats_to_use = {
                    k: {
                        "mean": torch.tensor(v["mean"] if isinstance(v, dict) else v[0]),
                        "std":  torch.tensor(v["std"]  if isinstance(v, dict) else v[1]),
                    }
                    for k, v in raw_stats.items()
                }
                print("🔧 成功全域載入歸一化 Stats！")
            except Exception as e:
                print(f"⚠️ 讀取本機 Stats 失敗: {e}")

    # ==========================
    # 2. 初始化機械臂與攝影機
    # ==========================
    robot = None
    if not args.dummy:
        print(f"\n🤖 正在初始化與連線至實體手臂 ({args.robot_port})...")
        try:
            robot_cfg = SO101FollowerConfig(id=args.robot_id, port=args.robot_port)

            def _cam_idx(s):
                return int(s) if s.isdigit() else s

            robot_cfg.cameras = {
                "front": OpenCVCameraConfig(
                    index_or_path=_cam_idx(args.front_camera_index),
                    fps=args.camera_fps,
                    width=args.camera_width,
                    height=args.camera_height,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=_cam_idx(args.wrist_camera_index),
                    fps=args.camera_fps,
                    width=args.camera_width,
                    height=args.camera_height,
                ),
            }

            robot = make_robot_from_config(robot_cfg)
            robot.connect()
            print("✅ 手臂與雙相機連線就緒！")
        except Exception as e:
            print(f"❌ 硬體連線失敗，請檢查權限或設備號: {e}")
            sys.exit(1)
    else:
        print("\n⚠️ 進入 Dummy 模式，使用隨機雜訊，不連接實體手臂。")

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

                if args.dummy:
                    observation = {
                        "observation.images.front": torch.randn(1, 3, args.camera_height, args.camera_width).to(args.device),
                        "observation.images.wrist": torch.randn(1, 3, args.camera_height, args.camera_width).to(args.device),
                        "observation.state": torch.randn(1, 6).to(args.device),
                    }
                else:
                    raw_obs = robot.get_observation()
                    if s == 0:
                        print(f"🔍 原始鍵值: {list(raw_obs.keys())}")

                    observation = {}

                    # 1. 處理雙相機影像
                    for cam_key in ("front", "wrist"):
                        if cam_key in raw_obs:
                            observation[f"observation.images.{cam_key}"] = process_image(raw_obs[cam_key], args.device)

                            # 儲存首幀供視角比對
                            if s == 0 and args.save_frame and cam_key == "front":
                                try:
                                    from PIL import Image as PilImage
                                    img_t = observation["observation.images.front"].squeeze(0).clamp(0, 1)
                                    pil_img = PilImage.fromarray((img_t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
                                    os.makedirs(os.path.dirname(os.path.abspath(args.save_frame)), exist_ok=True)
                                    pil_img.save(args.save_frame)
                                    print(f"📸 推論首幀 (front) 已儲存: {args.save_frame}")
                                except Exception as _e:
                                    print(f"⚠️ 儲存首幀失敗: {_e}")

                    # 2. 關節狀態
                    joint_names = [
                        'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                        'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'
                    ]
                    joint_values = [float(raw_obs.get(j, 0.0).item() if isinstance(raw_obs.get(j, 0.0), torch.Tensor) else raw_obs.get(j, 0.0)) for j in joint_names]
                    if len(joint_values) == 6:
                        observation["observation.state"] = torch.tensor([joint_values], dtype=torch.float32).to(args.device)

                    # 3. 歸一化
                    if hasattr(policy, "normalize_inputs"):
                        observation = policy.normalize_inputs(observation)
                    else:
                        if global_stats_to_use:
                            norm_keys = ["observation.images.front", "observation.images.wrist", "observation.state"]
                            for k in norm_keys:
                                if k in observation and k in global_stats_to_use:
                                    stat = global_stats_to_use[k]
                                    if "state" in k:
                                        mean = stat["mean"].to(args.device)
                                        std  = stat["std"].to(args.device)
                                        observation[k] = (observation[k] - mean) / (std + 1e-8)
                        elif s == 0:
                            print("🚨 嚴重警告: 找不到歸一化參數，模型推論極可能失控！")

                    # 4. 語言指令
                    observation["language_instruction"] = [cmd]

                if s == 0:
                    print(f"📊 準備推論，Observation 鍵值: {list(observation.keys())}")
                    for k, v in observation.items():
                        info = v.shape if hasattr(v, "shape") else v
                        print(f"   - {k}: {type(v)}, info={info}")

                with torch.no_grad():
                    action = policy.select_action(observation)

                if not hasattr(policy, "unnormalize_outputs") and global_stats_to_use and "action" in global_stats_to_use:
                    act_mean = global_stats_to_use["action"]["mean"].to(action.device)
                    act_std  = global_stats_to_use["action"]["std"].to(action.device)
                    action = action * act_std + act_mean
                elif hasattr(policy, "unnormalize_outputs"):
                    action = policy.unnormalize_outputs(action)

                current_action = action.squeeze(0).cpu()
                if current_action.dim() > 1:
                    current_action = current_action[0]

                dist = 0.0
                if prev_action is not None:
                    dist = torch.norm(current_action - prev_action).item()
                    static_steps = static_steps + 1 if dist < args.stop_threshold else 0

                if static_steps >= args.stop_patience:
                    print(f"   💡 偵測到動作收斂靜止 (位移<{args.stop_threshold})，判定任務完成。")
                    break

                if args.dummy:
                    if s % 50 == 0:
                        print(f"   ↳ [Step {s}] 位移量: {dist:.6f}")
                else:
                    action_dict = {
                        'shoulder_pan.pos':  current_action[0],
                        'shoulder_lift.pos': current_action[1],
                        'elbow_flex.pos':    current_action[2],
                        'wrist_flex.pos':    current_action[3],
                        'wrist_roll.pos':    current_action[4],
                        'gripper.pos':       current_action[5],
                    }
                    robot.send_action(action_dict)

                prev_action = current_action

            elapsed = time.time() - start_t
            print(f"✅ 任務「{cmd}」執行完畢 (⏱️ 單次決策耗時: {elapsed:.3f}s)，回到待命狀態。")

    except KeyboardInterrupt:
        print("\n👋 收到強制中斷 (Ctrl+C)，提早退出...")
    except Exception:
        print("\n❌ 推理過程中發生未預期異常：")
        traceback.print_exc()
    finally:
        if robot and not args.dummy:
            print("🔌 正在安全關閉硬體資源...")
            robot.disconnect()

if __name__ == "__main__":
    main()
