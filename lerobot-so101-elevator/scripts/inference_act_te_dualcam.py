"""inference_act_te_dualcam.py — ACT Task Embedding 雙相機推論腳本

與 inference_language_act_dualcam.py 的差異：
  - 不需要 DistilBERT / Tokenizer
  - 改以 --task 1/2/3 或 --instruction "press button 1" 指定任務
  - task_index (0-based int) 直接送入模型，無文字處理
"""

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
    from policies.act_te.modeling_act import ACTPolicy
    from policies.act_te.configuration_act import ACTConfig as TEACTConfig
except ImportError as e:
    print(f"❌ 無法匯入 act_te 模型：{e}")
    sys.exit(1)

# draccus 的 _choice_registry 必須在 from_pretrained 呼叫前更新，
# 否則 draccus 解析 config.json 時會用 vanilla ACTConfig（不含 num_tasks）。
try:
    from lerobot.configs.policies import PreTrainedConfig
    for _attr in ['_registry', '__registry__', '_subclass_registry', '_choice_registry']:
        _reg = getattr(PreTrainedConfig, _attr, None)
        if isinstance(_reg, dict) and 'act' in _reg:
            _reg['act'] = TEACTConfig
            break
except Exception as _e:
    print(f"⚠️ 替換 PreTrainedConfig registry 失敗：{_e}")

try:
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.so101_follower import SO101FollowerConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
except ImportError as e:
    print(f"⚠️ 找不到 LeRobot 機器人控制模組，若僅作離線測試請忽略。詳情: {e}")

# 指令 → task_index 對照表（0-based，與訓練時 meta/tasks 一致）
INSTRUCTION_MAP = {
    "press button 1": 0,
    "press button 2": 1,
    "press button 3": 2,
    "press button 4": 3,
    "press button 5": 4,
    "press button 6": 5,
    "button 1": 0,
    "button 2": 1,
    "button 3": 2,
    "1": 0,
    "2": 1,
    "3": 2,
}


def parse_task_index(task_str: str, task_int: int | None) -> int:
    """從字串指令或整數解析 task_index（0-based）。"""
    if task_int is not None:
        return task_int - 1  # CLI 用 1-based，模型用 0-based
    key = task_str.strip().lower()
    if key in INSTRUCTION_MAP:
        return INSTRUCTION_MAP[key]
    raise ValueError(
        f"無法解析指令 '{task_str}'。請使用 --task 1/2/3 或 --instruction 'press button 1'。"
    )


def process_image(img_raw, device):
    if not isinstance(img_raw, torch.Tensor):
        img_raw = torch.from_numpy(img_raw)
    if img_raw.ndim == 3 and img_raw.shape[-1] == 3:
        img_raw = img_raw.permute(2, 0, 1)
    img_raw = img_raw[[2, 1, 0], :, :]  # BGR → RGB
    return img_raw.float().div(255.0).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description="ACT Task Embedding 雙相機推論")

    # 模型與任務
    parser.add_argument("--repo_id", type=str,
                        default="RonLiao/so101-elevator-act-te-btn-1-to-3-dualcam")
    parser.add_argument("--task", type=int, default=None,
                        help="按鈕編號（1-based，例如 --task 1 表示按第一顆按鈕）")
    parser.add_argument("--instruction", type=str, default="press button 1",
                        help="文字指令（自動轉換為 task_index）")

    # 機器人與相機
    parser.add_argument("--robot_id", type=str, default="my_awesome_follower_arm")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--front_camera_index", type=str, default="0")
    parser.add_argument("--wrist_camera_index", type=str, default="2")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=30)

    # 推論參數
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--stop_threshold", type=float, default=0.001)
    parser.add_argument("--stop_patience", type=int, default=15)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="每次 chunk 執行步數後重新規劃（預設用模型 config，例如 --n_action_steps 20）")
    parser.add_argument("--dummy", action="store_true",
                        help="不連接實體手臂，使用隨機資料測試語言條件區辨")
    parser.add_argument("--save_frame", type=str,
                        default="outputs/train_frames/inference_te_frame.png")

    args = parser.parse_args()

    # 解析 task_index
    task_index = parse_task_index(args.instruction, args.task)
    task_display = args.task if args.task else args.instruction
    print("=====================================================")
    print("🚀 ACT Task Embedding 推論系統啟動")
    print(f"   任務：{task_display}  →  task_index = {task_index} (0-based)")
    print("=====================================================")

    # ── 1. 載入模型 ──────────────────────────────────────────────────────────
    ckpt_path = args.repo_id
    # HuggingFace repo ID 格式為 "namespace/repo_name"（恰好一個 /）
    # 本地路徑（相對或絕對）含有多個 / 或以 / 開頭
    is_hf_repo = not os.path.isabs(ckpt_path) and ckpt_path.count('/') == 1
    if not is_hf_repo and not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(root_dir, ckpt_path)
    print(f"\n📥 載入模型：{ckpt_path}")
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(args.device)
    policy.eval()
    print(f"   num_tasks       = {policy.config.num_tasks}")
    print(f"   n_action_steps  = {policy.config.n_action_steps} (model config)")
    if args.n_action_steps is not None:
        policy.config.n_action_steps = args.n_action_steps
        print(f"   n_action_steps  → {args.n_action_steps} (overridden)")

    # ── 2. 載入歸一化統計 ─────────────────────────────────────────────────────
    import json
    import shutil
    from huggingface_hub import hf_hub_download

    global_stats = None
    meta_path = os.path.join(root_dir, "configs", "stats_dualcam.json")
    if not os.path.exists(meta_path):
        print("📥 本地 stats 不存在，從 HuggingFace 下載...")
        try:
            dl_path = hf_hub_download(
                repo_id="RonLiao/lerobot-so101-elevator-6btn-dual-cam",
                filename="meta/stats.json",
                repo_type="dataset"
            )
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            shutil.copy(dl_path, meta_path)
        except Exception as _e:
            print(f"⚠️ 下載失敗：{_e}")
            meta_path = None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            raw_stats = json.load(f)
        global_stats = {
            k: {
                "mean": torch.tensor(v["mean"] if isinstance(v, dict) else v[0]),
                "std":  torch.tensor(v["std"]  if isinstance(v, dict) else v[1]),
            }
            for k, v in raw_stats.items()
        }
        print(f"✅ 歸一化統計已載入：{meta_path}")
        if "observation.state" in global_stats:
            s = global_stats["observation.state"]
            print(f"   state mean: {s['mean'].tolist()}")
            print(f"   state std:  {s['std'].tolist()}")
    else:
        print("🚨 嚴重警告：找不到歸一化統計，模型推論極可能失控！")

    # ── 3. Dummy 測試模式 ─────────────────────────────────────────────────────
    if args.dummy:
        print("\n🔧 Dummy 模式：使用隨機輸入測試 Task Embedding 區辨能力")
        from lerobot.utils.constants import OBS_STATE, OBS_IMAGES

        all_displacements = {}
        for tidx in range(policy.config.num_tasks):
            dummy_state = torch.zeros(1, policy.config.robot_state_feature.shape[0]).to(args.device)

            # 按照 config.image_features 的 key 放入各相機影像
            batch = {OBS_STATE: dummy_state,
                     "task_index": torch.tensor([tidx], dtype=torch.long).to(args.device)}
            for img_key in policy.config.image_features:
                batch[img_key] = torch.rand(1, 3, args.camera_height, args.camera_width).to(args.device)
            policy.reset()
            prev_action = None
            total_displacement = 0.0
            for step in range(150):
                action = policy.select_action(batch)
                action_np = action.cpu().numpy().squeeze()
                if prev_action is not None:
                    disp = float(np.abs(action_np - prev_action).sum())
                    total_displacement += disp
                prev_action = action_np
            all_displacements[f"task_{tidx}"] = total_displacement
            print(f"   task_{tidx} (button {tidx+1}) 總位移量 = {total_displacement:.2f}")

        print("\n   區辨比較（數值差異越大代表 Task Embedding 越有效）：")
        vals = list(all_displacements.values())
        for k, v in all_displacements.items():
            print(f"   {k}: {v:.2f}")
        print(f"   最大/最小比 = {max(vals)/max(min(vals), 1e-6):.2f}×")
        return

    # ── 4. 實機推論 ───────────────────────────────────────────────────────────
    print(f"\n🤖 連接機器人：port={args.robot_port}")
    def _cam_idx(s):
        return int(s) if s.isdigit() else s

    try:
        robot_config = SO101FollowerConfig(
            id=args.robot_id,
            port=args.robot_port,
            cameras={
                "front": OpenCVCameraConfig(
                    index_or_path=_cam_idx(args.front_camera_index),
                    width=args.camera_width,
                    height=args.camera_height,
                    fps=args.camera_fps,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=_cam_idx(args.wrist_camera_index),
                    width=args.camera_width,
                    height=args.camera_height,
                    fps=args.camera_fps,
                ),
            }
        )
        robot = make_robot_from_config(robot_config)
        robot.connect()
        print("✅ 機器人連接成功")
    except Exception as e:
        print(f"❌ 機器人連接失敗：{e}")
        traceback.print_exc()
        return

    from lerobot.utils.constants import OBS_STATE
    policy.reset()

    JOINT_NAMES = [
        'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
        'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos',
    ]

    prev_action = None
    still_count = 0
    save_done = False

    print(f"\n▶  開始推論（{args.num_steps} steps）...")
    try:
        for step in range(args.num_steps):
            raw_obs = robot.get_observation()
            if step == 0:
                print(f"   原始 obs keys: {list(raw_obs.keys())}")

            # ── 影像 ──────────────────────────────────────────────────────
            observation = {}
            for cam_key in ("front", "wrist"):
                if cam_key in raw_obs:
                    observation[f"observation.images.{cam_key}"] = process_image(raw_obs[cam_key], args.device)

            # ── 關節狀態 ──────────────────────────────────────────────────
            joint_vals = [
                float(raw_obs[j].item() if isinstance(raw_obs.get(j), torch.Tensor) else raw_obs.get(j, 0.0))
                for j in JOINT_NAMES
            ]
            state_np = np.array(joint_vals)
            observation[OBS_STATE] = torch.from_numpy(state_np).float().unsqueeze(0).to(args.device)
            if global_stats and "observation.state" in global_stats:
                mean = global_stats["observation.state"]["mean"].to(args.device)
                std  = global_stats["observation.state"]["std"].to(args.device)
                observation[OBS_STATE] = (observation[OBS_STATE] - mean) / (std + 1e-8)

            # ── task_index ────────────────────────────────────────────────
            observation["task_index"] = torch.tensor([task_index], dtype=torch.long).to(args.device)

            # ── 儲存首幀 ──────────────────────────────────────────────────
            if not save_done and args.save_frame and "observation.images.front" in observation:
                try:
                    from PIL import Image as PilImage
                    img_t = observation["observation.images.front"].squeeze(0).clamp(0, 1)
                    pil_img = PilImage.fromarray((img_t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
                    os.makedirs(os.path.dirname(os.path.abspath(args.save_frame)), exist_ok=True)
                    pil_img.save(args.save_frame)
                    print(f"📷 推論首幀已儲存：{args.save_frame}")
                except Exception as _e:
                    print(f"⚠️ 儲存首幀失敗：{_e}")
                save_done = True

            action = policy.select_action(observation)
            if global_stats and "action" in global_stats:
                act_mean = global_stats["action"]["mean"].to(action.device)
                act_std  = global_stats["action"]["std"].to(action.device)
                action = action * act_std + act_mean
            action_np = action.cpu().numpy().squeeze()

            # 自動靜止停止
            if prev_action is not None:
                disp = float(np.abs(action_np - prev_action).sum())
                if disp < args.stop_threshold:
                    still_count += 1
                    if still_count >= args.stop_patience:
                        print(f"\n⏹  Step {step}: 連續 {args.stop_patience} 步靜止，自動停止。")
                        break
                else:
                    still_count = 0
            prev_action = action_np.copy()

            action_dict = {
                'shoulder_pan.pos':  action_np[0],
                'shoulder_lift.pos': action_np[1],
                'elbow_flex.pos':    action_np[2],
                'wrist_flex.pos':    action_np[3],
                'wrist_roll.pos':    action_np[4],
                'gripper.pos':       action_np[5],
            }
            robot.send_action(action_dict)
            print(f"  Step {step:4d}  action={np.round(action_np[:4], 3)}", end="\r")

    except KeyboardInterrupt:
        print("\n⏹  使用者中斷推論。")
    finally:
        robot.disconnect()
        print("\n✅ 機器人已斷線。")


if __name__ == "__main__":
    main()
