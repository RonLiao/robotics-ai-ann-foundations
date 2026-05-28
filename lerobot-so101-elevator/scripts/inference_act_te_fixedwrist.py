"""inference_act_te_fixedwrist.py — ACT Task Embedding 雙相機推論腳本（fixedwrist 版）

與 inference_act_te_dualcam.py 的差異：
  - Stats 預設來源：RonLiao/lerobot-so101-elevator-6btn-dual-cam-fixedwrist
  - Stats 本地快取：configs/stats_fixedwrist.json（不覆蓋舊的 stats_dualcam.json）
  - 預設 repo_id：RonLiao/so101-elevator-act-te-btn-1-to-2-fixedwrist-v1
  - 新增 --stats_dataset 參數，可彈性指定 HuggingFace 資料集 repo 以自動下載 stats
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

# draccus registry 必須在 from_pretrained 前更新
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
    """BGR uint8 → RGB float [0, 1]。

    注意：LeRobotDataset 訓練時不做額外影像正規化，直接使用 [0, 1] 像素值。
    雖然訓練 config 顯示 use_imagenet_stats=True，但該設定僅影響 policy.config
    中儲存的 stats metadata，實際訓練時圖像並未套用 ImageNet 正規化（已確認）。
    """
    if not isinstance(img_raw, torch.Tensor):
        img_raw = torch.from_numpy(img_raw)
    if img_raw.ndim == 3 and img_raw.shape[-1] == 3:
        img_raw = img_raw.permute(2, 0, 1)
    img_raw = img_raw[[2, 1, 0], :, :]  # BGR → RGB
    return img_raw.float().div(255.0).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description="ACT Task Embedding 雙相機推論（fixedwrist 版）")

    # 模型與任務
    parser.add_argument("--repo_id", type=str,
                        default="RonLiao/so101-elevator-act-te-btn-1-to-2-fixedwrist-v1")
    parser.add_argument("--task", type=int, default=None,
                        help="按鈕編號（1-based，例如 --task 1 表示按第一顆按鈕）")
    parser.add_argument("--instruction", type=str, default="press button 1",
                        help="文字指令（自動轉換為 task_index）")

    # Stats 來源設定
    parser.add_argument("--stats_dataset", type=str,
                        default="RonLiao/lerobot-so101-elevator-6btn-dual-cam-fixedwrist",
                        help="用於下載 stats 的 HuggingFace 資料集 repo（若本地快取不存在時自動下載）")
    parser.add_argument("--stats_cache", type=str,
                        default="configs/stats_fixedwrist.json",
                        help="Stats 本地快取路徑（預設：configs/stats_fixedwrist.json）")

    # 機器人與相機
    parser.add_argument("--robot_id", type=str, default="my_awesome_follower_arm")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--front_camera_index", type=str, default="0")
    parser.add_argument("--wrist_camera_index", type=str, default="2")
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=30)

    # 推論參數
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--stop_threshold", type=float, default=0.001)
    parser.add_argument("--stop_patience", type=int, default=15)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_action_steps", type=int, default=20,
                        help="Action Queue 模式：每幾步重新預測一次（預設 20；僅在 --no_te 時有效）")
    parser.add_argument("--no_te", action="store_true",
                        help="停用 Temporal Ensembling，改用 Action Queue（診斷用；預設開啟 TE）")
    parser.add_argument("--te_coeff", type=float, default=0.01,
                        help="Temporal Ensembling 指數權重係數（預設 0.01）")
    parser.add_argument("--dummy", action="store_true",
                        help="不連接實體手臂，使用隨機資料測試 Task Embedding 區辨")
    parser.add_argument("--diag_only", action="store_true",
                        help="離線 Chunk 診斷：不連機器人，直接用 --init_state 指定關節角度（預設取 dataset 初始位置）跑 predict_action_chunk")
    parser.add_argument("--init_state", type=float, nargs=6,
                        default=None,
                        metavar=("SP", "SL", "EF", "WF", "WR", "GR"),
                        help="--diag_only 用：6 個關節初始角度（degrees）。預設：dataset 典型初始位置")
    parser.add_argument("--save_frame", type=str,
                        default="outputs/train_frames/inference_te_fixedwrist_frame.png")

    args = parser.parse_args()

    # 解析 task_index
    task_index = parse_task_index(args.instruction, args.task)
    task_display = args.task if args.task else args.instruction
    print("=====================================================")
    print("🚀 ACT Task Embedding 推論系統啟動（fixedwrist 版）")
    print(f"   任務：{task_display}  →  task_index = {task_index} (0-based)")
    print("=====================================================")

    # ── 1. 載入模型 ──────────────────────────────────────────────────────────
    ckpt_path = args.repo_id
    is_hf_repo = not os.path.isabs(ckpt_path) and ckpt_path.count('/') == 1
    if not is_hf_repo and not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(root_dir, ckpt_path)
    print(f"\n📥 載入模型：{ckpt_path}")
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(args.device)
    policy.eval()
    print(f"   num_tasks       = {policy.config.num_tasks}")
    print(f"   n_action_steps  = {policy.config.n_action_steps} (model config)")

    # Temporal Ensembling / Action Queue 模式選擇
    if args.no_te:
        # Action Queue 模式：不使用 TE，每 n_action_steps 步重新預測
        # 優點：不會因 action[0]≈current_state 而死鎖；手臂確實會依 chunk 執行動作
        policy.config.temporal_ensemble_coeff = None
        policy.config.n_action_steps = args.n_action_steps
        policy._action_queue = __import__('collections').deque([], maxlen=policy.config.n_action_steps)
        print(f"   [模式] Action Queue (--no_te)：每 {args.n_action_steps} 步重新預測")
    else:
        # Temporal Ensembling 模式（預設）
        policy.config.n_action_steps = policy.config.chunk_size  # TE 模式下 queue 不介入
        policy.config.temporal_ensemble_coeff = args.te_coeff
        from policies.act_te.modeling_act import ACTTemporalEnsembler
        policy.temporal_ensembler = ACTTemporalEnsembler(args.te_coeff, policy.config.chunk_size)
        print(f"   temporal_ensemble_coeff = {args.te_coeff} (Enabled for closed-loop control)")

    # ── 2. 載入歸一化統計 ─────────────────────────────────────────────────────
    import json
    import shutil
    from huggingface_hub import hf_hub_download

    global_stats = None
    # Resolve stats cache path
    meta_path = args.stats_cache
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(root_dir, meta_path)

    if not os.path.exists(meta_path):
        print(f"📥 本地 stats 不存在（{meta_path}），從 HuggingFace 下載...")
        print(f"   資料集：{args.stats_dataset}")
        try:
            dl_path = hf_hub_download(
                repo_id=args.stats_dataset,
                filename="meta/stats.json",
                repo_type="dataset"
            )
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            shutil.copy(dl_path, meta_path)
            print(f"✅ Stats 已下載並快取至：{meta_path}")
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

    # ── 3.5 離線 Chunk 診斷（--diag_only）──────────────────────────────────────
    if args.diag_only:
        from lerobot.utils.constants import OBS_STATE, OBS_IMAGES
        print("\n🔬 [--diag_only] 離線 Chunk 診斷模式（不連機器人）")

        # 預設初始角度取 dataset episode 0 第一幀的典型值
        DEFAULT_INIT = [-0.91, -98.36, 99.82, -98.64, 1.05, 2.20]
        init_angles = args.init_state if args.init_state is not None else DEFAULT_INIT
        JOINT_NAMES_DIAG = [
            'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
            'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos',
        ]
        joint_abbr = ["sp", "sl", "ef", "wf", "wr", "gr"]
        print(f"   初始關節角度（degrees）：")
        for jn, jv in zip(JOINT_NAMES_DIAG, init_angles):
            print(f"      {jn:25s}: {jv:8.3f}")

        state_np_d = np.array(init_angles, dtype=np.float32)
        obs = {}
        obs[OBS_STATE] = torch.from_numpy(state_np_d).float().unsqueeze(0).to(args.device)
        if global_stats and "observation.state" in global_stats:
            mn = global_stats["observation.state"]["mean"].to(args.device)
            st = global_stats["observation.state"]["std"].to(args.device)
            obs[OBS_STATE] = (obs[OBS_STATE] - mn) / (st + 1e-8)
        obs["task_index"] = torch.tensor([task_index], dtype=torch.long).to(args.device)
        # 使用隨機圖像（與 dummy 模式一致）
        for img_key in policy.config.image_features:
            obs[img_key] = torch.rand(1, 3, args.camera_height, args.camera_width).to(args.device)

        chunk_size = policy.config.chunk_size
        policy.reset()
        with torch.no_grad():
            raw_chunk = policy.predict_action_chunk(obs)  # (1, chunk_size, 6)
        raw_chunk_np = raw_chunk.squeeze(0).cpu().numpy()

        if global_stats and "action" in global_stats:
            act_mean_np = global_stats["action"]["mean"].numpy()
            act_std_np  = global_stats["action"]["std"].numpy()
            act_mean_t  = torch.tensor(act_mean_np).to(raw_chunk.device)
            act_std_t   = torch.tensor(act_std_np).to(raw_chunk.device)
            chunk_deg   = (raw_chunk * act_std_t + act_mean_t).squeeze(0).cpu().numpy()
        else:
            chunk_deg = raw_chunk_np

        print(f"\n📊 Chunk 診斷（{chunk_size} 步，隨機圖像 + 初始狀態）：")
        print(f"  [denorm degrees]")
        print(f"  {'step':>6}  " + "  ".join(f"{n:>7}" for n in joint_abbr))
        sample_steps = sorted(set(list(range(0, chunk_size, max(1, chunk_size // 10))) + [chunk_size - 1]))
        for ci in sample_steps:
            print(f"  {ci:>6}  " + "  ".join(f"{v:7.2f}" for v in chunk_deg[ci]))
        delta = chunk_deg[chunk_size - 1] - chunk_deg[0]
        print(f"  chunk[-1]-[0] 位移: " + "  ".join(f"{v:+.2f}" for v in delta))

        print(f"\n  [raw normalized] chunk[0]：")
        print("  " + "  ".join(f"{n:>7}" for n in joint_abbr))
        print("  " + "  ".join(f"{v:7.3f}" for v in raw_chunk_np[0]))

        print(f"\n  [chunk[0] vs 初始狀態（degrees）]（diff≈0 → 模型預測原地不動）：")
        print(f"  {'joint':>20}  {'chunk[0]':>9}  {'init':>9}  {'diff':>9}")
        for i, jn in enumerate(JOINT_NAMES_DIAG):
            diff = chunk_deg[0, i] - init_angles[i]
            print(f"  {jn:>20}  {chunk_deg[0,i]:9.3f}  {init_angles[i]:9.3f}  {diff:+9.3f}")

        if global_stats and "action" in global_stats:
            expected_copy = (state_np_d - act_mean_np) / (act_std_np + 1e-8)
            print(f"\n  [診斷] 若模型「複製當前狀態」raw[0] 應≈ {np.round(expected_copy, 2).tolist()}")
            print(f"  [實際] raw_chunk[0]              = {np.round(raw_chunk_np[0], 2).tolist()}")
            # 相似度
            diff_from_copy = np.abs(raw_chunk_np[0] - expected_copy).mean()
            diff_from_zero = np.abs(raw_chunk_np[0]).mean()
            print(f"\n  → 與「複製狀態」的平均絕對差 = {diff_from_copy:.4f}")
            print(f"  → 與「輸出 mean (=0)」的平均絕對差 = {diff_from_zero:.4f}")
            if diff_from_copy < diff_from_zero:
                print(f"  🔴 結論：模型輸出接近「複製當前狀態」→ 機器人接到「留在原地」指令")
            else:
                print(f"  🟡 結論：模型輸出接近 action mean → 機器人接到「移向平均位置」指令")

        print("\n📌 提示：若要用指定角度測試，使用 --init_state SP SL EF WF WR GR")
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

    from lerobot.utils.constants import OBS_STATE, OBS_IMAGES
    # 診斷：確認 image_features 是否正確載入
    print(f"   [診斷] policy.config.image_features = {policy.config.image_features}")
    policy.reset()

    JOINT_NAMES = [
        'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
        'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos',
    ]

    prev_action = None
    still_count = 0
    save_done = False

    print(f"\n▶  開始推論（{args.num_steps} steps, 目標頻率: {args.camera_fps} Hz）...")
    dt = 1.0 / args.camera_fps
    try:
        for step in range(args.num_steps):
            step_start_time = time.perf_counter()

            raw_obs = robot.get_observation()
            if step == 0:
                print(f"   原始 obs keys: {list(raw_obs.keys())}")
                # 打印初始關節角度
                initial_joints_display = {
                    j: float(raw_obs[j].item() if isinstance(raw_obs.get(j), torch.Tensor) else raw_obs.get(j, 0.0))
                    for j in JOINT_NAMES
                }
                print(f"   🦾 初始關節角度（degrees）：")
                for jn, jv in initial_joints_display.items():
                    print(f"      {jn:25s}: {jv:8.3f}")

            # ── 影像 ──────────────────────────────────────────────────────
            observation = {}
            # 訓練時圖像直接使用 [0,1]，process_image 保持一致（不做 ImageNet 正規化）
            for cam_key in ("front", "wrist"):
                if cam_key in raw_obs:
                    img_tensor = process_image(raw_obs[cam_key], args.device)
                    img_stat_key = f"observation.images.{cam_key}"
                    observation[img_stat_key] = img_tensor

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

            # ── 手動設定 OBS_IMAGES（修正：from_pretrained 不保留 input_features）──
            # policy.config.image_features 在 checkpoint 載入後為空 {}，導致
            # predict_action_chunk 跳過 batch[OBS_IMAGES] 的設定，模型無法看到影像。
            # 此處手動將兩顆相機影像依訓練時順序放入 OBS_IMAGES list。
            cam_images = []
            for cam_key in ("observation.images.front", "observation.images.wrist"):
                if cam_key in observation:
                    cam_images.append(observation[cam_key])
            if cam_images:
                observation[OBS_IMAGES] = cam_images

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

            # ── 首步：打印完整 chunk 診斷 ────────────────────────────────────────
            if step == 0:
                chunk_size = policy.config.chunk_size
                print(f"\n📊 [Step 0 Chunk 診斷] 模型預測完整 {chunk_size}-step 軌跡（每 10 步採樣）：")
                with torch.no_grad():
                    raw_chunk = policy.predict_action_chunk(observation)  # (1, chunk_size, 6)
                raw_chunk_np = raw_chunk.squeeze(0).cpu().numpy()  # (chunk_size, 6) 未 denorm
                if global_stats and "action" in global_stats:
                    act_mean_d = global_stats["action"]["mean"].to(raw_chunk.device)
                    act_std_d  = global_stats["action"]["std"].to(raw_chunk.device)
                    chunk_deg  = raw_chunk * act_std_d + act_mean_d  # 去歸一化
                else:
                    chunk_deg = raw_chunk
                chunk_np = chunk_deg.squeeze(0).cpu().numpy()  # (chunk_size, 6)
                joint_abbr = ["sp", "sl", "ef", "wf", "wr", "gr"]
                # 1) 完整 chunk（degree 空間）
                print(f"  [denorm degree]")
                print(f"  {'step':>6}  " + "  ".join(f"{n:>7}" for n in joint_abbr))
                sample_steps = list(range(0, chunk_size, max(1, chunk_size // 10))) + [chunk_size - 1]
                for ci in sorted(set(sample_steps)):
                    row = chunk_np[ci]
                    print(f"  {ci:>6}  " + "  ".join(f"{v:7.2f}" for v in row))
                delta = chunk_np[chunk_size - 1] - chunk_np[0]
                print(f"  chunk[-1]-[0] 位移: " + "  ".join(f"{v:+.2f}" for v in delta))
                # 2) 原始正規化值（判斷模型輸出是否接近 0 = mean）
                print(f"\n  [raw normalized] chunk[0]（接近 0 → 輸出 mean；接近 -1 → 複製初始狀態）：")
                print("  " + "  ".join(f"{n:>7}" for n in joint_abbr))
                print("  " + "  ".join(f"{v:7.3f}" for v in raw_chunk_np[0]))
                # 3) chunk[0] vs 當前狀態對比（直接判斷「複製狀態」行為）
                print(f"\n  [chunk[0] vs 當前狀態比較]（若 diff≈0 → 模型預測「原地不動」）：")
                print(f"  {'joint':>20}  {'chunk[0]':>9}  {'current':>9}  {'diff':>9}")
                for i, jn in enumerate(JOINT_NAMES):
                    diff = chunk_np[0, i] - state_np[i]
                    print(f"  {jn:>20}  {chunk_np[0,i]:9.3f}  {state_np[i]:9.3f}  {diff:+9.3f}")
                # 4) 如果是「複製狀態」，raw normalized 期望值應等於多少
                if global_stats and "action" in global_stats:
                    act_mean_np = global_stats["action"]["mean"].numpy()
                    act_std_np  = global_stats["action"]["std"].numpy()
                    expected_copy = (state_np - act_mean_np) / (act_std_np + 1e-8)
                    print(f"\n  [診斷] 若模型「複製當前狀態」，raw[0] 應≈ {np.round(expected_copy, 2).tolist()}")
                    print(f"  [實際] raw_chunk[0]              = {np.round(raw_chunk_np[0], 2).tolist()}")
                print()

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

            # FPS 控制：確保推論頻率與錄製頻率（30Hz）一致
            elapsed = time.perf_counter() - step_start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            actual_fps = 1.0 / (time.perf_counter() - step_start_time)

            # 前 10 步顯示詳細資訊（不覆蓋），方便診斷手臂是否在動
            if step < 10:
                print(f"  Step {step:4d}  cmd={np.round(action_np, 2)} | FPS: {actual_fps:.1f}")
                if step == 0:
                    print(f"         ↑ 比較初始角度：確認指令是否與起始位置不同")
            else:
                print(f"  Step {step:4d}  action={np.round(action_np[:4], 3)} | FPS: {actual_fps:.1f}", end="\r")

    except KeyboardInterrupt:
        print("\n⏹  使用者中斷推論。")
    finally:
        robot.disconnect()
        print("\n✅ 機器人已斷線。")


if __name__ == "__main__":
    main()
