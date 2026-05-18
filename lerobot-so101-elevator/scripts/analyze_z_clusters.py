#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE z 群集分析工具 (analyze_z_clusters.py)

假設：VAE encoder 在訓練時能從 [state + action_chunk] 識別任務（按鈕1/2/3），
     因此 z_mean (mu) 會依任務形成分離的群集。
     若群集間距離 >> 群集內方差，代表 z 吸收了任務辨識資訊，
     推論時 z=0 會導致 Mode Collapse，z-dropout 訓練有理論依據。

用法（在容器內，進入 lerobot-so101-elevator 目錄後執行）:
  python scripts/analyze_z_clusters.py \
    --checkpoint outputs/train/act_lc_btn_1_to_3_dualcam_v3/checkpoints/last/pretrained_model \
    --repo_id RonLiao/lerobot-so101-elevator-6btn-dual-cam \
    --num_samples 300 \
    --batch_size 32
"""

import os
import sys
import argparse
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# ── Monkey-patch: 替換 ACTPolicy 與 ACTConfig ──────────────────────────────
import lerobot.datasets.lerobot_dataset
from act_lc_dataset import ACTLCDataset
lerobot.datasets.lerobot_dataset.LeRobotDataset = ACTLCDataset

try:
    try:
        import lerobot.policies.factory as policy_factory
        import lerobot.policies.act.modeling_act as act_modeling
    except ImportError:
        import lerobot.common.policies.factory as policy_factory
        import lerobot.common.policies.act.modeling_act as act_modeling

    from policies.act_lc.modeling_act import ACTPolicy as CustomACTPolicy
    policy_factory.ACTPolicy = CustomACTPolicy
    act_modeling.ACTPolicy = CustomACTPolicy

    try:
        try:
            import lerobot.policies.act.configuration_act as act_config_module
        except ImportError:
            import lerobot.common.policies.act.configuration_act as act_config_module
        from policies.act_lc.configuration_act import ACTConfig as CustomACTConfig
        act_config_module.ACTConfig = CustomACTConfig
    except Exception as e:
        print(f"⚠️ 替換 ACTConfig 時發生錯誤: {e}")
except Exception as e:
    print(f"⚠️ Monkey-patch 失敗: {e}")

import torch
import einops
from torch.utils.data import DataLoader, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from policies.act_lc.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_STATE, ACTION


def extract_mu_batch(policy, batch, device):
    """
    直接呼叫 VAE encoder，從一個訓練批次中提取 z_mean (mu)。
    不需要呼叫完整 forward，避免視覺 backbone 的 BatchNorm 問題。
    """
    model = policy.model
    config = model.config

    state = batch[OBS_STATE].to(device)   # (B, state_dim)
    action = batch[ACTION].to(device)      # (B, chunk_size, action_dim)
    action_is_pad = batch["action_is_pad"].to(device)  # (B, chunk_size)
    B = state.shape[0]

    with torch.no_grad():
        # state 可能因 delta_timestamps 設定不同而帶有時間維度 (B, 1, state_dim) 或 (B, state_dim)
        if state.ndim == 3:
            state = state[:, 0, :]  # 取第一個時間步 → (B, state_dim)

        # [CLS] token
        cls_embed = einops.repeat(
            model.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=B
        )  # (B, 1, D)

        # Robot state projection
        robot_state_embed = model.vae_encoder_robot_state_input_proj(state).unsqueeze(1)  # (B, 1, D)

        # Action sequence projection
        action_embed = model.vae_encoder_action_input_proj(action)  # (B, S, D)

        # Concatenate: [cls, state, action_seq]
        vae_input = torch.cat([cls_embed, robot_state_embed, action_embed], dim=1)  # (B, S+2, D)

        # Positional encoding
        pos_embed = model.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

        # Key padding mask (False = not padding for cls/state, then action_is_pad)
        cls_state_not_pad = torch.zeros(B, 2, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_state_not_pad, action_is_pad], dim=1)  # (B, S+2)

        # VAE encoder forward
        cls_token_out = model.vae_encoder(
            vae_input.permute(1, 0, 2),
            pos_embed=pos_embed.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]  # (B, D) — first output = cls token

        # Latent distribution parameters
        latent_pdf_params = model.vae_encoder_latent_output_proj(cls_token_out)
        mu = latent_pdf_params[:, :config.latent_dim]  # (B, latent_dim)

    return mu.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="VAE z 群集分析工具")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/train/act_lc_btn_1_to_3_dualcam_v3/checkpoints/last/pretrained_model",
                        help="本地 checkpoint 路徑（pretrained_model 目錄）")
    parser.add_argument("--repo_id", type=str,
                        default="RonLiao/lerobot-so101-elevator-6btn-dual-cam",
                        help="HuggingFace dataset repo_id")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="每個任務最多分析的樣本數（-1 表示全部）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30,
                        help="Dataset FPS（用於計算 delta_timestamps，須與錄製時一致）")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── 1. 載入模型 ──────────────────────────────────────────────────────────
    ckpt_path = os.path.join(root_dir, args.checkpoint)
    print(f"\n📥 載入模型：{ckpt_path}")
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(args.device)
    policy.eval()

    chunk_size = policy.config.chunk_size
    print(f"  latent_dim = {policy.config.latent_dim}")
    print(f"  chunk_size = {chunk_size}")
    print(f"  fps        = {args.fps}")

    # ── 2. 載入資料集 ─────────────────────────────────────────────────────────
    # VAE encoder 只需要 state + action，無需影像。
    # delta_timestamps 只對 action 設定時間窗口（chunk_size 幀）；
    # observation.state 無 delta_timestamps → 直接返回當前幀 shape=(B, state_dim)。
    print(f"\n📂 載入資料集：{args.repo_id}")
    delta_timestamps = {
        "action": [i / args.fps for i in range(chunk_size)],
    }
    dataset = ACTLCDataset(repo_id=args.repo_id, delta_timestamps=delta_timestamps)
    print(f"  總樣本數 = {len(dataset)}")

    # ── 3. 建立 DataLoader ────────────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,   # 打亂順序確保各任務均勻採樣，不會先把 task1 跑完才碰 task2
        num_workers=0,
        drop_last=False,
    )

    # ── 4. 建立 task_index → 任務名稱 對照表 ────────────────────────────────
    # 從 dataset.meta.tasks 建立 {int → str} 映射，作為 language_instruction 為空時的備援
    task_index_to_name: dict[int, str] = {}
    try:
        meta = dataset.meta if hasattr(dataset, "meta") else dataset.dataset.meta
        tasks_info = meta.tasks
        if isinstance(tasks_info, dict):
            for k, v in tasks_info.items():
                name = v.get("task", str(v)) if isinstance(v, dict) else str(v)
                task_index_to_name[int(k)] = name
        elif isinstance(tasks_info, list):
            for i, v in enumerate(tasks_info):
                name = v.get("task", str(v)) if isinstance(v, dict) else str(v)
                task_index_to_name[i] = name
        print(f"  task_index 對照表：{task_index_to_name}")
    except Exception as e:
        print(f"  ⚠️ 無法建立 task_index 對照表：{e}")

    # ── 5. 收集每個任務的 z_mean ──────────────────────────────────────────────
    task_mus: dict[str, list] = {}
    max_per_task = args.num_samples if args.num_samples > 0 else float("inf")

    print(f"\n🔍 開始提取 z_mean（每任務上限 {args.num_samples} 個）...\n")

    for batch_idx, batch in enumerate(loader):
        # 第一批次：印出鍵值診斷
        if batch_idx == 0:
            print(f"  [診斷] batch keys: {list(batch.keys())}")
            if "language_instruction" in batch:
                print(f"  [診斷] language_instruction[0:3]: {batch['language_instruction'][:3]}")
            if "task_index" in batch:
                print(f"  [診斷] task_index[0:3]: {batch['task_index'][:3]}")
            print()

        # 應用輸入正規化（由 PreTrainedPolicy 的 normalizer 處理）
        try:
            batch = policy.normalize_inputs(batch)
            if ACTION in batch:
                batch = policy.normalize_targets(batch)
        except Exception:
            pass  # 若不支援，直接用原始資料（z 分析仍有效，只是 scale 不同）

        # 解析 label：優先用 language_instruction，備援用 task_index
        raw_labels = batch.get("language_instruction", [""] * args.batch_size)
        raw_task_idx = batch.get("task_index", None)

        mu_batch = extract_mu_batch(policy, batch, args.device)  # (B, latent_dim)

        for i in range(mu_batch.shape[0]):
            label = raw_labels[i] if i < len(raw_labels) else ""
            if not isinstance(label, str):
                label = label[0] if isinstance(label, (list, tuple)) else str(label)
            label = label.strip()

            # 備援：用 task_index 查表
            if not label and raw_task_idx is not None:
                tidx = raw_task_idx[i]
                if isinstance(tidx, torch.Tensor):
                    tidx = tidx.item()
                label = task_index_to_name.get(int(tidx), f"task_{int(tidx)}")

            if not label:
                continue

            if label not in task_mus:
                task_mus[label] = []
            if len(task_mus[label]) < max_per_task:
                task_mus[label].append(mu_batch[i])

        all_full = bool(task_mus) and all(len(v) >= max_per_task for v in task_mus.values())

        if batch_idx % 10 == 0 or all_full:
            summary = "  |  ".join(f"{k}: {len(v)}" for k, v in sorted(task_mus.items()))
            print(f"  Batch {batch_idx:04d}  [{summary}]")

        if all_full:
            print("  ✅ 所有任務已達樣本上限，提前結束。")
            break

    # ── 5. 統計分析 ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 VAE z 群集統計分析")
    print(f"{'='*60}")

    if len(task_mus) < 2:
        print("⚠️ 偵測到的任務數不足 2 個，無法計算群集距離。請檢查資料集的 language_instruction 欄位。")
        return

    # 轉換為 numpy arrays
    task_arrays = {k: np.array(v) for k, v in task_mus.items()}
    task_names = sorted(task_arrays.keys())

    # 計算每個任務的群集中心 (mean) 與內部方差
    task_centroids = {}
    task_intra_var = {}
    for t in task_names:
        arr = task_arrays[t]  # (N, latent_dim)
        centroid = arr.mean(axis=0)
        intra_var = ((arr - centroid) ** 2).sum(axis=1).mean()
        task_centroids[t] = centroid
        task_intra_var[t] = float(intra_var)
        print(f"\n  任務: '{t}'")
        print(f"    樣本數    = {len(arr)}")
        print(f"    mu 均值   = {centroid[:8]} ... (前8維)")
        print(f"    mu 標準差 = {arr.std(axis=0)[:8]} ...")
        print(f"    群集內方差（intra-cluster MSE） = {intra_var:.4f}")

    # 計算群集間距離（Euclidean）
    print(f"\n{'─'*60}")
    print("  群集間距離（Inter-cluster Euclidean Distance）:")
    inter_distances = []
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if j <= i:
                continue
            dist = float(np.linalg.norm(task_centroids[t1] - task_centroids[t2]))
            inter_distances.append(dist)
            print(f"    '{t1}' ↔ '{t2}': {dist:.4f}")

    mean_intra = np.mean(list(task_intra_var.values()))
    mean_inter = np.mean(inter_distances) if inter_distances else 0.0
    separation_ratio = mean_inter / (np.sqrt(mean_intra) + 1e-8)

    print(f"\n  平均群集內標準差（√intra_var）: {np.sqrt(mean_intra):.4f}")
    print(f"  平均群集間距離（inter_dist）   : {mean_inter:.4f}")
    print(f"  分離比（inter / √intra）       : {separation_ratio:.2f}")

    # ── 6. 結論 ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("🧪 假設驗證結論")
    print(f"{'='*60}")

    if separation_ratio > 3.0:
        print(f"  ✅ z 高度任務可分 (ratio={separation_ratio:.2f} >> 3.0)")
        print("  → VAE z 在訓練時確實吸收了任務辨識資訊。")
        print("  → 推論時 z=0 跳過了任務線索，Mode Collapse 根因確認。")
        print("  → z-dropout 訓練有強力理論依據，建議立即實施。")
    elif separation_ratio > 1.5:
        print(f"  ⚠️ z 中度任務可分 (ratio={separation_ratio:.2f}，1.5~3.0)")
        print("  → z 攜帶部分任務資訊，但語言信號也可能共同承擔。")
        print("  → z-dropout 仍值得嘗試，但效果可能有限；同時考慮增加訓練資料。")
    else:
        print(f"  ❌ z 幾乎不可分 (ratio={separation_ratio:.2f} < 1.5)")
        print("  → VAE z 未吸收顯著任務資訊，z-dropout 對語言學習助益有限。")
        print("  → Mode Collapse 根因可能在於視覺特徵高度相似，而非 z 競爭。")
        print("  → 建議：增加視覺多樣性（光線、角度）、更長訓練步數。")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
