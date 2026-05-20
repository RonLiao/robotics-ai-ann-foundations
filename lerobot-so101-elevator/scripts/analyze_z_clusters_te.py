#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_z_clusters_te.py — ACT Task Embedding (act_te) VAE z 群集分析

與 analyze_z_clusters.py 的差異：
  - Monkey-patch 目標為 act_te 而非 act_lc
  - 不依賴 language_instruction，改以 task_index 作為任務標籤
  - 其餘分析邏輯（VAE encoder 結構、分離比計算）完全相同

用法：
  python scripts/analyze_z_clusters_te.py \
    --checkpoint outputs/train/act_te_btn_1_to_3_dualcam_v1/checkpoints/last/pretrained_model \
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

# ── Monkey-patch：替換 ACTPolicy 與 ACTConfig → act_te 版本 ─────────────────
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

    from policies.act_te.modeling_act import ACTPolicy as TEACTPolicy
    policy_factory.ACTPolicy = TEACTPolicy
    act_modeling.ACTPolicy = TEACTPolicy

    try:
        try:
            import lerobot.policies.act.configuration_act as act_config_module
        except ImportError:
            import lerobot.common.policies.act.configuration_act as act_config_module
        from policies.act_te.configuration_act import ACTConfig as TEACTConfig
        act_config_module.ACTConfig = TEACTConfig
    except Exception as _e:
        print(f"⚠️ 替換 ACTConfig 時發生錯誤: {_e}")

    try:
        from lerobot.configs.policies import PreTrainedConfig
        for _attr in ['_registry', '__registry__', '_subclass_registry', '_choice_registry']:
            _reg = getattr(PreTrainedConfig, _attr, None)
            if isinstance(_reg, dict) and 'act' in _reg:
                _reg['act'] = TEACTConfig
                break
    except Exception as _e:
        print(f"⚠️ 替換 PreTrainedConfig registry 時發生錯誤: {_e}")
except Exception as e:
    print(f"⚠️ Monkey-patch 失敗: {e}")

import torch
import einops
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from policies.act_te.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_STATE, ACTION

TASK_NAME_MAP = {0: "button_1", 1: "button_2", 2: "button_3",
                 3: "button_4", 4: "button_5", 5: "button_6"}


def extract_mu_batch(policy, batch, device):
    """VAE encoder 前向，取出 z_mean (mu)。與 act_lc 版本邏輯相同。"""
    model = policy.model
    config = model.config

    state = batch[OBS_STATE].to(device)
    action = batch[ACTION].to(device)
    action_is_pad = batch["action_is_pad"].to(device)
    B = state.shape[0]

    with torch.no_grad():
        if state.ndim == 3:
            state = state[:, 0, :]

        cls_embed = einops.repeat(
            model.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=B
        )
        robot_state_embed = model.vae_encoder_robot_state_input_proj(state).unsqueeze(1)
        action_embed = model.vae_encoder_action_input_proj(action)

        vae_input = torch.cat([cls_embed, robot_state_embed, action_embed], dim=1)
        pos_embed = model.vae_encoder_pos_enc.clone().detach()

        cls_state_not_pad = torch.zeros(B, 2, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([cls_state_not_pad, action_is_pad], dim=1)

        cls_token_out = model.vae_encoder(
            vae_input.permute(1, 0, 2),
            pos_embed=pos_embed.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]

        latent_pdf_params = model.vae_encoder_latent_output_proj(cls_token_out)
        mu = latent_pdf_params[:, :config.latent_dim]

    return mu.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="act_te VAE z 群集分析")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/train/act_te_btn_1_to_3_dualcam_v1/checkpoints/last/pretrained_model")
    parser.add_argument("--repo_id", type=str,
                        default="RonLiao/lerobot-so101-elevator-6btn-dual-cam")
    parser.add_argument("--num_samples", type=int, default=300,
                        help="每個任務最多分析的樣本數（-1 表示全部）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── 1. 載入模型 ──────────────────────────────────────────────────────────
    ckpt_path = os.path.join(root_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    print(f"\n📥 載入模型：{ckpt_path}")
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(args.device)
    policy.eval()
    print(f"   latent_dim  = {policy.config.latent_dim}")
    print(f"   chunk_size  = {policy.config.chunk_size}")
    print(f"   num_tasks   = {policy.config.num_tasks}")

    # ── 2. 載入資料集 ─────────────────────────────────────────────────────────
    chunk_size = policy.config.chunk_size
    delta_timestamps = {"action": [i / args.fps for i in range(chunk_size)]}
    print(f"\n📂 載入資料集：{args.repo_id}")
    dataset = ACTLCDataset(repo_id=args.repo_id, delta_timestamps=delta_timestamps)
    print(f"   總樣本數 = {len(dataset)}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    # ── 3. 收集各任務的 z_mean ────────────────────────────────────────────────
    task_mus: dict[str, list] = {}
    max_per_task = args.num_samples if args.num_samples > 0 else float("inf")
    print(f"\n🔍 開始提取 z_mean（每任務上限 {args.num_samples} 個）...\n")

    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print(f"  [診斷] batch keys: {list(batch.keys())}")

        try:
            batch = policy.normalize_inputs(batch)
            if ACTION in batch:
                batch = policy.normalize_targets(batch)
        except Exception:
            pass

        raw_task_idx = batch.get("task_index", None)
        mu_batch = extract_mu_batch(policy, batch, args.device)

        for i in range(mu_batch.shape[0]):
            tidx = raw_task_idx[i] if raw_task_idx is not None else 0
            if isinstance(tidx, torch.Tensor):
                tidx = tidx.item()
            label = TASK_NAME_MAP.get(int(tidx), f"task_{int(tidx)}")

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

    # ── 4. 統計分析 ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 act_te VAE z 群集統計分析")
    print(f"{'='*60}")

    if len(task_mus) < 2:
        print("⚠️ 偵測到的任務數不足 2 個，無法計算群集距離。")
        return

    task_arrays = {k: np.array(v) for k, v in task_mus.items()}
    task_names = sorted(task_arrays.keys())

    task_centroids = {}
    task_intra_var = {}
    for t in task_names:
        arr = task_arrays[t]
        centroid = arr.mean(axis=0)
        intra_var = ((arr - centroid) ** 2).sum(axis=1).mean()
        task_centroids[t] = centroid
        task_intra_var[t] = float(intra_var)
        print(f"\n  任務: '{t}'  (n={len(arr)})")
        print(f"    mu 均值（前4維）  = {centroid[:4]}")
        print(f"    mu 標準差（前4維）= {arr.std(axis=0)[:4]}")
        print(f"    群集內方差（MSE） = {intra_var:.4f}")

    print(f"\n{'─'*60}")
    inter_distances = []
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if j <= i:
                continue
            dist = float(np.linalg.norm(task_centroids[t1] - task_centroids[t2]))
            inter_distances.append(dist)
            print(f"  '{t1}' ↔ '{t2}': {dist:.4f}")

    mean_intra = np.mean(list(task_intra_var.values()))
    mean_inter = np.mean(inter_distances) if inter_distances else 0.0
    separation_ratio = mean_inter / (np.sqrt(mean_intra) + 1e-8)

    print(f"\n  平均群集內標準差（√intra_var）: {np.sqrt(mean_intra):.4f}")
    print(f"  平均群集間距離（inter_dist）   : {mean_inter:.4f}")
    print(f"  分離比（inter / √intra）       : {separation_ratio:.2f}")

    # ── 5. 結論 ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if separation_ratio > 3.0:
        verdict = f"✅ z 高度任務可分（ratio={separation_ratio:.2f} >> 3.0）\n  → z 攜帶任務資訊；降低 kl_weight 意義有限，z 已足夠分離。"
    elif separation_ratio > 1.5:
        verdict = f"⚠️ z 中度任務可分（ratio={separation_ratio:.2f}，1.5~3.0）\n  → 部分任務資訊在 z 中；可嘗試降低 kl_weight 進一步提升分離度。"
    else:
        verdict = (f"❌ z 幾乎不可分（ratio={separation_ratio:.2f} < 1.5）\n"
                   f"  → z 未攜帶任務資訊（kl_weight 過強）。\n"
                   f"  → 降低 kl_weight 可讓 encoder 分離不同路徑，但效果需重訓驗證；\n"
                   f"  → 重新錄製一致示範為更可靠的方向。")
    print(verdict)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
