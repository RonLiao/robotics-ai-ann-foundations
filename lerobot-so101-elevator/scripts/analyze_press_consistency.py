#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_press_consistency.py — 示範按壓位置一致性分析

對每個 episode 找出「按壓幀」（從 home 出發後關節總位移最大的幀），
統計各任務的按壓落點分散程度（std）。

用法：
  python scripts/analyze_press_consistency.py
  python scripts/analyze_press_consistency.py --repo_id RonLiao/lerobot-so101-elevator-6btn-dual-cam
"""

import os
import sys
import argparse
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
TASK_NAME_MAP = {0: "button_1", 1: "button_2", 2: "button_3",
                 3: "button_4", 4: "button_5", 5: "button_6"}


def main():
    parser = argparse.ArgumentParser(description="示範按壓位置一致性分析")
    parser.add_argument("--repo_id", type=str,
                        default="RonLiao/lerobot-so101-elevator-6btn-dual-cam",
                        help="HuggingFace dataset repo_id 或本地路徑")
    args = parser.parse_args()

    print(f"📂 載入資料集：{args.repo_id}")
    dataset = LeRobotDataset(repo_id=args.repo_id)
    hf = dataset.hf_dataset
    n_episodes = dataset.num_episodes
    print(f"  總集數 = {n_episodes}，總幀數 = {len(hf)}")

    # 從 episode_index 欄位推算每集的幀範圍（episodes 必為連續且有序）
    ep_indices = np.array(hf["episode_index"])
    boundaries = np.concatenate([
        [0],
        np.where(np.diff(ep_indices))[0] + 1,
        [len(ep_indices)]
    ])  # boundaries[ep] = 該 episode 起始幀（含）；boundaries[ep+1] = 結尾幀（不含）

    task_press_states = {}   # task_idx (int) -> list of np.array shape (6,)
    task_press_frames = {}   # task_idx -> list of (ep, press_frame_idx)

    for ep in range(n_episodes):
        start = int(boundaries[ep])
        end   = int(boundaries[ep + 1])   # exclusive

        # HF dataset slice → dict of lists
        ep_rows = hf[start:end]
        states = np.array(ep_rows["observation.state"])  # (T, 6)

        # task_index 在此 episode 固定，取第一幀
        raw_tidx = ep_rows["task_index"][0]
        task_idx = int(raw_tidx.item() if hasattr(raw_tidx, "item") else raw_tidx)

        # press 幀：從第 0 幀（home 位置）出發，六關節 L2 位移最大的幀
        home = states[0]
        displacements = np.linalg.norm(states - home, axis=1)
        press_frame = int(np.argmax(displacements))

        if task_idx not in task_press_states:
            task_press_states[task_idx] = []
            task_press_frames[task_idx] = []
        task_press_states[task_idx].append(states[press_frame])
        task_press_frames[task_idx].append((ep, press_frame))

        if ep % 30 == 0:
            print(f"  ep {ep:3d}: task={task_idx}  press_frame={press_frame:4d}  "
                  f"max_disp={displacements[press_frame]:6.1f}°  "
                  f"T={len(states)}")

    # ── 統計輸出 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("📊 示範按壓位置一致性分析（關節角度，單位：度）")
    print(f"{'='*78}")

    summary_rows = []

    for task_idx in sorted(task_press_states.keys()):
        arr = np.array(task_press_states[task_idx])   # (N, 6)
        n = len(arr)
        task_name = TASK_NAME_MAP.get(task_idx, f"task_{task_idx}")

        print(f"\n  任務：{task_name}（{n} 集）")
        print(f"  {'關節':<18}{'平均':>8}  {'std':>7}  {'max-min':>9}  評估")
        print(f"  {'-'*65}")

        for j, jname in enumerate(JOINT_NAMES):
            mean = arr[:, j].mean()
            std  = arr[:, j].std()
            rng  = arr[:, j].max() - arr[:, j].min()

            # 末端關節（wrist / gripper）標準較嚴
            is_wrist = "wrist" in jname or jname == "gripper"
            if std < (1.0 if is_wrist else 2.0):
                rating = "✅ 一致"
            elif std < (2.0 if is_wrist else 4.0):
                rating = "⚠️ 中等"
            else:
                rating = "❌ 分散"

            print(f"  {jname:<18}{mean:>8.2f}°  {std:>7.2f}°  {rng:>9.2f}°  {rating}")

        # 末端整體評估：用 wrist_flex + wrist_roll 的平均 std
        wrist_std = np.mean([arr[:, 3].std(), arr[:, 4].std()])
        if wrist_std < 1.0:
            verdict = "✅ 一致，補錄時維持同樣習慣即可"
        elif wrist_std < 2.0:
            verdict = "⚠️ 中等，補錄時請有意識地對準按鈕中心"
        else:
            verdict = "❌ 分散（std>2°），補更多同樣資料效果有限，需改善示範一致性"
        print(f"\n  → 末端關節平均 std = {wrist_std:.2f}°  {verdict}")
        summary_rows.append((task_name, wrist_std, verdict))

    # ── 摘要 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("📋 摘要")
    print(f"{'='*78}")
    for task_name, wrist_std, verdict in summary_rows:
        print(f"  {task_name:<12}  末端 std={wrist_std:.2f}°  {verdict}")
    print()
    print("💡 參考換算：末端 1° ≈ 末端位移 ~0.5cm（依臂長而異）")
    print("   std < 1° → 補錄量有意義；std 1~2° → 示範需更精準；std > 2° → 先改善手法再補錄")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
