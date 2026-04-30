#!/usr/bin/env python3
"""
check_train_frames.py
--------------------
Extract camera frames from the training dataset for visual comparison with
inference-time images. Saves one frame per episode (first frame by default)
across all tasks, and generates a side-by-side comparison grid per task.

Usage:
    python scripts/check_train_frames.py
    python scripts/check_train_frames.py --repo_id RonLiao/lerobot-so101-elevator-6btn-multitask
    python scripts/check_train_frames.py --episode 5 --frame 10   # specific episode/frame
    python scripts/check_train_frames.py --per_task 3             # 3 samples per task
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a (C, H, W) float [0,1] or uint8 tensor to a PIL image."""
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    t = t.clamp(0.0, 1.0)
    return TF.to_pil_image(t)


def add_label(img: Image.Image, text: str, font_size: int = 18) -> Image.Image:
    """Burn a text label into the bottom of an image."""
    draw = ImageDraw.Draw(img)
    # Use default PIL font — no external font needed
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    y = img.height - th - 6

    # Dark shadow for readability
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 0), font=font)
    return img


def make_grid(images: list, cols: int = 5, pad: int = 4) -> Image.Image:
    """Arrange a list of PIL images into a grid."""
    if not images:
        return Image.new("RGB", (1, 1))
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * (w + pad) - pad, rows * (h + pad) - pad), (40, 40, 40))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * (w + pad), r * (h + pad)))
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract training dataset frames for visual inspection")
    parser.add_argument("--repo_id", default="RonLiao/lerobot-so101-elevator-6btn-multitask",
                        help="HuggingFace dataset repo ID (or local cache)")
    parser.add_argument("--out_dir", default="outputs/train_frames",
                        help="Directory to save extracted frames")
    parser.add_argument("--episode", type=int, default=None,
                        help="Extract frames from a specific episode index (0-indexed)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Which frame offset within an episode to extract (default: 0 = first frame)")
    parser.add_argument("--per_task", type=int, default=3,
                        help="How many sample frames to extract per task (default: 3)")
    parser.add_argument("--cols", type=int, default=5,
                        help="Columns in the grid image (default: 5)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.repo_id} ...")
    dataset = LeRobotDataset(args.repo_id)

    # -- Discover image keys --------------------------------------------------
    sample0 = dataset[0]
    image_keys = [k for k in sample0 if "image" in k.lower()]
    if not image_keys:
        print("ERROR: No image keys found in dataset. Available keys:")
        for k in sample0:
            print(f"  {k}: {type(sample0[k])}")
        return

    print(f"Found {len(image_keys)} image key(s): {image_keys}")
    print(f"Total frames in dataset: {len(dataset)}")

    # -- Build episode -> frame-index mapping ---------------------------------
    # LeRobot v3: use meta.episodes list; each entry has 'length' field.
    # We rebuild cumulative frame ranges (from, to) ourselves.
    ep_meta_list = []
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        raw = dataset.meta.episodes
        if isinstance(raw, list):
            ep_meta_list = raw
        elif hasattr(raw, "__len__"):  # dict-like or other sequence
            ep_meta_list = [raw[i] for i in range(len(raw))]

    if not ep_meta_list:
        print("ERROR: Cannot read episode metadata from dataset.meta.episodes")
        return

    num_episodes = len(ep_meta_list)
    print(f"Total episodes: {num_episodes}")

    # Build frame-range index using the pre-computed fields in each episode dict
    ep_from = [item["dataset_from_index"] for item in ep_meta_list]
    ep_to   = [item["dataset_to_index"]   for item in ep_meta_list]

    # -- Build task label per episode -----------------------------------------
    # LeRobot v3 tasks DataFrame: index=task_string, column="task_index" (int)
    # We need a reverse mapping: {task_index_int -> task_string}
    tasks_map: dict = {}
    obj_tasks = getattr(dataset, "tasks", None)
    if obj_tasks is None and hasattr(dataset, "meta"):
        obj_tasks = getattr(dataset.meta, "tasks", None)

    if obj_tasks is not None:
        if isinstance(obj_tasks, pd.DataFrame) and "task_index" in obj_tasks.columns:
            # index = task string, "task_index" column = integer key
            for task_str, row in obj_tasks.iterrows():
                tasks_map[int(row["task_index"])] = str(task_str)
        elif isinstance(obj_tasks, pd.Series):
            for task_str, task_idx in obj_tasks.items():
                tasks_map[int(task_idx)] = str(task_str)
        elif isinstance(obj_tasks, dict):
            # might be {task_index: task_string} or {task_string: task_index}
            for k, v in obj_tasks.items():
                if isinstance(v, str):
                    tasks_map[int(k)] = v
                else:
                    tasks_map[int(v)] = str(k)
        elif isinstance(obj_tasks, list):
            tasks_map = {i: s for i, s in enumerate(obj_tasks)}
    else:
        print("WARNING: dataset.tasks and dataset.meta.tasks are both None")

    print(f"Known tasks ({len(tasks_map)}): {tasks_map}")

    def get_task_label(ep_idx: int) -> str:
        """Retrieve task string for a given episode index."""
        item = ep_meta_list[ep_idx]
        # LeRobot v3: 'tasks' is a list of task strings e.g. ['press button 1']
        val = item.get("tasks", None) if isinstance(item, dict) else getattr(item, "tasks", None)
        if val is not None:
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return str(val[0])
            return str(val)
        return "unknown"

    # -- Single episode mode --------------------------------------------------
    if args.episode is not None:
        ep = args.episode
        frame_start = ep_from[ep]
        frame_end = ep_to[ep]
        target_frame = min(frame_start + args.frame, frame_end - 1)
        task_label = get_task_label(ep)

        print(f"\nEpisode {ep} | Task: '{task_label}' | Frames {frame_start}–{frame_end}")
        print(f"Extracting frame offset +{args.frame} → global frame index {target_frame}")

        sample = dataset[target_frame]
        for cam_key in image_keys:
            img = tensor_to_pil(sample[cam_key])
            label_text = f"ep{ep} | frame+{args.frame} | {cam_key.split('.')[-1]}"
            img = add_label(img, label_text)
            fname = out_dir / f"train_ep{ep:03d}_frame{args.frame:03d}_{cam_key.split('.')[-1]}.png"
            img.save(fname)
            print(f"  Saved: {fname}")
        return

    # -- Per-task sampling mode -----------------------------------------------
    # Group episodes by task label
    task_episodes: dict = defaultdict(list)
    for ep in range(num_episodes):
        label = get_task_label(ep)
        task_episodes[label].append(ep)

    print(f"\nEpisodes per task:")
    for t, eps in task_episodes.items():
        print(f"  '{t}': {len(eps)} episodes (e.g. ep indices {eps[:5]}...)")

    for cam_key in image_keys:
        cam_name = cam_key.split(".")[-1]
        cam_out_dir = out_dir / cam_name
        cam_out_dir.mkdir(parents=True, exist_ok=True)

        for task_label, ep_list in task_episodes.items():
            # Sample evenly across available episodes
            step = max(1, len(ep_list) // args.per_task)
            sampled_eps = ep_list[::step][: args.per_task]

            task_images = []
            for ep in sampled_eps:
                frame_start = ep_from[ep]
                frame_end = ep_to[ep]
                target_frame = min(frame_start + args.frame, frame_end - 1)
                sample = dataset[target_frame]

                img = tensor_to_pil(sample[cam_key])
                label_text = f"ep{ep} | frame+{args.frame}"
                img = add_label(img, label_text)
                task_images.append(img)

                # Also save individual file
                safe_task = task_label.replace(" ", "_").replace("/", "-")
                fname = cam_out_dir / f"{safe_task}_ep{ep:03d}_frame{args.frame:03d}.png"
                img.save(fname)

            # Save grid for this task
            grid = make_grid(task_images, cols=args.cols)
            safe_task = task_label.replace(" ", "_").replace("/", "-")
            grid_path = cam_out_dir / f"grid_{safe_task}.png"
            grid.save(grid_path)
            print(f"  [{cam_name}] Task '{task_label}': saved {len(task_images)} frames → grid: {grid_path}")

    print(f"\nDone. All frames saved to: {out_dir.resolve()}")
    print("\nNext step: run inference and save the first frame, then compare side-by-side.")
    print("Tip: compare outputs/train_frames/<cam_name>/grid_press_button_3.png with your inference snapshot.")


if __name__ == "__main__":
    main()
