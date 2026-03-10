#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import sys

def verify_latest_episode():
    # 配置路徑
    REPO_ID = "RonLiao/lerobot-so101-elevator-dataset"
    DATASET_PATH = Path.home() / ".cache/huggingface/lerobot" / REPO_ID
    DATA_DIR = DATASET_PATH / "data/chunk-000"

    if not DATA_DIR.exists():
        print(f"❌ 找不到數據目錄：{DATA_DIR}")
        return

    # 取得最新的一個 Parquet 檔案 (相容 episode_*.parquet 與 file-*.parquet)
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ 在 {DATA_DIR} 中找不到任何 episode 檔案")
        return

    latest_file = parquet_files[-1]
    print(f"🔍 正在驗證最新檔案: {latest_file.name}")

    try:
        df = pd.read_parquet(latest_file)
        print(f"=== {latest_file.name} 數據概況 ===")
        print(f"總幀數: {len(df)}")
        
        # 檢查關鍵欄位
        missing_cols = []
        for col in ["observation.images.front", "observation.state", "action"]:
            if col in df.columns:
                print(f"✅ 欄位 {col} 正常")
            else:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"⚠️ 缺失關鍵欄位: {missing_cols}")

        # 顯示軌跡變動確認
        if "observation.state" in df.columns:
            start_state = df["observation.state"].iloc[0]
            end_state = df["observation.state"].iloc[-1]
            print("\n手臂軌跡確認 (首幀 vs 末幀):")
            print(f"Start: {start_state}")
            print(f"End:   {end_state}")
            
            if (start_state == end_state).all():
                print("⚠️ 警告：手臂角度完全靜止，請檢查錄製是否成功")
            else:
                print("✅ 手臂角度有變動數據")

    except Exception as e:
        print(f"❌ 讀取檔案時發生錯誤: {e}")

if __name__ == "__main__":
    verify_latest_episode()
