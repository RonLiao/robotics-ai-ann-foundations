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

    # 匯總所有可能的數據檔案
    available_files = list(DATA_DIR.iterdir())
    parquet_files = sorted([f for f in available_files if f.suffix == ".parquet"])
    
    if not parquet_files:
        print(f"❌ 在 {DATA_DIR} 中找不到任何 .parquet 檔案")
        print(f"資料夾內現有檔案：{[f.name for f in available_files]}")
        return

    latest_file = parquet_files[-1]
    print(f"🔍 正在驗證最新檔案: {latest_file.name}")

    try:
        df = pd.read_parquet(latest_file)
        print(f"=== {latest_file.name} 數據概況 ===")
        print(f"總幀數: {len(df)}")
        
        # 檢查關鍵欄位 (支援多種可能的影像欄位命名)
        found_cols = df.columns.tolist()
        has_images = any(c for c in found_cols if "images" in c)
        
        if has_images:
            image_col = [c for c in found_cols if "images" in c][0]
            print(f"✅ 發現影像欄位: {image_col}")
        else:
            print(f"⚠ 缺失影像欄位！現有欄位: {found_cols}")

        for col in ["observation.state", "action"]:
            if col in found_cols:
                print(f"✅ 欄位 {col} 正常")
            else:
                print(f"❌ 缺失關鍵欄位: {col}")

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
