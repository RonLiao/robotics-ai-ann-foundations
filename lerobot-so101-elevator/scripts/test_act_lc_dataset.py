# -*- coding: utf-8 -*-
import sys
import os

# 將父目錄加入 sys.path，以便可能需要引入其他模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from torch.utils.data import DataLoader
    from act_lc_dataset import ACTLCDataset
except ImportError as e:
    print(f"Import Error: {e}")
    print("請確認您在支援 LeRobot 的環境中執行此腳本。")
    sys.exit(1)

def test_dataset():
    # 使用我們練習階段錄製的 Dataset 進行測試
    # 若有新的多任務資料集，可改為 RonLiao/lerobot-so101-elevator-6btn-multitask
    repo_id = "RonLiao/lerobot-so101-elevator-dataset"
    print(f"1. Loading ACTLCDataset from {repo_id}...")
    
    try:
        dataset = ACTLCDataset(repo_id)
        print(f"   -> Dataset loaded. Total length: {len(dataset)}\n")
        
        print("2. Fetching first item (__getitem__)...")
        item = dataset[0]
        print("   -> Item Keys:", list(item.keys()))
        
        print("\n3. Verifying Text Instruction Injection...")
        if "language_instruction" in item:
            print(f"   ✅ Success! Found language_instruction: '{item['language_instruction']}'")
        else:
            print("   ❌ Failed! 'language_instruction' not found in item.")
            
        print("\n4. Verifying DataLoader Batching...")
        # 測試 Dataloader 是否能順利將字串組成 List
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        if "language_instruction" in batch:
            print(f"   ✅ DataLoader Success! Batch language_instruction:")
            print(f"      {batch['language_instruction']}")
        else:
            print("   ❌ DataLoader Failed to batch 'language_instruction'.")
            
        print("\n🎉 All Dataset integration tests finished!")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")

if __name__ == "__main__":
    test_dataset()
