# -*- coding: utf-8 -*-
import sys
import os
import torch

# 將 lerobot-so101-elevator 目錄加入 sys.path 以便 import policies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from policies.act_lc.configuration_act import ACTConfig
    from policies.act_lc.modeling_act import ACTPolicy
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_forward():
    print("1. Initializing ACTConfig...")
    # Mock features metadata that PreTrainedConfig expects in LeRobot
    config = ACTConfig(
        language_model_name="distilbert-base-uncased",
        language_dim=768,
        max_text_length=16,
        vision_backbone="resnet18",
        dim_model=512,
        chunk_size=10,
        n_action_steps=10
    )
    
    # 手動設定 input_features (LeRobot 內部是用這兩個欄位推導出 image_features 等屬性)
    from lerobot.configs.types import FeatureType
    class MockFeature:
        def __init__(self, shape, type_enc):
            self.shape = tuple(shape)
            self.type = type_enc

    config.input_features = {
        "observation.images.cam_high": MockFeature([3, 480, 640], FeatureType.VISUAL),
        "observation.state": MockFeature([6], FeatureType.STATE)
    }
    config.output_features = {
        "action": MockFeature([6], FeatureType.ACTION)
    }

    print("2. Initializing ACTPolicy (ACT-LC)...")
    policy = ACTPolicy(config)
    # 不載入權重，僅驗證資料流維度
    
    print("3. Preparing Mock Batch...")
    batch_size = 2
    batch = {
        "observation.images.cam_high": torch.randn(batch_size, 3, 480, 640),
        "observation.state": torch.randn(batch_size, 6),
        "action": torch.randn(batch_size, 10, 6),
        "action_is_pad": torch.zeros(batch_size, 10, dtype=torch.bool),
        "language_instruction": ["press button 3", "press button 5"]
    }

    print("4. Testing predict_action_chunk (Inference Mode)...")
    policy.eval()
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    print(f"  -> Predicted actions shape: {actions.shape} (Expected: [2, 10, 6])")
    assert actions.shape == (2, 10, 6), "Inference action shape mismatch!"

    print("5. Testing forward (Training Mode)...")
    # For forward we need to be in train mode for VAE to be used properly.
    policy.train()
    loss, loss_dict = policy(batch)
    print(f"  -> Loss: {loss.item():.4f}")
    print(f"  -> Loss dict: {loss_dict}")

    print("✅ All tests passed successfully!")

if __name__ == "__main__":
    test_forward()
