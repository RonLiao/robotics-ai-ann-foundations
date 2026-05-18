# scripts/train_act_lc.py
import sys
import os
import shutil
import runpy
import datetime

# 將外層目錄加入 PATH 確保能 import policies 與 act_lc_dataset
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# ==========================================
# 1. 建立 Log 儲存功能 (模擬 bash tee 指令)
# ==========================================
record_dir = os.path.join(root_dir, "record")
os.makedirs(record_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(record_dir, f"act_lc_train_{timestamp}.log")

class TeeLogger:
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, message):
        self.original_stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return hasattr(self.original_stream, 'isatty') and self.original_stream.isatty()

log_file = open(log_file_path, "a", encoding="utf-8")
sys.stdout = TeeLogger(sys.stdout, log_file)
sys.stderr = TeeLogger(sys.stderr, log_file)

print(f"📄 訓練 Log 將同步儲存至: {log_file_path}")
# ==========================================

# 1. 替換 LeRobot 原生的 Dataset
import lerobot.datasets.lerobot_dataset
from act_lc_dataset import ACTLCDataset

# Monkey-patch (動態替換) LeRobot 原生的 Dataset
lerobot.datasets.lerobot_dataset.LeRobotDataset = ACTLCDataset
print("✅ 成功應用 Monkey-patch: ACTLCDataset 已替換 LeRobotDataset")

# 2. 替換 LeRobot 原生的 ACT 模型與配置
try:
    try:
        # 新版 LeRobot 架構 (移除 common)
        import lerobot.policies.factory as policy_factory
        import lerobot.policies.act.modeling_act as act_modeling
    except ImportError:
        # 舊版 LeRobot 架構
        import lerobot.common.policies.factory as policy_factory
        import lerobot.common.policies.act.modeling_act as act_modeling

    from policies.act_lc.modeling_act import ACTPolicy as CustomACTPolicy
    
    # 針對 factory 與 modeling_act 中的類別進行抽取替換
    policy_factory.ACTPolicy = CustomACTPolicy
    act_modeling.ACTPolicy = CustomACTPolicy
    print("✅ 成功應用 Monkey-patch: CustomACTPolicy (Language-Conditioned) 已替換原生 ACTPolicy")

    # 同時替換 ACTConfig，確保 language_model_name 等欄位隨訓練 config 一起載入
    try:
        try:
            import lerobot.policies.act.configuration_act as act_config_module
        except ImportError:
            import lerobot.common.policies.act.configuration_act as act_config_module
        from policies.act_lc.configuration_act import ACTConfig as CustomACTConfig
        act_config_module.ACTConfig = CustomACTConfig
        print("✅ 成功應用 Monkey-patch: CustomACTConfig (Language-Conditioned) 已替換原生 ACTConfig")
    except Exception as _e:
        print(f"⚠️ 替換 ACTConfig 時發生錯誤: {_e}")
except Exception as e:
    print(f"⚠️ 替換 ACTPolicy 時發生錯誤，請確認路徑或導入是否正確: {e}")

# 3. 找到 lerobot-train 指令的實際執行路徑
lerobot_train_path = shutil.which("lerobot-train")
if not lerobot_train_path:
    print("❌ 找不到 lerobot-train 指令，請確認 lerobot 已正確安裝（或在 Docker 容器內可用）。")
    sys.exit(1)

# 4. 調整 sys.argv 以符合 lerobot-train 的需求
# a) 移除腳本本身路徑
# b) 攔截過濾 --policy.path，因為這是我們自定義的程式碼路徑而非 pretrained 權重庫，如果共用會觸發 ArgumentError
filtered_args = []
for arg in sys.argv[1:]:
    if arg.startswith("--policy.path="):
        print(f"🔧 攔截參數: 忽略 {arg} (因為我們已透過 Monkey-patch 將自定義 Policy 送入核心)")
        continue
    filtered_args.append(arg)

sys.argv = [lerobot_train_path] + filtered_args

print(f"🚀 將準備啟動: {lerobot_train_path}")

# 5. 透過 runpy 在同一個 process 中執行 lerobot-train
runpy.run_path(lerobot_train_path, run_name="__main__")
