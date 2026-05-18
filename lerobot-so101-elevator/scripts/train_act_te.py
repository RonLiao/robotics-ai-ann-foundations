# scripts/train_act_te.py
# ACT with Task Embedding (act_te) 訓練腳本
# 與 train_act_lc.py 的差異：Monkey-patch 目標改為 act_te 模組
import sys
import os
import shutil
import runpy
import datetime

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# ── Log 儲存 ─────────────────────────────────────────────────────────────────
record_dir = os.path.join(root_dir, "record")
os.makedirs(record_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(record_dir, f"act_te_train_{timestamp}.log")

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

# ── 1. 替換 Dataset ───────────────────────────────────────────────────────────
import lerobot.datasets.lerobot_dataset
from act_lc_dataset import ACTLCDataset
lerobot.datasets.lerobot_dataset.LeRobotDataset = ACTLCDataset
print("✅ Monkey-patch: ACTLCDataset 已替換 LeRobotDataset")

# ── 2. 替換 ACTPolicy 與 ACTConfig → act_te 版本 ──────────────────────────────
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
    print("✅ Monkey-patch: TEACTPolicy (Task Embedding) 已替換原生 ACTPolicy")

    try:
        try:
            import lerobot.policies.act.configuration_act as act_config_module
        except ImportError:
            import lerobot.common.policies.act.configuration_act as act_config_module
        from policies.act_te.configuration_act import ACTConfig as TEACTConfig
        act_config_module.ACTConfig = TEACTConfig
        print("✅ Monkey-patch: TEACTConfig (Task Embedding) 已替換原生 ACTConfig")
    except Exception as _e:
        print(f"⚠️ 替換 ACTConfig 時發生錯誤: {_e}")

    # 覆蓋 PreTrainedConfig registry：讓 draccus 解析 --policy.type="act" 時
    # 使用有 num_tasks 欄位的 TEACTConfig，而非 lerobot 內建的 vanilla ACTConfig
    try:
        from lerobot.configs.policies import PreTrainedConfig
        _patched = False
        for _attr in ['_registry', '__registry__', '_subclass_registry', '_choice_registry']:
            _reg = getattr(PreTrainedConfig, _attr, None)
            if isinstance(_reg, dict) and 'act' in _reg:
                _reg['act'] = TEACTConfig
                print(f"✅ Monkey-patch: PreTrainedConfig.{_attr}['act'] → TEACTConfig")
                _patched = True
                break
        if not _patched:
            # 印出所有含 'act' 的 dict 屬性供診斷
            for _attr in dir(PreTrainedConfig):
                _val = getattr(PreTrainedConfig, _attr, None)
                if isinstance(_val, dict) and 'act' in _val:
                    print(f"  ▷ 發現含 'act' 的屬性: {_attr} = {list(_val.keys())}")
            print("⚠️ 找不到 PreTrainedConfig registry，請確認上方診斷輸出")
    except Exception as _e:
        print(f"⚠️ 替換 PreTrainedConfig registry 時發生錯誤: {_e}")
except Exception as e:
    print(f"⚠️ 替換 ACTPolicy 時發生錯誤: {e}")

# ── 3. 啟動 lerobot-train ─────────────────────────────────────────────────────
lerobot_train_path = shutil.which("lerobot-train")
if not lerobot_train_path:
    print("❌ 找不到 lerobot-train 指令，請確認 lerobot 已正確安裝。")
    sys.exit(1)

filtered_args = []
for arg in sys.argv[1:]:
    if arg.startswith("--policy.path="):
        print(f"🔧 攔截並忽略: {arg}")
        continue
    filtered_args.append(arg)

sys.argv = [lerobot_train_path] + filtered_args
print(f"🚀 啟動: {lerobot_train_path}")
runpy.run_path(lerobot_train_path, run_name="__main__")
