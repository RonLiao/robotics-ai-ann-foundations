# 03 - LeRobot 框架深度解析與工具鏈 (Framework Anatomy)

這篇筆記旨在從架構層面理解 LeRobot 框架如何整合資料集、模型與監控工具，並記錄這些工具在具身智能（Embodied AI）工作流中的具體角色。

本章節以 [02-practice-circle-magnet](docs/02-practice-circle-magnet.md) 的 **Dataset/Model/Monitoring** 為例，展示 LeRobot 框架如何整合資料集、模型與監控工具，並記錄這些工具在具身智能（Embodied AI）工作流中的具體角色。
  - [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/RonLiao/lerobot-so101-elevator-dataset)
  - [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/RonLiao/so101-elevator-act)
  - [![WandB Monitoring](https://img.shields.io/badge/Weights%20%26%20Biases-Monitoring-FF4654?logo=weightsandbiases)](https://wandb.ai/ron-liao-nuwa-robotics/lerobot-so101-elevator)

---

## 第一章：Hugging Face 工具介紹 (Hugging Face Tools)

Hugging Face 在專案中扮演了資料與權重的託管倉庫，以下是其核心功能解析。

### 1.1 Datasets & Models
- **Datasets**：存放「教材（訓練數據）」。由於影像數據巨大，HF 提供了版本管理（Revision）與高效分流機制。
- **Models**：存放「成果（訓練權重）」。訓練出的 `model.safetensors` 是 AI 學習後的智慧結晶。

### 1.2 Dataset Viewer 介面指南
這是在 Hugging Face 網頁上即時預覽數據的強大工具：
- **長條圖 (Histograms)**：位於表格標題下方，代表該欄位數據的 **統計分佈 (Distribution)**。
    - 例如 `episode_index` 的分佈顯示每段錄製的長度；`timestamp` 的分佈可確認採樣是否均勻。
- **"Use this dataset" 按鈕**：提供一鍵複製的程式碼片段，方便在其他環境載入數據。
- **"Edit dataset card" 按鈕**：編輯該儲存庫的說明文件 (README)。這是向外界展示實驗環境、硬體型號與經驗的重要窗口。

---

## 第二章：Weights & Biases (訓練監控工具)

- **用途**：即時監控 Loss 下降曲線、GPU 資源佔用以及學習率變化。
- **意義**：判斷模型是否收斂，作為調整超參數（Batch Size, LR）的科學依據。

---

## 第三章：LeRobot 框架深度解析 (LeRobot Framework Analysis)

LeRobot 採用模組化設計，整體可用**兩個維度**交叉理解：**六個模組（空間）**與**三個工作流程（時間）**。

### 3.1 框架骨架：六模組 × 三工作流程

| 模組 | 錄製 | 訓練 | 推論 |
|---|---|---|---|
| `scripts`（命令行工具）| `lerobot-record` | `lerobot-train` | 自定義推論腳本 |
| `policies`（控制策略）| 不參與 | `forward()` 計算 Loss | `select_action()` 預測動作 |
| `datasets`（資料處理）| 寫入感測器資料 | 讀出 batch 供訓練 | 不參與 |
| `robot_devices`（硬體介面）| 讀取硬體感測器 | 不參與 | 讀取感測器 + 下發馬達指令 |
| `config`（配置系統）| 宣告錄製參數 | 宣告訓練超參數 | 宣告推論參數 |
| `envs`（仿真環境）| 同robot_devices| 同robot_devices | 同robot_devices|

> **envs的介面和robot_devices完全相同，但提供了與模擬環境的互動，因此可以用於模擬訓練。**

---

### 3.2 六個核心模組職責

#### 3.2.1 scripts — 發號施令層

`scripts/` 下的命令行工具是整個工作流的入口，負責串接其他所有模組。使用者直接互動的就是這一層（`lerobot-record`、`lerobot-train`、`lerobot-teleoperate`）。自定義的訓練腳本（如本專案的 `train_act_lc.py`）和推論腳本（`inference_language_act.py`）也屬於這一層的延伸。

##### ▎以 ACT + SO-101 移植為例

scripts 層本身的訓練 / 錄製指令不需加以改寫，因為 `lerobot-record` 和 `lerobot-train` 是框架提供的通用工具，只透過兩個機制與其他模組溝通，對任何算法和硬體都通用：

1. **Hydra Config + Registry**：`--policy.type=act` 讓 script 從 Policy Registry 找到 `ACTPolicy`；`--robot.type=so101_follower` 讓 script 從 Robot Registry 找到 `SO101Robot`。不認識的 type 字串就報錯，認識的直接實例化。
2. **標準化介面**：script 只呼叫介面方法，不知道底層是哪個實作
   ```python
   # record.py internal logic (simplified)
   robot = make_robot(cfg.robot)           # looks up SO101Robot from registry
   policy = make_policy(cfg.policy)        # looks up ACTPolicy from registry

   obs = robot.capture_observation()       # calls SO101Robot.capture_observation()
   action = policy.select_action(obs)      # calls ACTPolicy.select_action()
   robot.send_action(action)               # calls SO101Robot.send_action()
   ```

**移植的全部工作在 scripts 層之外**——在 policies 層新增 `ACTPolicy` 並登記到 registry、在 robot_devices 層新增 `SO101Robot` 並登記到 registry，scripts 完全不需要動。`record.py` 更支援雙模式：有 `--teleop` 參數時為錄製模式；有 `--policy` 參數時切換為推論模式，無需另外的推論腳本。

> 📂 **實際原始碼位置（LeRobot GitHub）**
> - `lerobot/scripts/train.py` — `lerobot-train` 入口
> - `lerobot/scripts/record.py` — `lerobot-record` 入口（錄製 + 推論 雙模式）
> - `lerobot/scripts/eval.py` — `lerobot-eval` 入口（**模擬環境**用，非實體機器人）

##### ▎ACT-LC 追加調整

原生 `lerobot-train` 無法支援語言參數做為模型的新輸入條件，因此追加兩個客製化腳本：

- **`train_act_lc.py`**：透過 Monkey-patch 讓原生 `lerobot-train` 在不知情的情況下使用 LC 版本的 Policy 和 Dataset
  ```python
  # train_act_lc.py: patch before calling lerobot-train
  import lerobot.datasets.lerobot_dataset as ds_module
  import lerobot.policies.act.modeling_act as policy_module

  ds_module.LeRobotDataset = ACTLCDataset     # replace dataset class in registry
  policy_module.ACTPolicy   = CustomACTPolicy  # replace policy class in registry

  runpy.run_path(lerobot_train_path)           # lerobot-train runs with patched classes
  ```
  `lerobot-train` 啟動時從模組讀取 class reference，在此之前已被換掉，所以它實際上用的是 LC 版本，但自己完全不知情。

- **`inference_language_act.py`**：`lerobot-record` 的推論模式在啟動前就綁定了固定的文字指令，無法在執行中動態切換。此腳本提供互動式命令列介面，每次任務完成後可重新輸入新指令。另外加入「靜止偵測」取代手動 Ctrl+C 停止：
  ```python
  # inference_language_act.py: interactive inference loop
  while True:
      cmd = input("⌨️ 請輸入指令 (或 'q' 退出): ")  # dynamic text input per task

      for step in range(max_steps):
          obs = robot.get_observation()
          obs["language_instruction"] = [cmd]          # inject instruction each step
          action = policy.select_action(obs)
          robot.send_action(action)

          # auto-stop: if action displacement < threshold for N consecutive steps
          if torch.norm(action - prev_action) < stop_threshold:
              static_steps += 1
              if static_steps >= stop_patience:
                  break                                # task complete, back to prompt
  ```

#### 3.2.2 policies — 算法核心層

LeRobot 對算法的設計採用**父類別繼承**的方式：

- 所有算法都必須繼承 `PreTrainedPolicy` 並實作兩個核心介面：
  - `forward(batch)`：訓練時計算 Loss
  - `select_action(batch)`：推論時回傳下一步動作
- 每個 Policy 都有對應的 **Config 類別**，用於定義超參數（如 chunk size、hidden dim）
- 移植新算法進 LeRobot，就是繼承 `PreTrainedPolicy` 實作這兩個方法

##### ▎以 ACT 移植為例

需要建立一個符合 LeRobot 介面規範的 Policy 類別，包含兩個檔案：

1. **`configuration_act.py`**：定義超參數 Config
   ```python
   # configuration_act.py: define ACT hyperparameters inheriting PreTrainedConfig
   @dataclass
   class ACTConfig(PreTrainedConfig):
       chunk_size: int = 50           # number of action steps to predict
       hidden_dim: int = 512          # transformer hidden dimension
       n_heads: int = 8               # attention heads
       n_encoder_layers: int = 4      # CVAE encoder depth
       n_decoder_layers: int = 7      # CVAE decoder depth
       # input/output shapes are auto-filled from dataset features at training time
   ```

2. **`modeling_act.py`**：實作 Policy 主體
   ```python
   # modeling_act.py: implement ACTPolicy inheriting PreTrainedPolicy
   class ACTPolicy(PreTrainedPolicy):
       def forward(self, batch):
           # training: run CVAE encoder to get z, run decoder, compute L1 + KL loss
           images = {k: v for k, v in batch.items() if k.startswith("observation.images")}
           state  = batch["observation.state"]   # shape (B, 6) for SO-101
           action = batch["action"]              # shape (B, 50, 6), future 50 steps
           # ... CVAE encode(state + action) -> z -> decode(images + state + z) -> loss

       def select_action(self, batch):
           # inference: skip CVAE encoder, set z=0, run decoder only
           z = torch.zeros(...)                  # no leader arm needed at inference
           # ... decode(images + state + z) -> predicted action chunk
   ```

> 📂 **實際原始碼位置（LeRobot GitHub）**
> - `lerobot/common/policies/act/configuration_act.py` — ACTConfig 定義
> - `lerobot/common/policies/act/modeling_act.py` — ACTPolicy 實作
> - `lerobot/common/policies/__init__.py` — Policy 類型的 registry 對應表

##### ▎ACT-LC 追加調整

由於多了一個語言輸入，我們必須對ACT Policy進行修改。但**如 [04-practice-6-button-panel.md 第三步的 Monkey-patch 設計](04-practice-6-button-panel.md#第三步啟動條件式訓練-language-conditioned-training)** 所述，為了相容於LeRobot 之後的版本，我們不直接修改原生的 `modeling_act.py` **。取而代之的是在本專案目錄內新建一份 `policies/act_lc/modeling_act.py`（複製自 LeRobot 原版後修改），類別名稱維持 `ACTPolicy` 以便 Monkey-patch 無縫替換：

```python
# policies/act_lc/modeling_act.py (project-level copy, NOT LeRobot's original)
class ACTPolicy(PreTrainedPolicy):    # same class name for transparent monkey-patch
    def __init__(self, config):
        super().__init__(config)                                               # [original]
        if hasattr(config, 'language_model_name'):
            self.tokenizer = AutoTokenizer.from_pretrained(...)                # [NEW] load text tokenizer
        self.model = ACT(config)                                               # [original] ACT.__init__ also extended

# ACT.__init__() additions (inside ACT class, same file):
if hasattr(config, 'language_model_name'):                                     # [NEW]
    self.text_encoder = AutoModel.from_pretrained(config.language_model_name)  # [NEW] language backbone
    for param in self.text_encoder.parameters():
        param.requires_grad = False                                            # [NEW] freeze weights
    self.text_proj = nn.Linear(config.language_dim, config.dim_model)          # [NEW] project 768 -> dim_model
    self.encoder_text_feat_pos_embed = nn.Embedding(                           # [NEW] text positional embedding
        config.max_text_length, config.dim_model)

# ACT.forward() additions: inject text tokens into encoder input sequence
if hasattr(config, 'language_model_name') and "text_inputs" in batch:          # [NEW]
    text_tokens = self.text_proj(                                              # [NEW]
        self.text_encoder(**batch["text_inputs"]).last_hidden_state)
    encoder_in_tokens.extend(list(text_tokens.transpose(0, 1)))                # [NEW] append text tokens
```

> 📂 **實際原始碼位置（本專案）**
> - `policies/act_lc/modeling_act.py` — ACT-LC Policy 主體（含 Text Encoder 擴充）
> - `policies/act_lc/configuration_act.py` — ACTConfig（含語言相關欄位）
> - `scripts/train_act_lc.py` — Monkey-patch 入口，將上述類別替換進 LeRobot 訓練引擎


#### 3.2.3 datasets — 資料契約層

Dataset 層是 Policy 與硬體之間的橋樑，負責定義「資料長什麼樣子」。

**`LeRobotDataset` 是框架的通用讀取器**，不預先知道有哪些欄位，而是在初始化時讀取 `meta/info.json` 的 `features` 動態得知 Schema，再依此讀取 Parquet 和 MP4，最後以 `dict` 輸出給 Policy。

Dataset 層的通用介面：
- **`features`**：欄位定義。**由錄製時宣告的感測器自動生成**，非手動定義。`action`、`observation.state`、`observation.images.<name>` 等 key 由感測器種類決定
- **`delta_timestamps`**：時間軸擴充。告知 `__getitem__` 要往過去或未來多取幾幀，不同算法可設定不同的時間窗口
- **資料儲存格式**：數值型資料（關節角度、task_index）存 **Parquet**；影像存 **MP4**，讀取時依 timestamp 逐幀解碼

**Dataset 層的可客製化之處**

Dataset 層有三個客製化點：

1. **`lerobot-record` 參數設定**：這是最隱含、但影響最大的客製化。Policy 的 Config（如 `ACTConfig`）定義了它期望讀到的 feature key（例如 `observation.images.front`、`observation.state`），`lerobot-record` 的參數必須讓 `features` 的 key 與 shape **完全符合**，否則訓練時 Policy 讀不到對應欄位就會失敗。

   > LeRobot 官方的每個 Policy 網頁（如 [ACT 的 Policy 文件](https://huggingface.co/docs/lerobot/act)）都會列出建議的 `lerobot-record` 參數範例，實質上就是在告知「錄製時應宣告哪些感測器才能配合這個 Policy」。

2. **`delta_timestamps`**：Policy 向 Dataset 宣告「我需要哪些時間點的資料」的機制。不同算法的時間需求不同——ACT 需要未來 50 步的動作序列（Action Chunking）；純視覺反應型算法可能只需要當下這一幀。這個 dict 由 Policy 定義，在訓練初始化時傳入 `LeRobotDataset`，控制 `__getitem__` 的時間軸擴充行為。

3. **`__getitem__` Wrapper**：若需要在 `__getitem__` 輸出的 dict 上注入額外欄位（原始感測器資料中不存在的欄位，如語言指令），需用 **Wrapper 包裝**——繼承 PyTorch `Dataset`，在內部持有 `LeRobotDataset` 實例，攔截 `__getitem__` 的回傳值並手動注入。

##### ▎以 ACT + SO-101 移植為例

錄製時傳入的感測器參數會自動決定 `features` 的欄位結構：

```json
// meta/info.json features（由錄製 script 自動生成）
"features": {
    "action":                   {"dtype": "float32", "shape": [6]}, // 手臂有6個馬達
    "observation.state":        {"dtype": "float32", "shape": [6]},
    "observation.images.front": {"dtype": "video",   "shape": [480, 640, 3]},
    "task_index":               {"dtype": "int64",   "shape": [1]}
}
```

- **`action`**：Leader ARM 的 6 個馬達位置，這是 ACT 算法學習的目標軌跡
- **`observation.state`**：Follower ARM 當下的 6 個關節角度
- **`observation.images.front`**：`front` 是來自 `--robot.cameras="{front: ...}"` 的鍵名
- **`task_index`**：多任務錄製時區分任務，對應 `meta/tasks.parquet` 的文字指令映射表

ACT 的 `delta_timestamps` 設定（向未來取 50 步的 action chunk）：

```python
# ACT delta_timestamps: fetch current state/image + future 50-step action sequence
delta_timestamps = {
    "observation.state":        [0.0],
    "observation.images.front": [0.0],
    "action": [i * (1/30) for i in range(50)]  # future 50 steps at 30fps
}
```

**移植的全部工作在 Dataset 層之外**——只要錄製時指定好感測器，`LeRobotDataset` 就能自動讀出正確的 Schema，ACT + SO-101 完全不需要修改這層任何程式碼。

> 📂 **實際原始碼位置（LeRobot GitHub）**
> - `lerobot/common/datasets/lerobot_dataset.py` — LeRobotDataset 通用讀取器

##### ▎ACT-LC 追加調整

原生 `LeRobotDataset` 只輸出數值欄位與影像，沒有文字指令欄位。ACT-LC 需要在每筆資料上附加 `language_instruction`，因此以 Wrapper 包裝：

```python
# scripts/act_lc_dataset.py: Wrapper injecting language_instruction at __getitem__
class ACTLCDataset(Dataset):         # inherits PyTorch Dataset, NOT LeRobotDataset
    def __init__(self, repo_id):
        self.dataset = LeRobotDataset(repo_id)   # wrap internally

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # translate task_index -> instruction string from tasks.parquet
        task_str = self.dataset.meta.tasks[item["task_index"].item()]  # [original]
        item["language_instruction"] = task_str                        # [NEW] inject text
        return item
```

> 📂 **實際原始碼位置（本專案）**
> - `scripts/act_lc_dataset.py` — ACTLCDataset Wrapper 實作

#### 3.2.4 robot_devices — 硬體抽象層

負責與**所有實體周邊裝置**溝通，統一抽象為標準介面供上層（scripts / Dataset）使用。目錄階層如下：

```
robot_devices 層
├── robots/         ← 手臂驅動（繼承 Robot 介面）
│   ├── so101_follower.py  → SO101Follower（執行端，接收指令、輸出關節角度）
│   ├── so101_leader.py    → SO101Leader（操縱端，teleoperation 用）
│   └── utils.py           → make_robot_from_config() 工廠函數 + robot type Registry
└── cameras/        ← 相機驅動（繼承 Camera 介面，與 robots/ 平行）
    ├── opencv/     → OpenCVCamera（USB 一般相機）
    └── realsense/  → RealSenseCamera（深度相機）
```

**注意**：`teleop` 不是裝置型別，而是**操作模式**。`lerobot-teleoperate` 這個 script 同時持有一個 `SO101Leader` 和一個 `SO101Follower`，讓 Leader ARM 的位置即時鏡射到 Follower ARM。兩者都屬於 `robots/` 下的硬體驅動。

`lerobot-record` 中的 `--robot.type=so101_follower` 和 `--robot.cameras=...` 參數，分別透過 `robots/` 和 `cameras/` 的 Registry 查找對應驅動並初始化。

**對本專案而言，這層不需要自行實作**：SO-101 的硬體驅動已內建於 LeRobot。`--robot.type=so101_follower` 這個參數就是啟動對應驅動的開關。只有在使用 LeRobot 完全不支援的全新硬體時，才需要在此層新增驅動程式，下面的章節只是用於說明。

##### ▎以 SO-101 移植為例

需要在 `robots/` 下分別為兩種角色實作，各自繼承 `Robot` 父類別：

**Follower（執行端）**：接收 Policy 或 teleoperate 傳來的指令、回報自身狀態

```python
# robots/so101_follower.py: implement SO-101 Follower ARM driver
class SO101Follower(Robot):
    def connect(self):
        # open serial port to Follower ARM (/dev/ttyACM1)
        self.port = serial.Serial("/dev/ttyACM1", baudrate=1000000)

    def capture_observation(self) -> dict:
        # read 6 motor positions from Follower ARM at 30Hz
        angles = self._read_motor_positions()   # returns list of 6 floats
        return {
            "observation.state": torch.tensor(angles) # cameras are handled separately by the cameras/ layer
        }

    def send_action(self, action: torch.Tensor):
        # write 6 motor target positions to Follower ARM
        self._write_motor_positions(action.tolist())
```

**Leader（操縱端）**：讀取人手施加的關節角度，以「動作」的語意輸出

```python
# robots/so101_leader.py
class SO101Leader(Robot):
    def connect(self):
        self.port = serial.Serial("/dev/ttyACM0", baudrate=1000000)

    def get_action(self) -> dict:
        # read human-guided joint angles → this IS the action (not an observation)
        angles = self._read_motor_positions()
        return {"action": torch.tensor(angles)}
    # no send_action() needed: leader only outputs, never receives commands
```

`lerobot-teleoperate` 串接兩者的邏輯：

```python
# teleoperate script internal logic (simplified)
action = leader.get_action()         # human moves leader → becomes action
follower.send_action(action)         # mirror to follower arm

obs = follower.capture_observation() # record follower state (for dataset)
```

接著需要在 `robots/utils.py` 的 Robot type Registry 中分別登記兩個 type：

```python
# robots/utils.py: register both types in Robot Registry
ROBOT_CLASSES = {
    "so101_follower": SO101Follower,  # --robot.type=so101_follower
    "so101_leader":   SO101Leader,    # used by lerobot-teleoperate internally
}
```

**Follower vs Leader API 對照**

| 方法 | `SO101Follower` | `SO101Leader` |
|---|---|---|
| `connect()` | 開 `/dev/ttyACM1` | 開 `/dev/ttyACM0` |
| `capture_observation()` | ✅ 讀自身狀態 → Policy 輸入 | ❌ 無此方法 |
| `get_action()` | ❌ 無此方法 | ✅ 讀人手位置 → 作為動作輸出 |
| `send_action()` | ✅ 驅動馬達 | ❌ Leader 從不接收指令 |

> 📂 **實際原始碼位置（LeRobot GitHub）**
> - `lerobot/robots/so101_follower.py` — SO101Follower 執行端驅動
> - `lerobot/robots/so101_leader.py` — SO101Leader 操縱端驅動
> - `lerobot/robots/utils.py` — `make_robot_from_config()` 工廠函數與 robot type Registry
> - `lerobot/cameras/opencv/` — OpenCV 相機驅動

##### ▎ACT-LC 追加調整

新增手眼相機 `wrist`，只需在 `--robot.cameras` 加入新鍵名，Dataset 的 `observation.images.wrist` 欄位即自動產生，不需修改此層任何程式碼。

#### 3.2.5 config — 配置系統

LeRobot 使用 Python dataclass 搭配 Hydra/OmegaConf 管理所有模組的超參數。每個 Policy 都有配對的 Config 類別（如 `ACTConfig`），訓練時框架從 Dataset 的 `features` 自動讀取動作維度並填入 Config，因此同一個 Policy 類別可以不修改程式碼地適用於不同關節數的手臂。

##### ▎以 ACT + SO-101 移植為例

Policy Config 已在 3.2.2 中實作。訓練時框架自動從 Dataset `features` 讀取並填入 `ACTConfig` 的輸入輸出形狀：

```python
# training engine auto-fills these from dataset features (no manual config)
# output_shapes["action"]           <- features["action"]["shape"]   = [6]
# input_shapes["observation.state"] <- features["observation.state"]["shape"] = [6]
# camera resolutions                <- all observation.images.* shapes
```

若換成七軸手臂重錄資料，`ACTConfig` 完全不需要修改，框架自動調整輸出頭維度。

> 📂 **實際原始碼位置（LeRobot GitHub）**
> - `lerobot/common/policies/act/configuration_act.py` — ACTConfig（同 policies 章節）
> - `lerobot/common/policies/act/modeling_act.py` — 讀取 config 並建立對應維度的 Linear 層

##### ▎ACT-LC 追加調整

在 `policies/act_lc/configuration_act.py` 中新增語言模型相關欄位：

```python
# configuration_act.py: newly added fields for language conditioning
language_model_name: str = "distilbert-base-uncased"  # text encoder
language_dim: int = 768            # DistilBERT hidden size
max_text_length: int = 16          # tokenizer max length
```

#### 3.2.6 envs — 仿真環境層（實體機器人路線可跳過）

提供基於 Gym 介面的模擬環境，讓 Policy 在不連接實體硬體的情況下進行訓練與評估。本專案完全採用實體機器人路線，此模組在整個工作流中從未被使用。

---

### 3.3 三個工作流程中的模組協作

**錄製（Data Collection）**

```
lerobot-record 指令（scripts）
    → robot_devices 讀取 Leader ARM 角度 → 即時傳給 Follower ARM
    → robot_devices 讀取 Follower ARM 角度 + 相機影像
    → datasets 將資料依 features 命名規範寫入 parquet / mp4
    → meta/info.json 自動更新 features、統計量
```

**訓練（Training）**

```
lerobot-train（scripts）
    → datasets.__getitem__() 讀取 batch（依 delta_timestamps 取多時間步）
    → DataLoader 打包成 batch dict
    → policies.forward(batch) 計算 Loss
    → 反向傳播更新 Policy 權重
    → WandB 記錄 Loss 曲線（監控工具）
```

**推論（Inference）**

```
自定義推論腳本（scripts）
    → robot_devices 讀取 Follower ARM 當前角度 + 相機影像
    → 組裝 observation dict
    → policies.select_action(observation) 預測下一步動作
    → robot_devices 將動作下發給 Follower ARM 馬達執行
```

Policy 與 Dataset 之間的唯一接點是 **batch dict 的 key 名稱**。只要 Dataset 吐出的欄位名稱與 Policy 的 `forward()` 預期讀取的 key 一致，兩層就能無縫銜接，而不需要強型別的繼承關係約束。

---
