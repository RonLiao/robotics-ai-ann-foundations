# 04 - 階段二實作：語意驅動的特定面板按壓 (6 顆按鈕)

在進入適應所有電梯面板的通用任務前，此中繼階段的目標是：**針對特定款式（6 顆按鈕，按下會亮藍光）的面板，讓模型能「聽懂指令」，由輸入的文字字串決定按哪個按鈕。**

環境樣式參考：
![6顆按鈕電梯面板](assets/6-button-panel-environment.jpg)

## 任務目標

1. **指令驅動動作**：使用者無論是藉由口頭（用語音轉文字）還是文字輸入，手臂都能根據傳入的動態字串參數決定按下對應的目標按鈕。
2. **區辨能力驗證**：確認單一系統架構能否準確解析相近按鈕影像與文字指令的對應關係，不誤觸相鄰的其它選項。
3. **特徵驗證**：觀察按下後發出的「藍光」是否能作為動作完成的視覺回饋特徵。

---

## 💡 理論探討：指令驅動按壓的三層級解決方案

為實現「由輸入字串決定發動哪個按壓行為」，在目前具身智能（Embodied AI）的發展路徑上，有三個截然不同的工程與演算法層次可以探索：

### 層次一：工程中控台路由 (Wrapper 與權重動態切換)
本質上仍是傳統的「一個按鈕，訓練一個專屬模型」。首先由外部程式（如 Python 腳本、語法解析器或輕量級 LLM Router）攔截使用者的語音或字串，解析出目標按鈕為何（例如：`target=Btn3`）後，程式再去動態讀取並載入 `btn3` 的權重（pretrained_model）來執行。
- **特性**：實作簡單直觀，但欠缺泛化性。未來若有百個按鈕，就需要維護百個龐大的模型庫，且切換權重時會有將模型掛載至 VRAM 的載入延遲。

### 層次二：條件式輸入的多任務模型 (Language-Conditioned ACT)
**【⭐ 本階段 (04 筆記) 先行採用的方案】**
不再將每一顆按鈕視作獨立外掛，而是將 6 個按鈕的所有示範軌跡整合，訓練這「唯一一部」了解所有情況的 **Multi-Task ACT 模型**。
透過將代表任務的「文字指令」轉換為特徵（Language Embeddings），當作一個額外的條件 (Condition) 一併餵給類神經網路。模型在推論時會根據文字條件，自動將視覺注意力 (Attention) 對齊畫面中對應的號碼。只要傳入 `"Press button 3"` 的字串，這個單一模型就能動態產生前往按鈕 3 的動作軌跡！

### 層次三：導入視覺-語言-動作大模型 (VLA, Vision-Language-Action)
**【🚀 未來規劃：將於後續的 `05_xxx` 階段實作】**
免去手動萃取特徵與繁瑣的模型訓練設定，直接導入端到端的大型多模態基礎模型（例如 OpenVLA）。VLA 模型就像是擁有 ChatGPT 般的強大常識，不只能同時看圖與讀懂文字指令，產出的也不是文字語言，而是能直接驅動馬達的 **Action Tokens**！即使是模型以前完全沒見過的按鈕面板，單憑其強大的預訓練語言常識也有極高的機率達成 Zero-shot 的零樣本按壓。

---

## 具體實作與原始碼修改步驟 (層次二架構)

要讓原生的 ACT 模型學會「聽懂人話」，必須在 LeRobot 中進行架構級別的改裝（Language-Conditioned ACT Mod）。以下為核心的實作步驟：

### 第0步：了解 lerobot 的 ACT 實作
在動手修改架構之前，先深入理解 LeRobot 預設的原生 ACT 模型實作（原始碼與資料流），確認 Transformer Encoder/Decoder 的實際運作方式，這部分的探討與筆記對應至理論學習的 ACT 分析中，程式碼註解已加入至 [`policies/act/modeling_act.py`](../policies/act/modeling_act.py)。

### 第一步：修改 ACT 網路架構 (注入 Text Embeddings)

為保留原始實作，我們已將 `policies/act` 目錄複製一份為 `policies/act_lc`(Language-Conditioned ACT)。接下來的修改都將在 `act_lc` 目錄中進行。

原生 ACT 實作(位於 `lerobot/common/policies/act/modeling_act.py`) 僅接收影像與本體狀態。以下為架構改造的核心：

**📍 架構變更圖解 (ACT-LC 資料流)：**
將提取出的語意特徵注入到 Transformer Encoder 的輸入端，強制模型進行跨模態注意力計算。

![ACT-LC Architecture](assets/ACT_LC_Architecture.png)

具體必須做以下改造：

1. **引入文字編碼器 (Text Encoder)**：
    - 在 ACT Policy 的 `__init__` 中，載入輕量且推理快速的預訓練語言模型（採用 `distilbert-base-uncased`），並將其權重凍結 (`requires_grad=False`) 以減少機器手臂訓練初期的計算資源負擔與過度擬合問題。
2. **新增特徵對齊層 (Projection MLP)**：
    - 文字編碼器輸出的隱藏維度（DistilBERT 為 768）與 ACT 內部 Transformer 的維度（預設為 512）不同。需加入一層 Linear 進行特徵降維與映射：
    ```python
    self.text_proj = nn.Linear(config.language_dim, config.hidden_dim)
    ```
3. **改造 Transformer Encoder 流水線**：
    - 原生的 `forward()` 與 `compute_loss()` 需修改為可以接收 `batch` 中的 `language_instruction` 的批次字串，並在內部呼叫 Tokenizer 將字串轉為 `input_ids` 與 `attention_mask`。送入 Text Encoder 與 `text_proj` 後，得到 `text_tokens`（維度為 `(Batch, Seq_Len, 512)`）。
    - 接著，進入 Transformer (`self.model.encoder`) 之前，將 `text_tokens` 加入至序列中（與 `is_pad`、`z`、`proprioception`、`image_tokens` 串接），這樣 Encoder 的 Self-Attention 機制就會強制讓「語意 Token」與「影像空間 Token」產生跨模態的權重聯結。
   這樣 Encoder 的 Self-Attention 機制就會強制讓「語意 Token」與「影像空間 Token」產生跨模態的權重聯結。
4. **驗證**：
    - 在不開啟正式訓練迴圈的情況下，撰寫一小段 Python 測試腳本，初始化改版後的 ACTPolicy 並模擬輸入 batch (包含一組假影像、假 proprioception 與文字指令 ["press button 3"])。
    - 確認 forward() 能夠順利產出維度正確的 action_tokens 而不拋出維度不匹配或是 Cuda Memory 錯誤。


### 第二步：多任務資料集混合 (Multi-task Data Collection)

1. **指令標註**：
   在錄製這 6 顆按鈕時，將所有軌跡錄進 **同一個 Repo**，但透過給定特定的 `--dataset.single_task` 來給予模型訓練時的解題線索：
   ```bash
   lerobot-record ... \
     --dataset.repo_id=RonLiao/lerobot-so101-elevator-6btn-multitask \
     --dataset.single_task="press button 3" # 錄製按鈕 3 的回合
   ```

2. **修改 Dataset 解析器**：
   因為原生的 `LeRobotDataset` 預設只回傳數值型與影像特徵，為了讓改版後的 ACT-LC 模型能順利讀取到文字指令，必須撰寫一個 Wrapper 類別（實作了 `ACTLCDataset`）。
   - **實作邏輯**：在 `__getitem__` 取資料時，攔截當下回傳的字典，並從 `item["task_index"]` 查詢資料集自帶的 `meta/info.json`。提取出對應的 task string 後，手動將其注入到 `item["language_instruction"]` 欄位中供 `DistilBERT` 編碼。

- **經驗：為何在多任務混合錄製時，需要給定特定的 `--dataset.single_task`？**
  - **核心考量**：正常邏輯下，錄製 Episode 前應給定一串指令（比如"press button 3"，並補入每一幀 (Frame) 的數據中。但此舉會導致同一 Episode 的每個 Frame 都存入完全重複的文字，造成嚴重的空間浪費。
  - **既有機制**：LeRobot 框架分為 Dataset/Episode/Frame 三層級。每個 Episode 附帶一個 `task_index`，並自動包含於該 Episode 下的所有 Frame 中。
  - **優化策略**：既然每個 Frame 原來就有 `task_index`，且指令字串也是以 Episode 為單位，故直接沿用此索引。訓練時透過 `act_lc_dataset.py` 將索引動態展開為“dataset.single_task"指定的對應文字指令，達成節省資料夾體積但訓練具備語意輸入的效果。
  - **參數價值**：`--dataset.single_task` 用於「為該次錄製的所有 Episode 下定義」，它會將此標籤存於 `meta/info.json` 中，符合上面提到的需求。此外也能避免每個 Episode 都需手動輸入字串，支援一次錄製多個 Episode。

- **經驗：推論時若輸入略有不同的字串（如 "3" 或 "button 3"）還能運作嗎？**
  - **初期現狀**：雖然 `DistilBERT` 具備語意理解能力（換句話說，這兩個字串經過DistilBERT編碼後，會得到非常接近的向量），但 ACT 模型目前訓練時僅見過極少數的特定 Embedding。若推論時輸入未曾對齊過動作的字串，即便語意接近，仍極可能失敗。此階段建議維持與錄製時 **完全一致** 的指令。
  - **泛化解決方案（文字擾動）**：後續在訓練階段可引入「文字擾動 (Text Augmentation)」，例如將標準指令隨機更換為多樣化的說法（如 "press 3"、"go to 3" 或 "button 3"）。
  - **實施建議**：此擾動不建議在錄製時手動為 50 個 Episode 分別輸入不同字串，而應在訓練程式讀取資料時，由 DataLoader 自動進行隨機替換，迫使模型學到「語意相似」即代表「動作相同」，顯著提升系統對自然語言輸入的容錯與泛化能力。


### 第三步：啟動條件式訓練 (Language-Conditioned Training)

此時不再分開訓練 6 次。一次將混合了上述所有按鈕的巨大 Dataset 餵給改裝後的 ACT。
- **訓練哲學**：模型在預測馬達位置時，會發現同樣是「六顆按鈕的畫面」，但最佳軌跡卻有 6 種解。這會迫使 Loss 函數推動網路依賴剛加入的 `text_tokens` 作為解題的先驗條件，進而學會「看字決定按哪顆按鈕」的區辨能力。

**具體實施步驟：**
1. **資料集準備**：確保已錄製足夠的多任務 Episode (建議每顆按鈕 50+ 個)，並上傳至 Hugging Face。
2. **訓練環境配置**：啟動 Docker 容器並掛載 `ron_so101_v2` 環境。
3. **執行訓練腳本 (`scripts/train_act_lc.py`)**：
   此腳本將整合 `ACTLCDataset` 與 `act_lc` policy，並調用 `lerobot` 的訓練循環。需指定以下關鍵參數：
   - `dataset.repo_id`: 您的多任務資料集 ID。
   - `policy.path`: 指向 `policies/act_lc` 的本地路徑。
   - `training.batch_size`: 根據顯存調整 (建議 16 或以上)。
4. **監控與驗證**：透過 WandB 觀察不同任務指令下，Loss 曲線是否穩定下降並收斂。

### 第四步：客製化推論腳本 (Inference Router)

推論時不能直接呼叫預設的 `lerobot-record`。需要自行編寫 Python 腳本（如 `inference_language_act.py`）：
1. 載入這顆唯一的 Multi-Task ACT Checkpoint。
2. 開啟一個迴圈接受 `input()` 字串（或是結合 Whisper 的語音辨識端點）。
3. 將接收到的字串實時編碼為 Vector，餵給模型。
4. 模型根據當前攝影機畫面，結合該文字 Vector，即時吐出針對特定按鈕的 Action Tokens 控制機械臂！
