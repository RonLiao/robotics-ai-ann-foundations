#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from policies.act_lc.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """
    # 這是對外的高層級介面，負責管理模型的生命週期與推理邏輯

    config_class = ACTConfig
    name = "act_lc"

    def __init__(
        self,
        config: ACTConfig,
    ):
        """
        初始化 ACT 模型, 並根據配置決定是否啟用「時序集成」(Temporal Ensembling)
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 初始化 Tokenizer（使用 getattr 確保原生 config 也能正常建立）
        from transformers import AutoTokenizer
        _lang_model = getattr(self.config, 'language_model_name', 'distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(_lang_model, clean_up_tokenization_spaces=True)

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        # 在機器人每個 Episode 開始前重置狀態（如清空動作隊列或整合器）
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.
        推理時的主函式, 負責從預測的動作塊(Chunk)中提取下一個要執行的動作

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        # 呼叫模型來預測一整段動作序列
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # 處理語言指令轉 Token
        if "language_instruction" in batch:
            device = batch[OBS_STATE].device if OBS_STATE in batch else next(self.parameters()).device
            text_inputs = self.tokenizer(
                batch["language_instruction"],
                padding="max_length",
                max_length=self.config.max_text_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            batch["text_inputs"] = text_inputs

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        # 訓練核心. 計算預測動作與真實動作的 L1 Loss, 若啟用 VAE 則加上 KLD Loss
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # 處理語言指令轉 Token
        if "language_instruction" in batch:
            device = batch[OBS_STATE].device if OBS_STATE in batch else next(self.parameters()).device
            text_inputs = self.tokenizer(
                batch["language_instruction"],
                padding="max_length",
                max_length=self.config.max_text_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            batch["text_inputs"] = text_inputs

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    """ 負責處理「動作平滑化」的工具類別
        當模型在每個時間點都預測未來 S 步動作時, 會產生大量重疊預測. 此類別利用指數加權平均數(Exponential Weights)
        將這些重疊動作融合, 減少機器人動作的抖動
    """

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.
    ACT 演算法的本體網路, 整合了所有子模組

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).

        # 繼承自 nn.Module，初始化基礎網路結構
        super().__init__()
        self.config = config

        # --- 1. VAE 編碼器部分 (僅在訓練且啟用 use_vae 時使用) ---
        if self.config.use_vae:
            # 初始化 BERT 風格的 Transformer 編碼器，用於將動作序列與機器人狀態編碼進潛在空間 (Latent Space)
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)

            # [cls] token 的 Embedding，其輸出的隱藏向量會被用來計算潛在分佈的均值與方差
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)

            # 如果配置中有機器人狀態特徵 (如關節角度)，建立一個線性層將其投影到模型維度 (dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )

            # 動作序列 (Action Sequence) 的投影層，將目標動作投影到模型維度
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )

            # VAE 輸出投影層：將 [cls] token 的輸出轉換為潛在分佈參數 (均值 mu + 對數方差 log_sigma_x2)
            # 輸出維度為 latent_dim * 2
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)

            # 為 VAE 編碼器的輸入 (cls + robot_state + action_chunk) 註冊固定的正弦位置編碼 (Sinusoidal Positional Embedding)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # --- 2. 視覺特徵提取 (Backbone) ---
        # Backbone for image feature extraction.
        if self.config.image_features:
            # 根據配置獲取 torchvision 中的模型 (由vision_backbone決定，通常是 resnet18)，並設定權重與 Batch Norm 模式
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )

            # 使用 IntermediateLayerGetter 提取 ResNet 的最後一層卷積特徵圖 (layer4)
            # 註：這裡假設使用ResNet，如果config.vision_backbone非ResNet18或ResNet50，這行需要修改
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # --- 3. 核心 Transformer 結構 (推理與訓練的主體) ---
        # 這裡的 Transformer encoder/decoder 即為 VAE 的解碼器部分（在推理時獨立運行）
        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # --- 4. Transformer 編碼器輸入投影 (Encoder Input Projections) ---
        # 將各種輸入特徵對齊到 Transformer 的 hidden dimension (dim_model)
        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].

        # 機器人當前狀態投影
        # 註：這裡的encoder_robot_state_input_proj用於訓練和推論時，輸入：隱變數(latent)+關節狀態+影像特徵
        #     上面的vae_encoder_robot_state_input_proj只用在訓練，輸入：[CLS]+關節狀態+動作序列
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )

        # 環境狀態投影 (如果有提供)
        # 註：這不是ACT論文原生的變數，而是LeRobot實作加入的，通常是用於在模擬環境（如 Mujoco 或 Unity）
        #     直接提供環境真值（Ground Truth State）
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )

        # VAE decoder採樣的隱變量z (Latent) 投影
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        # 影像特徵投影：使用 1x1 卷積將 Backbone 輸出的通道數轉為模型維度
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        # 文本特徵提取 (Language Backbone for ACT-LC)
        # 使用 getattr 確保即使 config 來自原生 LeRobot ACTConfig 也能正確建立語言組件
        _lang_model = getattr(self.config, 'language_model_name', 'distilbert-base-uncased')
        _lang_dim = getattr(self.config, 'language_dim', 768)
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(_lang_model)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_proj = nn.Linear(_lang_dim, config.dim_model)

        # --- 5. Transformer 編碼器位置編碼 ---
        # Transformer encoder positional embeddings.

        # 為 1D 特徵 (Latent, Robot State, Env State) 建立可學習的位置編碼
        n_1d_tokens = 1  # 至少有 latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        # 為 2D 影像特徵建立二維正弦位置編碼
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        _max_text_len = getattr(self.config, 'max_text_length', 16)
        self.encoder_text_feat_pos_embed = nn.Embedding(_max_text_len, config.dim_model)

        # --- 6. Transformer 解碼器部分 ---
        # Transformer decoder.
        # 建立可學習的 Query 位置編碼，長度為 chunk_size (即一次預測的動作步數)，風格類似 DETR
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # --- 7. 動作輸出頭 (Action Regression Head) ---
        # 最後的線性層，將 Transformer 解碼器的輸出轉回實際的動作維度 (如 14 維的關節角度)
        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        # 初始化參數 (使用 Xavier Uniform)
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """
        前向傳播主邏輯：
          輸入 (batch): 包含影像 (OBS_IMAGES)、機器人狀態 (OBS_STATE)、動作 (ACTION) 等。
          輸出: (預測動作, (隱變數均值 mu, 隱變數方差對數 log_sigma_x2))

        A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """

        # 1. 驗證訓練模式下的 VAE 輸入

        # 如果啟用了 VAE 且在訓練模式，必須提供目標動作序列 (ACTION) 才能進行編碼
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        # 獲取 Batch Size (優先從影像張量獲取，若無則從環境狀態獲取)
        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # --- 第一階段：準備潛在變數 (Latent) ---
        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # 【訓練模式】使用 VAE Encoder 提取特徵

            # 準備 VAE Encoder 的輸入 Token：[cls, (robot_state), action_sequence]
            # 註：因為電腦其實是同時平行用B(Batch)筆資料訓練，所以[CLS]需要複制B份，用來儲存這B筆錄制Data的特徵
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)

            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)

            # 每筆訓練資料只包含一個Chunk的動作序列，共S個時間步
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            # 將 Token 拼接成序列並加上位置編碼
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # 加上固定的正弦位置編碼
            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # 處理 Padding Mask (標記序列中哪些是填充的無效位元)
            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # 通過 VAE Encoder 並取回第一個 Token ([cls]) 的輸出
            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)

            # 預測高斯分佈參數：均值 mu 與 對數方差 log_sigma_x2
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # 重參數化技巧 (Reparameterization Trick) 採樣隱變數
            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # 【推理模式】或不使用 VAE 時，將隱變數設為全 0

            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # --- 第二階段：構建 Transformer Encoder 輸入 ---

        # 1. 放入 Latent Token
        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # 2. 放入機器人狀態 Token
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))

        # 3. 放入環境狀態 Token (若有)
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # 4. 放入多攝像頭影像 Token
        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch[OBS_IMAGES]:
                # 通過 Backbone (如 ResNet) 提取特徵圖
                cam_features = self.backbone(img)["feature_map"]
                # 加上 2D 正弦位置編碼
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                # 通過 1x1 卷積對齊維度
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # 將 (B, C, H, W) 展平成 (H*W, B, C) 的序列格式
                # H和W分別是特徵圖的高度和寬度，C是每個特徵點的向量維度，B是batch size，也就是電腦同時執行訓練或推論的資料數量
                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # 5. 放入 Text Token (ACT-LC)
        if hasattr(self.config, 'language_model_name') and "text_inputs" in batch:
            text_inputs = batch["text_inputs"]
            # pass through text_encoder
            text_outputs = self.text_encoder(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            # (Batch, Seq_Len, 768)
            text_embeds = text_outputs.last_hidden_state
            
            # Project to model dim
            text_tokens = self.text_proj(text_embeds)
            
            # Position embeddings for text (ACT 要求位置編碼不含 batch 維度，形狀應為 (Seq_Len, 1, Dim))
            seq_len = text_tokens.shape[1]
            text_pos_embed = self.encoder_text_feat_pos_embed.weight[:seq_len].unsqueeze(1)
            
            # Rearrange text_tokens to (Seq_Len, Batch, Dim) for Transformer Encoder
            text_tokens = text_tokens.transpose(0, 1)
            
            encoder_in_tokens.extend(list(text_tokens))
            encoder_in_pos_embed.extend(list(text_pos_embed))

        # 堆疊所有 Token 成為最終輸入序列
        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # --- 第三階段：Transformer 編碼與解碼 ---

        # Encoder 處理所有觀測資訊
        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # Decoder 準備全 0 的 Query Token，長度為 chunk_size，也就是S
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )

        # Decoder 進行 Cross-Attention，從 Encoder 輸出中提取動作資訊
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # --- 第四階段：輸出動作預測 ---

        # 轉回 (B, S, C) 格式
        decoder_out = decoder_out.transpose(0, 1)

        # 通過線性層回歸具體的動作數值
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """負責管理並串接多層的 ACTEncoderLayer
       執行多層編碼器 (Encoder Layers) 的模組，並在最後選擇性加上歸一化層。"""
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder

        # 1. 決定層數：
        # ACT 架構中，VAE 編碼器和主幹的 Transformer 編碼器可能會有不同的層數配置。
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers

        # 2. 建立層列表：
        # 使用 nn.ModuleList 將多個 ACTEncoderLayer 串接起來
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])

        # 3. 最終歸一化 (LayerNorm)：
        # 如果配置檔設定為 pre_norm (在注意力機制前做歸一化)，通常會在整個堆疊的最後補上一次歸一化。
        # 如果不是 pre_norm (即 post_norm)，則使用 nn.Identity() 不做任何事。
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        # 依次將資料通過每一層編碼器層
        for layer in self.layers:
            # 每一層都會接收當前的特徵 x、位置編碼，以及 padding 遮罩
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)

        # 輸出前進行最終的歸一化
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    """單一 Transformer 編碼器層，包含自注意力機制 (Self-Attention) 和前饋神經網路 (Feed Forward)"""
    
    def __init__(self, config: ACTConfig):
        super().__init__()
        # 1. 自注意力層 (Self-Attention)
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # 2. 前饋網路層 (Feed Forward Network, FFN)
        # 先將維度放大 (dim_model -> dim_feedforward)，再縮小回來，用來提取更複雜的非線性特徵
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # 3. 歸一化與 Dropout
        # norm1 用於 Self-Attention 前/後；norm2 用於 FFN 前/後
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        # 4. 激活函數 (Activation Function)
        # 決定 FFN 中間層使用的非線性激活函數，如 ReLU、GELU 等
        self.activation = get_activation_fn(config.feedforward_activation)

        # 5. 歸一化策略 (Pre-Norm vs Post-Norm)
        # pre_norm=True: 歸一化在 Attention/FFN 之前 (標準 Transformer 結構)
        # pre_norm=False: 歸一化在 Attention/FFN 之後 (原始 Transformer 結構)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x # 儲存輸入以便稍後做殘差連接 (Residual Connection)

        # --- 模塊 A：自注意力機制 ---
        if self.pre_norm:
            x = self.norm1(x)

        # 構建 Query(q) 和 Key(k)：
        # 在 Attention 中，位置資訊通常只加在 q 和 k 上, 不加在 Value(v) 上
        q = k = x if pos_embed is None else x + pos_embed

        # 執行 Self-Attention. 回傳的 tuple 中, [0] 是輸出特徵, [1] 是注意力權重分佈 (這裡不需要，所以只取 [0])
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights

        # 殘差連接：原本的輸入 + 注意力層的輸出 (防止梯度消失)
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x # 更新 skip 準備給下一個模塊用
            x = self.norm2(x)
        else:
            x = self.norm1(x) # 如果是 post-norm，在這裡才做歸一化
            skip = x

        # --- 模塊 B：前饋網路 (FFN) ---

        # 依序通過：線性層 1 -> 激活函式 -> Dropout -> 線性層 2
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # 第二次殘差連接
        x = skip + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)

        return x


class ACTDecoder(nn.Module):
    """負責管理多層的 ACTDecoderLayer, 用於生成動作
       簡單來說, Transformer 解碼器會不斷向 Transformer 編碼器 詢問: 
       [根據你剛剛融合好的當前畫面/機器人狀態, 以及 VAE 給你的潛在風格指令, 我這一步該做出什麼動作?]
    """

    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        # 根據配置檔設定解碼器的層數 (通常比編碼器少或一樣)
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])

        # 最後一層的歸一化
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        # 逐層執行解碼
        for layer in self.layers:
            # 解碼器需要同時接收「自身的輸入 x」與「編碼器的輸出 encoder_out」
            # 並分別配對對應的位置編碼
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )

        # 最終歸一化
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    """單一 Transformer 解碼器層，包含三個子模塊：
    1. 自注意力 (Self-Attention)：處理輸入序列自身的上下文。
    2. 交叉注意力 (Cross-Attention)：將編碼器的輸出 (Encoder Output) 引入。
    3. 前饋網路 (FFN)：進一步處理特徵。"""
    def __init__(self, config: ACTConfig):
        super().__init__()

        # 解碼器有兩個 Attention 模組：
        # 1. 處理解碼器內部 Query 之間的互動 (例如動作第 1 步與第 2 步的關係)
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        # 2. 處理 Query 與 Transformer Encoder 輸出特徵 (影像/狀態) 之間的互動
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # 前饋網路 (FFN)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # 解碼器有三個子模塊，所以需要三個 LayerNorm 和 Dropout
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        """輔助函式：如果位置編碼存在，就將其加到張量上。"""
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        x: 解碼器的輸入 (通常是全 0 的 Query, 形狀為 [Chunk_Size, Batch, Dim])
        encoder_out: 編碼器融合後的觀測特徵

        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """

        # --- 模塊 A：自注意力 (Self-Attention) ---
        skip = x
        if self.pre_norm:
            x = self.norm1(x)

        # 對 Query 加入解碼器的位置編碼 (代表這是預測序列的第幾步)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)

        # --- 模塊 B：交叉注意力 (Cross-Attention) --- 
        # 這是 Decoder 最關鍵的一步：透過 Query 去 Encoder 那邊「尋找」需要的特徵
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # 注意這裡的 Q, K, V 分配：
        # Query: 來自解碼器自己 (加上了解碼器位置編碼)
        # Key: 來自編碼器輸出 (加上了編碼器影像/狀態的位置編碼)
        # Value: 來自編碼器輸出的純特徵
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)

        # --- 模塊 C：前饋網路 (FFN) ---
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.
    為二維影像特徵圖(Feature Map)生成正弦位置編碼. 這讓 Transformer 知道影像中每個像素點的相對空間位置(上下左右)

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
