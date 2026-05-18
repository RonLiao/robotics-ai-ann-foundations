from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig

try:
    from lerobot.policies.act.configuration_act import ACTConfig as _VanillaACTConfig
except ImportError:
    from lerobot.common.policies.act.configuration_act import ACTConfig as _VanillaACTConfig


@PreTrainedConfig.register_subclass("act_te")
@dataclass
class ACTConfig(_VanillaACTConfig):
    """ACT with Task Embedding.

    Inherits all fields from vanilla ACTConfig.
    Only addition: num_tasks for learnable FiLM conditioning (replaces DistilBERT).
    isinstance(config, vanilla_ACTConfig) = True — passes all lerobot factory checks.
    """

    num_tasks: int = 3

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None
