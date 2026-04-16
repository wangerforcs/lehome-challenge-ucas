from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_spatial_forcing")
@dataclass
class PI05SpatialForcingConfig(PI05Config):
    vggt_weight_path: str | None = None
    vggt_dim: int = 1024

    # Align a configurable VLA prefix hidden layer against VGGT features.
    vla_layers_align: int = 12
    pooling_func: str = "bilinear"
    use_vggt_pe: bool = False
    use_vlm_norm: bool = False
    align_loss_coeff: float = 0.0

    # Which camera streams to use for spatial forcing. Empty means "all visual inputs".
    spatial_forcing_image_keys: tuple[str, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if self.align_loss_coeff < 0:
            raise ValueError("align_loss_coeff must be >= 0")
        if self.pooling_func not in {"bilinear"}:
            raise ValueError(f"Unsupported pooling_func: {self.pooling_func}")
