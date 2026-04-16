from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("pi05_spatial_forcing")
class PI05SpatialForcingEvalPolicy(BasePolicy):
    """
    Evaluation wrapper for checkpoints trained with the local `pi05_spatial_forcing`
    LeRobot policy package.

    This keeps evaluation usage explicit, so users do not need to route these
    checkpoints through the generic `lerobot` policy type.
    """

    def __init__(
        self,
        policy_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        task_description: str = "fold the garment on the table",
        device: str = "cpu",
        model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        resolved_policy_path = policy_path or model_path
        if not resolved_policy_path:
            raise ValueError("PI05SpatialForcingEvalPolicy requires policy_path or model_path")
        if not dataset_root:
            raise ValueError("PI05SpatialForcingEvalPolicy requires dataset_root")

        self.inner_policy = LeRobotPolicy(
            policy_path=resolved_policy_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device,
        )

    def reset(self):
        self.inner_policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return self.inner_policy.select_action(observation)
