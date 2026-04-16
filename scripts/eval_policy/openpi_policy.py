from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import Dict
from typing import Optional

import numpy as np

from .base_policy import BasePolicy
from .registry import PolicyRegistry


def _resolve_openpi_repo_root(openpi_repo_root: Optional[str]) -> Path:
    if openpi_repo_root:
        return Path(openpi_repo_root).expanduser().resolve()
    return Path("/home/wzb/vla/openpi").resolve()


@PolicyRegistry.register("openpi")
class OpenPIPolicy(BasePolicy):
    """
    In-process adapter that runs an openpi policy directly inside LeHome evaluation.

    This keeps the Isaac Sim evaluation loop unchanged while letting LeHome use
    checkpoints trained with official openpi.
    """

    def __init__(
        self,
        policy_path: Optional[str] = None,
        model_path: Optional[str] = None,
        task_description: str = "fold the garment on the table",
        openpi_config_name: str = "pi05_lehome_top_short",
        openpi_repo_root: Optional[str] = None,
        openpi_default_prompt: Optional[str] = None,
        replan_steps: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        checkpoint_dir = policy_path or model_path
        if not checkpoint_dir:
            raise ValueError("OpenPIPolicy requires policy_path or model_path")

        repo_root = _resolve_openpi_repo_root(openpi_repo_root)
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from openpi.policies import policy_config as openpi_policy_config
        from openpi.training import config as openpi_train_config

        self.task_description = task_description
        self.replan_steps = replan_steps
        self._action_queue: collections.deque[np.ndarray] = collections.deque()
        self._policy = openpi_policy_config.create_trained_policy(
            openpi_train_config.get_config(openpi_config_name),
            checkpoint_dir,
            default_prompt=openpi_default_prompt or task_description,
        )

    def reset(self):
        self._action_queue.clear()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if not self._action_queue:
            element = {
                "observation/top_image": observation["observation.images.top_rgb"],
                "observation/left_wrist_image": observation["observation.images.left_rgb"],
                "observation/right_wrist_image": observation["observation.images.right_rgb"],
                "observation/state": observation["observation.state"],
                "prompt": self.task_description,
            }
            action_chunk = self._policy.infer(element)["actions"]
            if len(action_chunk) < self.replan_steps:
                raise ValueError(
                    f"Requested replan_steps={self.replan_steps}, but policy only returned {len(action_chunk)} steps"
                )
            self._action_queue.extend(np.asarray(action_chunk[: self.replan_steps], dtype=np.float32))

        return self._action_queue.popleft()
