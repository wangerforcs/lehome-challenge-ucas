from collections import deque
from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


@PolicyRegistry.register("pi05_custom")
class PI05CustomPolicy(BasePolicy):
    """
    Minimal wrapper around LeRobotPolicy for PI05 checkpoints.

    This exists so you can later edit execution behavior in one dedicated file
    without touching the default lerobot adapter.

    Current execution schedule:
    - first inference: execute 30 actions
    - second inference: execute 10 actions
    - all later inferences: execute 5 actions
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
            raise ValueError("PI05CustomPolicy requires policy_path or model_path")
        if not dataset_root:
            raise ValueError("PI05CustomPolicy requires dataset_root")

        self.inner_policy = LeRobotPolicy(
            policy_path=resolved_policy_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device,
        )
        self.replan_schedule = [30, 10]
        self.steady_replan_steps = 5
        self.schedule_index = 0
        self.remaining_actions_in_window = 0

        runtime_cfg = getattr(self.inner_policy.policy, "config", None)
        self.native_n_action_steps = (
            getattr(runtime_cfg, "n_action_steps", None) if runtime_cfg is not None else None
        )
        max_requested_steps = max(max(self.replan_schedule), self.steady_replan_steps)
        if (
            self.native_n_action_steps is not None
            and max_requested_steps > self.native_n_action_steps
        ):
            raise ValueError(
                "Custom replan schedule exceeds policy n_action_steps: "
                f"{max_requested_steps} > {self.native_n_action_steps}"
            )

    def reset(self):
        self.schedule_index = 0
        self.remaining_actions_in_window = 0
        self.inner_policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if self.remaining_actions_in_window == 0:
            window_size = self._next_window_size()
            self._force_replan_with_window(window_size)
            action = self.inner_policy.select_action(observation)
            self._trim_action_queue(window_size - 1)
            self.remaining_actions_in_window = window_size - 1
            logger.info(
                f"PI05CustomPolicy forcing fresh inference window: {window_size} actions"
            )
            return action

        self.remaining_actions_in_window -= 1
        return self.inner_policy.select_action(observation)

    def _next_window_size(self) -> int:
        if self.schedule_index < len(self.replan_schedule):
            window_size = self.replan_schedule[self.schedule_index]
        else:
            window_size = self.steady_replan_steps
        self.schedule_index += 1
        return window_size

    def _force_replan_with_window(self, window_size: int) -> None:
        policy = self.inner_policy.policy
        action_queue = getattr(policy, "_action_queue", None)
        if action_queue is None:
            raise AttributeError("Underlying policy does not expose _action_queue")

        action_queue.clear()

        if hasattr(action_queue, "maxlen") and action_queue.maxlen is not None:
            if window_size > action_queue.maxlen:
                raise ValueError(
                    f"Requested window_size={window_size} exceeds queue maxlen={action_queue.maxlen}"
                )

        if hasattr(policy, "_queues") and isinstance(policy._queues, dict):
            for queue in policy._queues.values():
                if hasattr(queue, "clear"):
                    queue.clear()

        policy._action_queue = deque(maxlen=action_queue.maxlen)
        policy._custom_replan_window = window_size

    def _trim_action_queue(self, keep_count: int) -> None:
        action_queue = getattr(self.inner_policy.policy, "_action_queue", None)
        if action_queue is None:
            return
        while len(action_queue) > keep_count:
            action_queue.pop()
