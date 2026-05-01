import collections
import os
from types import MethodType
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from torchvision import models, transforms

from openpi_client import websocket_client_policy as _websocket_client_policy

from .base_policy import BasePolicy
from .registry import PolicyRegistry

@PolicyRegistry.register("custom")
class CustomPolicy(BasePolicy):
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 openpi_host: str = "127.0.0.1", openpi_port: int = 8123,
                 openpi_replan_steps: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)

        self.openpi_host = openpi_host
        self.openpi_port = openpi_port
        self.openpi_replan_steps = openpi_replan_steps
        self._openpi_client = None
        self._action_queue: collections.deque[np.ndarray] = collections.deque()
        
        self.base_dir = os.getcwd()
        
        self.current_expert_name = None
        self.current_policy = None
        self.step_count = 0

    def load_expert(self):

        print(f"🔗 连接 openpi server ({self.openpi_host}:{self.openpi_port})")
        self._openpi_client = _websocket_client_policy.WebsocketClientPolicy(
            self.openpi_host, self.openpi_port
        )

    def reset(self):
        self.step_count = 0
        self._action_queue.clear()
        if self.current_policy is not None:
            self.current_policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if self.step_count == 0:
            self.load_expert()
        action = self._select_action_openpi(observation)
        self.step_count += 1
        return action

    def _select_action_openpi(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """openpi server 推理路径：映射观测 key，通过 WebSocket 获取动作"""
        if not self._action_queue:
            element = {
                "observation/top_image": observation["observation.images.top_rgb"],
                "observation/left_wrist_image": observation["observation.images.left_rgb"],
                "observation/right_wrist_image": observation["observation.images.right_rgb"],
                "observation/state": observation["observation.state"],
                "prompt": "fold the garment on the table",
            }
            action_chunk = self._openpi_client.infer(element)["actions"]
            self._action_queue.extend(
                np.asarray(action_chunk[: self.openpi_replan_steps], dtype=np.float32)
            )
        return self._action_queue.popleft()
