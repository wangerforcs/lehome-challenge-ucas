import collections
import os
from types import MethodType
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from torchvision import models, transforms

from openpi_client import websocket_client_policy as _websocket_client_policy

# 官方规定的基类和注册器
from .base_policy import BasePolicy
from .registry import PolicyRegistry

@PolicyRegistry.register("custom")
class CustomPolicy(BasePolicy):
    """
    LeHome Challenge 2026 - 动态路由混合专家策略 (Dynamic MoE Router)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 openpi_host: str = "127.0.0.1", openpi_port: int = 8123,
                 openpi_replan_steps: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)

        self.openpi_host = openpi_host
        self.openpi_port = openpi_port
        self.openpi_replan_steps = openpi_replan_steps
        self._openpi_client = None
        self._use_openpi = False
        self._action_queue: collections.deque[np.ndarray] = collections.deque()
        
        # 【终极修复】：无视外部传入的默认路径干扰，强制锁定当前项目的物理根目录！
        self.base_dir = os.getcwd()
        
        print(f"\n🚀 [MoE Router] 系统初始化启动! (Device: {self.device})")
        print(f"📁 [MoE Router] 锁定工作根目录: {self.base_dir}")

        # 1. 挂载四大专家模型的路径 + 对应的数据集元数据路径
        self.expert_configs = {
            'pant_long': {
                'policy_path': os.path.join(self.base_dir, "outputs/train/pant_long_best"),
                'dataset_root': os.path.join(self.base_dir, "Datasets/example/pant_long_merged")
            },
            'pant_short': {
                'policy_path': os.path.join(self.base_dir, "outputs/train/pant_short_best"),
                'dataset_root': os.path.join(self.base_dir, "Datasets/example/pant_short_merged")
            },
            'top_long': {
                'policy_path': os.path.join(self.base_dir, "outputs/train/top_long_best"),
                'dataset_root': os.path.join(self.base_dir, "Datasets/example/top_long_merged")
            },
            'top_short': {
                'policy_path': os.path.join(self.base_dir, "outputs/train/top_short_best"),
                'dataset_root': os.path.join(self.base_dir, "Datasets/example/top_short_merged")
            }
        }
        
        self.idx_to_class = {0: 'pant_long', 1: 'pant_short', 2: 'top_long', 3: 'top_short'}
        
        # 2. 唤醒前置视觉大脑 (ResNet18 Classifier)
        print("🧠 [MoE Router] 正在加载视觉分类器网络...")
        self.classifier = models.resnet18(weights=None)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, 4)
        
        classifier_weight_path = os.path.join(self.base_dir, "outputs/classifier/garment_classifier_resnet18.pth")
        self.classifier.load_state_dict(torch.load(classifier_weight_path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.vision_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.current_expert_name = None
        self.current_policy = None
        self.step_count = 0

    # 这两个都是为了xvla修改的
    def _prepare_observation_for_expert(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add XVLA-compatible image aliases without mutating the original observation."""
        prepared = dict(observation)

        image_aliases = {
            "observation.images.top_rgb": "observation.images.image",
            "observation.images.left_rgb": "observation.images.image2",
            "observation.images.right_rgb": "observation.images.image3",
        }

        for src_key, dst_key in image_aliases.items():
            if src_key in prepared and dst_key not in prepared:
                prepared[dst_key] = prepared[src_key]

        return prepared

    def _patch_current_policy_select_action(self):
        """Patch the loaded LeRobotPolicy instance locally for XVLA compatibility."""
        if self.current_policy is None or getattr(self.current_policy, "_custom_eval_patch_applied", False):
            return

        def _patched_select_action(policy_self, observation: Dict[str, np.ndarray]) -> np.ndarray:
            if policy_self.input_features:
                observation = policy_self._filter_observations(observation, policy_self.input_features)

            batch_obs = policy_self._process_observation(observation)

            with torch.inference_mode():
                batch_action = policy_self.policy.select_action(batch_obs)

            if policy_self.postprocessor:
                batch_action = policy_self.postprocessor(batch_action)

            return batch_action.squeeze(0).to(dtype=torch.float32).cpu().numpy()

        self.current_policy.select_action = MethodType(_patched_select_action, self.current_policy)
        self.current_policy._custom_eval_patch_applied = True

    def load_expert(self, garment_type):
        """动态显存管理：卸载旧专家，top_short 走 openpi server，其余走 lerobot"""
        if self.current_expert_name == garment_type:
            return

        print(f"\n👁️ [MoE Router] 视觉确认目标: {garment_type}。正在执行热切换...")

        # 卸载旧模型，释放显存
        if self.current_policy is not None:
            del self.current_policy
            self.current_policy = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._use_openpi = False
        self._openpi_client = None
        self._action_queue.clear()

        if garment_type == "top_short":
            # top_short: 通过 WebSocket 连接 openpi server
            print(f"🔗 [MoE Router] top_short -> 连接 openpi server ({self.openpi_host}:{self.openpi_port})")
            self._openpi_client = _websocket_client_policy.WebsocketClientPolicy(
                self.openpi_host, self.openpi_port
            )
            self._use_openpi = True
        else:
            # 其余衣物类型：本地加载 lerobot 策略
            config = self.expert_configs[garment_type]
            if not os.path.exists(config['policy_path']):
                raise FileNotFoundError(f"❌ 找不到专家模型: {config['policy_path']}")

            self.current_policy = PolicyRegistry.create(
                "lerobot",
                policy_path=config['policy_path'],
                dataset_root=config['dataset_root'],
                device=self.device.type,
                task_description=f"fold the {garment_type}"
            )
            self._patch_current_policy_select_action()

        self.current_expert_name = garment_type
        print(f"✅ [MoE Router] 专家 [{garment_type}] 已接管机械臂控制权！\n")

    def reset(self):
        self.step_count = 0
        self._action_queue.clear()
        if self.current_policy is not None:
            self.current_policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        # 1. 第 0 步：截获图像进行分类
        if self.step_count == 0:
            img_key = next((k for k in observation.keys() if 'top' in k and 'image' in k), None)
            raw_img_numpy = observation[img_key]

            with torch.no_grad():
                processed_img = self.vision_transform(raw_img_numpy).unsqueeze(0).to(self.device)
                outputs = self.classifier(processed_img)
                _, preds = torch.max(outputs, 1)
                predicted_idx = preds.item()

            garment_type = self.idx_to_class[predicted_idx]
            self.load_expert(garment_type)
            if not self._use_openpi:
                self.current_policy.reset()

        # 2. 根据后端选择推理路径
        if self._use_openpi:
            action = self._select_action_openpi(observation)
        else:
            expert_observation = self._prepare_observation_for_expert(observation)
            action = self.current_policy.select_action(expert_observation)

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