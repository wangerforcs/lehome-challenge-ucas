import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from torchvision import models, transforms

# 官方规定的基类和注册器
from .base_policy import BasePolicy
from .registry import PolicyRegistry

@PolicyRegistry.register("custom")
class CustomPolicy(BasePolicy):
    """
    LeHome Challenge 2026 - 动态路由混合专家策略 (Dynamic MoE Router)
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)

        # 【终极修复】：无视外部传入的默认路径干扰，强制锁定当前项目的物理根目录！
        self.base_dir = "/home/wzb/challenges/weights/act_moe"

        print(f"\n🚀 [MoE Router] 系统初始化启动! (Device: {self.device})")
        print(f"📁 [MoE Router] 锁定工作根目录: {self.base_dir}")

        # 1. 挂载四大专家模型的路径 + 对应的数据集元数据路径
        self.expert_configs = {
            'pant_long': {
                'policy_path': os.path.join(self.base_dir, "outputs/train/act_h200_pant_long/checkpoints/080000/pretrained_model"),
                'dataset_root': os.path.join(self.base_dir, "Datasets/example/pant_long_merged")
            },
            'pant_short': {
                'policy_path': os.path.join(self.base_dir, "outputs/train/act_h200_pant_short/checkpoints/080000/pretrained_model"),
                'dataset_root': os.path.join(self.base_dir, "Datasets/example/pant_short_merged")
            },
            'top_long': {
                'policy_path': os.path.join(self.base_dir, "outputs/train/act_h200_top_long/checkpoints/last/pretrained_model"),
                'dataset_root': os.path.join(self.base_dir, "Datasets/example/top_long_merged")
            },
            'top_short': {
                'policy_path': os.path.join(self.base_dir, "outputs/train/act_h200_top_short/checkpoints/last/pretrained_model"),
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

    def load_expert(self, garment_type):
        """动态显存管理：卸载旧专家，使用 PolicyRegistry.create 实例化新专家"""
        if self.current_expert_name == garment_type:
            return

        print(f"\n👁️ [MoE Router] 视觉确认目标: {garment_type}。正在执行热切换...")

        # 卸载旧模型，释放显存
        if self.current_policy is not None:
            del self.current_policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        config = self.expert_configs[garment_type]
        if not os.path.exists(config['policy_path']):
            raise FileNotFoundError(f"❌ 找不到专家模型: {config['policy_path']}")

        # 调用官方注册好的 lerobot 包装器
        self.current_policy = PolicyRegistry.create(
            "lerobot",
            policy_path=config['policy_path'],
            dataset_root=config['dataset_root'],
            device=self.device.type,
            task_description=f"fold the {garment_type}"
        )

        self.current_expert_name = garment_type
        print(f"✅ [MoE Router] 专家 [{garment_type}] 已接管机械臂控制权！\n")

    def reset(self):
        self.step_count = 0
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
            self.current_policy.reset()

        # 2. 直接将原生的 Numpy observation 丢给官方的包装器
        action = self.current_policy.select_action(observation)

        self.step_count += 1
        return action(lehome)