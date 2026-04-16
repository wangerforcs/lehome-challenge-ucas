"""
LeHome Challenge Policy Module

This module provides the base policy interface and implementations
for the LeHome Challenge evaluation framework.
"""

from .base_policy import BasePolicy
from .registry import PolicyRegistry

# Import policy implementations (this will auto-register them)
from .lerobot_policy import LeRobotPolicy
from .example_participant_policy import CustomPolicy
from .pi05_custom_policy import PI05CustomPolicy
from .pi05_spatial_forcing_policy import PI05SpatialForcingEvalPolicy
from .docker_policy import DockerPolicy
from .openpi_policy import OpenPIPolicy

__all__ = [
    "BasePolicy",
    "PolicyRegistry",
    "LeRobotPolicy",
    "CustomPolicy",
    "PI05CustomPolicy",
    "PI05SpatialForcingEvalPolicy",
    "DockerPolicy",
    "OpenPIPolicy",
]
