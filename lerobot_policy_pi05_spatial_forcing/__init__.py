"""Local LeRobot custom policy package for PI05 Spatial Forcing."""

from pathlib import Path
import sys


def _ensure_spatial_forcing_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    spatial_forcing_src = repo_root / "Spatial-Forcing" / "openpi-SF" / "src"
    if spatial_forcing_src.exists():
        src_str = str(spatial_forcing_src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_spatial_forcing_src_on_path()

# Import config for registration side effects.
from .configuration_pi05_spatial_forcing import PI05SpatialForcingConfig

__all__ = ["PI05SpatialForcingConfig"]
