"""Custom LeRobot training entrypoint that registers local third-party policies before parsing config."""

from pathlib import Path
import sys


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main():
    _ensure_repo_root_on_path()

    # Import for registration side effects.
    import lerobot_policy_pi05_spatial_forcing  # noqa: F401

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
