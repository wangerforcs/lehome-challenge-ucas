"""Quick validator for garment checkpoint indices against the loaded cloth mesh size."""

import argparse
import multiprocessing
from pathlib import Path

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

from scripts.utils import common


def _read_garment_names(args) -> list[str]:
    if args.garment_name:
        return [args.garment_name]

    if not args.garment_list_file:
        raise ValueError("Either --garment_name or --garment_list_file is required.")

    garment_list_file = Path(args.garment_list_file)
    if not garment_list_file.exists():
        raise FileNotFoundError(f"Garment list file not found: {garment_list_file}")

    return [line.strip() for line in garment_list_file.read_text().splitlines() if line.strip()]


def _get_mesh_point_count(env) -> int:
    try:
        transformed_mesh_points, _, _, _ = env.object.get_current_mesh_points()
        return int(transformed_mesh_points.shape[0])
    except Exception:
        positions = env.object._cloth_prim_view.get_world_positions().squeeze(0)
        return int(positions.shape[0])


def _make_zero_action(env):
    import torch

    obs = env._get_observations()
    action_dim = len(obs["observation.state"])
    return torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)


def check_garments(args) -> int:
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    from scripts.utils.common import stabilize_garment_after_reset

    garment_names = _read_garment_names(args)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[Warn] Requested device '{requested_device}' is unavailable; falling back to cpu."
        )
        requested_device = "cpu"

    env_cfg = parse_env_cfg(args.task, device=requested_device)
    env_cfg.sim.use_fabric = False
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path
    initial_garment_name = args.switch_from if args.switch_from else garment_names[0]
    env_cfg.garment_name = initial_garment_name
    env_cfg.garment_version = args.garment_version

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    failed = 0
    try:
        current_garment = initial_garment_name
        for garment_name in garment_names:
            if garment_name != current_garment:
                env.switch_garment(garment_name, args.garment_version)
                current_garment = garment_name

            env.reset()
            stabilize_garment_after_reset(env, args)

            check_points = list(getattr(env.object, "check_points", []))
            point_count = _get_mesh_point_count(env)
            invalid = [idx for idx in check_points if idx < 0 or idx >= point_count]

            print(f"\n[{garment_name}]")
            print(f"mesh_point_count={point_count}")
            print(f"check_points={check_points}")

            if invalid:
                failed += 1
                print(f"INVALID check_points={invalid}")
            else:
                print("check_points OK")

            if args.check_success:
                zero_action = _make_zero_action(env)
                try:
                    for check_idx in range(1, args.check_calls + 1):
                        if args.step_between_checks:
                            env.step(zero_action)
                        result = env._get_success().item()
                        print(f"_get_success() call {check_idx}/{args.check_calls} -> {result}")
                except Exception as exc:
                    failed += 1
                    print(f"_get_success() FAILED -> {type(exc).__name__}: {exc}")
    finally:
        env.close()

    print(f"\nSummary: checked={len(garment_names)}, failed={failed}")
    return 1 if failed else 0


def main() -> None:
    isaac_args_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(isaac_args_parser)

    parser = argparse.ArgumentParser(
        description="Validate garment check_point indices against actual loaded cloth mesh size.",
        parents=[isaac_args_parser],
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task environment.",
    )
    parser.add_argument(
        "--garment_name",
        type=str,
        default=None,
        help="Single garment to validate, e.g. Top_Short_Seen_3.",
    )
    parser.add_argument(
        "--garment_list_file",
        type=str,
        default=None,
        help="Text file containing one garment name per line.",
    )
    parser.add_argument(
        "--switch_from",
        type=str,
        default=None,
        help="Optional garment to load first, then switch to --garment_name / list entries.",
    )
    parser.add_argument(
        "--garment_version",
        type=str,
        default="Release",
        help="Garment asset version.",
    )
    parser.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of the garment configuration.",
    )
    parser.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path of the particle configuration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Simulation device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--check_success",
        action="store_true",
        default=False,
        help="Also call env._get_success() repeatedly to reproduce runtime errors directly.",
    )
    parser.add_argument(
        "--check_calls",
        type=int,
        default=60,
        help="Number of consecutive env._get_success() calls. Use >=50 to trigger the actual checker.",
    )
    parser.add_argument(
        "--step_between_checks",
        action="store_true",
        default=False,
        help="Step a zero action before each env._get_success() call, matching replay more closely.",
    )

    args = parser.parse_args()
    simulation_app = common.launch_app_from_args(args)
    try:
        import lehome.tasks.bedroom  # noqa: F401

        raise SystemExit(check_garments(args))
    finally:
        common.close_app(simulation_app)


if __name__ == "__main__":
    main()
