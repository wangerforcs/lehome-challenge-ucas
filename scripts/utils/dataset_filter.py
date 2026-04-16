"""Filter garment datasets by replaying episodes with the evaluation success logic."""

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import pandas as pd
import torch

from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.utils.logger import get_logger
from lehome.utils.record import RateLimiter, append_episode_initial_pose

from .common import stabilize_garment_after_reset
from .eval_utils import save_videos_from_observations

logger = get_logger(__name__)


def _resolve_dataset_roots(args) -> list[Path]:
    roots: list[Path] = []
    if args.dataset_root:
        roots.append(Path(args.dataset_root).resolve())

    if args.dataset_parent_root:
        parent = Path(args.dataset_parent_root).resolve()
        matched = sorted(
            path
            for path in parent.glob(args.dataset_pattern)
            if path.is_dir() and (path / "meta" / "info.json").exists()
        )
        roots.extend(matched)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)

    if not deduped:
        raise ValueError(
            "No dataset roots found. Use --dataset_root or --dataset_parent_root."
        )

    return deduped


def _load_episode_records(
    dataset_root: Path, expected_total_episodes: Optional[int] = None
) -> tuple[dict[int, dict[str, Any]], str]:
    garment_info_path = dataset_root / "meta" / "garment_info.json"
    if not garment_info_path.exists():
        raise FileNotFoundError(f"garment_info.json not found: {garment_info_path}")

    with garment_info_path.open("r", encoding="utf-8") as f:
        garment_info = json.load(f)

    episode_records: dict[int, dict[str, Any]] = {}
    duplicate_global_keys = False
    total_entries = 0
    for garment_name, episodes in garment_info.items():
        for episode_idx_str, episode_meta in episodes.items():
            episode_idx = int(episode_idx_str)
            if episode_idx in episode_records:
                duplicate_global_keys = True
            episode_records[episode_idx] = {
                "garment_name": garment_name,
                "object_initial_pose": episode_meta.get("object_initial_pose"),
                "scale": episode_meta.get("scale"),
            }
            total_entries += 1

    if not garment_info:
        raise ValueError(f"No episode records found in {garment_info_path}")

    if not duplicate_global_keys:
        if expected_total_episodes is not None and len(episode_records) != expected_total_episodes:
            logger.warning(
                "garment_info.json uses global episode indices, but the count does not match "
                f"meta episodes: garment_info={len(episode_records)}, meta={expected_total_episodes}"
            )
        return episode_records, "global"

    sequential_records: dict[int, dict[str, Any]] = {}
    next_global_idx = 0
    for garment_name, episodes in garment_info.items():
        for episode_idx in sorted(episodes, key=lambda x: int(x)):
            episode_meta = episodes[episode_idx]
            sequential_records[next_global_idx] = {
                "garment_name": garment_name,
                "object_initial_pose": episode_meta.get("object_initial_pose"),
                "scale": episode_meta.get("scale"),
                "local_episode_index": int(episode_idx),
            }
            next_global_idx += 1

    if expected_total_episodes is not None and next_global_idx != expected_total_episodes:
        raise ValueError(
            "garment_info.json appears to use per-garment local indices, but the reconstructed "
            f"episode count does not match meta episodes: reconstructed={next_global_idx}, "
            f"meta={expected_total_episodes}"
        )

    logger.warning(
        "Detected repeated local episode indices in garment_info.json. "
        "Reconstructed global episode mapping by iterating garments in file order."
    )
    logger.info(
        f"Reconstructed {next_global_idx} global episode records from {total_entries} garment_info entries"
    )
    return sequential_records, "sequential_by_garment"


def _load_episode_table(dataset_root: Path) -> pd.DataFrame:
    episodes_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes parquet not found: {episodes_path}")

    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
    ]
    table = pd.read_parquet(episodes_path, columns=columns).sort_values("episode_index")
    return table.reset_index(drop=True)


def _create_output_dataset(
    source_dataset: LeRobotDataset,
    dataset_root: Path,
    output_root: Optional[Path],
    disable_depth: bool,
) -> Optional[LeRobotDataset]:
    if output_root is None:
        return None

    target_root = output_root / dataset_root.name
    if target_root.exists():
        logger.warning(f"Target path already exists, removing: {target_root}")
        shutil.rmtree(target_root)
    features = dict(source_dataset.meta.features)
    if disable_depth and "observation.top_depth" in features:
        features = {
            key: value for key, value in features.items() if key != "observation.top_depth"
        }

    logger.info(f"Creating filtered dataset at: {target_root}")
    return LeRobotDataset.create(
        repo_id=f"{dataset_root.name}_filtered",
        fps=source_dataset.fps,
        root=target_root,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )


def _export_episode(
    source_dataset: LeRobotDataset,
    filtered_dataset: LeRobotDataset,
    frame_start: int,
    frame_end: int,
    task_description: str,
) -> None:
    auto_managed_keys = {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    feature_keys = {
        key for key in filtered_dataset.features if key not in auto_managed_keys
    }
    for frame_idx in range(frame_start, frame_end):
        item = source_dataset[frame_idx]
        frame = {}
        for key in feature_keys:
            if key not in item:
                continue
            value = item[key]
            if (
                key.startswith("observation.images.")
                and hasattr(value, "ndim")
                and value.ndim == 3
                and value.shape[0] in (1, 3)
            ):
                # LeRobot returns images in CHW; recording expects HWC.
                value = value.permute(1, 2, 0)
            frame[key] = value
        frame["task"] = item.get("task", task_description)
        filtered_dataset.add_frame(frame)
    filtered_dataset.save_episode()


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _record_invalid_config(
    report: dict[str, Any],
    garment_name: str,
    episode_index: int,
    reason: str,
) -> None:
    report["invalid_config_episode_indices"].append(episode_index)
    report["invalid_config_details"].append(
        {
            "episode_index": episode_index,
            "garment_name": garment_name,
            "reason": reason,
        }
    )
    report["per_garment"][garment_name]["invalid_config"].append(episode_index)


def _parse_index_filter(raw: Optional[str]) -> set[int]:
    if not raw:
        return set()
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def _matches_garment_filters(
    garment_name: str,
    explicit_names: set[str],
    seen_indices: set[int],
    unseen_indices: set[int],
) -> bool:
    if explicit_names and garment_name in explicit_names:
        return True

    if seen_indices:
        for idx in seen_indices:
            if f"_Seen_{idx}" in garment_name:
                return True

    if unseen_indices:
        for idx in unseen_indices:
            if f"_Unseen_{idx}" in garment_name:
                return True

    if explicit_names or seen_indices or unseen_indices:
        return False

    return True


def _switch_to_garment(env, env_cfg, task: str, garment_name: str, garment_version: str):
    if hasattr(env, "switch_garment"):
        env.switch_garment(garment_name, garment_version)
    else:
        env.close()
        env_cfg.garment_name = garment_name
        env = gym.make(task, cfg=env_cfg).unwrapped
        env.initialize_obs()
    env_cfg.garment_name = garment_name
    return env


def filter_single_dataset(args, dataset_root: Path) -> dict[str, Any]:
    logger.info(f"Filtering dataset: {dataset_root}")

    source_dataset = LeRobotDataset(repo_id="filter_source", root=dataset_root)
    episode_table = _load_episode_table(dataset_root)
    episode_records, mapping_mode = _load_episode_records(
        dataset_root, expected_total_episodes=len(episode_table)
    )
    logger.info(f"garment_info mapping mode: {mapping_mode}")

    start_episode = args.start_episode
    end_episode = (
        args.end_episode if args.end_episode is not None else int(episode_table["episode_index"].max()) + 1
    )
    explicit_garment_names = {
        name.strip() for name in args.garment_names.split(",") if name.strip()
    } if args.garment_names else set()
    seen_indices = _parse_index_filter(args.seen_indices)
    unseen_indices = _parse_index_filter(args.unseen_indices)

    episode_table = episode_table[
        (episode_table["episode_index"] >= start_episode)
        & (episode_table["episode_index"] < end_episode)
    ].copy()
    episode_table = episode_table[
        episode_table["episode_index"].map(
            lambda ep: _matches_garment_filters(
                episode_records[int(ep)]["garment_name"],
                explicit_garment_names,
                seen_indices,
                unseen_indices,
            )
        )
    ].copy()

    if episode_table.empty:
        raise ValueError(
            f"No episodes selected after applying range/garment filters for {dataset_root}"
        )
    if args.num_trials < 1:
        raise ValueError("--num_trials must be >= 1")
    if args.min_successes < 1:
        raise ValueError("--min_successes must be >= 1")
    if args.min_successes > args.num_trials:
        raise ValueError("--min_successes cannot be greater than --num_trials")

    missing_records = [
        int(ep) for ep in episode_table["episode_index"].tolist() if int(ep) not in episode_records
    ]
    if missing_records:
        raise ValueError(
            f"Missing garment_info entries for episodes: {missing_records[:10]}"
        )

    requested_device = getattr(args, "device", None) or "cuda"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            f"Requested device '{requested_device}' is unavailable; falling back to cpu."
        )
        requested_device = "cpu"

    env_cfg = parse_env_cfg(args.task, device=requested_device)
    env_cfg.sim.use_fabric = False
    if hasattr(env_cfg, "use_random_seed"):
        env_cfg.use_random_seed = False
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = 42
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
        env_cfg.sim.seed = 42
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    first_record = episode_records[int(episode_table.iloc[0]["episode_index"])]
    current_garment_name = first_record["garment_name"]
    env_cfg.garment_name = current_garment_name
    env_cfg.garment_version = args.garment_version

    logger.info(f"Creating environment with initial garment: {current_garment_name}")
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    output_root = Path(args.output_root).resolve() if args.output_root else None
    report_root = Path(args.report_root).resolve()
    replay_video_root = None
    if args.save_replay_videos:
        replay_video_root = (
            Path(args.replay_video_dir).resolve()
            if args.replay_video_dir
            else (report_root / "replay_videos")
        ) / dataset_root.name
        replay_video_root.mkdir(parents=True, exist_ok=True)
    saved_replay_videos = 0
    max_replay_videos = int(args.max_replay_videos)
    filtered_dataset = _create_output_dataset(
        source_dataset=source_dataset,
        dataset_root=dataset_root,
        output_root=output_root,
        disable_depth=args.disable_depth,
    )
    filtered_json_path = (
        filtered_dataset.root / "meta" / "garment_info.json"
        if filtered_dataset is not None
        else None
    )

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "total_selected_episodes": int(len(episode_table)),
        "num_trials": int(args.num_trials),
        "min_successes": int(args.min_successes),
        "valid_episode_indices": [],
        "invalid_episode_indices": [],
        "invalid_config_episode_indices": [],
        "invalid_config_details": [],
        "trial_results": [],
        "saved_replay_videos": 0,
        "max_replay_videos": max_replay_videos,
        "per_garment": defaultdict(
            lambda: {"valid": [], "invalid": [], "invalid_config": []}
        ),
    }

    rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None

    try:
        for row in episode_table.itertuples(index=False):
            episode_index = int(row.episode_index)
            garment_record = episode_records[episode_index]
            garment_name = garment_record["garment_name"]
            initial_pose = garment_record.get("object_initial_pose")

            if garment_name != current_garment_name:
                logger.info(
                    f"Switching garment: {current_garment_name} -> {garment_name}"
                )
                try:
                    maybe_new_env = _switch_to_garment(
                        env, env_cfg, args.task, garment_name, args.garment_version
                    )
                    env = maybe_new_env
                except Exception as exc:
                    reason = f"switch_garment failed: {type(exc).__name__}: {exc}"
                    logger.warning(
                        f"Skipping episode {episode_index} for {garment_name}: {reason}"
                    )
                    _record_invalid_config(report, garment_name, episode_index, reason)
                    env_cfg.garment_name = garment_name
                    current_garment_name = garment_name
                    continue
                env_cfg.garment_name = garment_name
                current_garment_name = garment_name

            logger.info(
                f"Replaying episode {episode_index} for {garment_name} "
                f"(frames {row.dataset_from_index}:{row.dataset_to_index})"
            )

            trial_outcomes: list[bool] = []
            invalid_config_reason = None
            required_successes = args.min_successes

            for trial_idx in range(args.num_trials):
                logger.info(
                    f"Episode {episode_index} trial {trial_idx + 1}/{args.num_trials}"
                )
                trial_frames = None
                trial_frame_steps = None
                try:
                    if args.switch_each_trial:
                        logger.info(
                            f"Episode {episode_index} trial {trial_idx + 1}/{args.num_trials}: "
                            f"rebuilding garment object via switch_garment({garment_name})"
                        )
                        maybe_new_env = _switch_to_garment(
                            env, env_cfg, args.task, garment_name, args.garment_version
                        )
                        env = maybe_new_env
                    env.reset()
                    if initial_pose is not None:
                        env.set_all_pose({"Garment": initial_pose})
                    stabilize_garment_after_reset(env, args)
                    if replay_video_root is not None:
                        initial_obs = env._get_observations()
                        replay_image_keys = [
                            key for key in initial_obs.keys() if key == "observation.images.top_rgb"
                        ]
                        trial_frames = {key: [] for key in replay_image_keys}
                        trial_frame_steps = {key: [] for key in replay_image_keys}
                        for key in replay_image_keys:
                            trial_frames[key].append(initial_obs[key].copy())
                            trial_frame_steps[key].append(0)
                except Exception as exc:
                    invalid_config_reason = f"reset/setup failed: {type(exc).__name__}: {exc}"
                    logger.warning(
                        f"Skipping episode {episode_index} for {garment_name}: {invalid_config_reason}"
                    )
                    _record_invalid_config(report, garment_name, episode_index, invalid_config_reason)
                    break

                trial_success = False
                for frame_idx in range(int(row.dataset_from_index), int(row.dataset_to_index)):
                    try:
                        if rate_limiter:
                            rate_limiter.sleep(env)
                        action_np = source_dataset.hf_dataset[frame_idx]["action"]
                        action = torch.as_tensor(action_np, device=env.device).unsqueeze(0)
                        env.step(action)
                        if trial_frames is not None and trial_frame_steps is not None:
                            observations = env._get_observations()
                            for key in trial_frames.keys():
                                trial_frames[key].append(observations[key].copy())
                                trial_frame_steps[key].append(
                                    frame_idx - int(row.dataset_from_index) + 1
                                )
                        if env._get_success().item():
                            trial_success = True
                    except Exception as exc:
                        invalid_config_reason = (
                            f"replay/success_check failed: {type(exc).__name__}: {exc}"
                        )
                        logger.warning(
                            f"Skipping episode {episode_index} for {garment_name}: {invalid_config_reason}"
                        )
                        _record_invalid_config(
                            report, garment_name, episode_index, invalid_config_reason
                        )
                        break

                if invalid_config_reason is not None:
                    break

                trial_outcomes.append(trial_success)
                should_save_replay_video = (
                    replay_video_root is not None
                    and trial_frames is not None
                    and (max_replay_videos == 0 or saved_replay_videos < max_replay_videos)
                )
                if should_save_replay_video:
                    save_videos_from_observations(
                        trial_frames,
                        save_dir=str(replay_video_root),
                        episode_idx=episode_index,
                        success=torch.tensor(trial_success),
                        step_overlays=trial_frame_steps,
                        filename_prefix=f"src_ep{episode_index:04d}_trial{trial_idx + 1}",
                    )
                    saved_replay_videos += 1
                    report["saved_replay_videos"] = saved_replay_videos
                successes_so_far = sum(trial_outcomes)
                remaining_trials = args.num_trials - (trial_idx + 1)
                max_possible_successes = successes_so_far + remaining_trials
                if successes_so_far >= required_successes:
                    logger.info(
                        f"Episode {episode_index} reached success threshold "
                        f"{successes_so_far}/{trial_idx + 1} >= {required_successes}"
                    )
                    break
                if max_possible_successes < required_successes:
                    logger.info(
                        f"Episode {episode_index} cannot reach success threshold anymore: "
                        f"{successes_so_far}+{remaining_trials} < {required_successes}"
                    )
                    break

            if invalid_config_reason is not None:
                continue

            success_count = sum(trial_outcomes)
            success = success_count >= required_successes
            report["trial_results"].append(
                {
                    "episode_index": episode_index,
                    "garment_name": garment_name,
                    "successes": success_count,
                    "trials_run": len(trial_outcomes),
                    "trial_outcomes": trial_outcomes,
                }
            )

            logger.info(
                f"Episode {episode_index} final replay result for {garment_name}: "
                f"{'PASS' if success else 'FAIL'} "
                f"({success_count}/{len(trial_outcomes)} successful trials, "
                f"required {required_successes}/{args.num_trials})"
            )

            if success:
                report["valid_episode_indices"].append(episode_index)
                report["per_garment"][garment_name]["valid"].append(episode_index)
                if filtered_dataset is not None:
                    _export_episode(
                        source_dataset=source_dataset,
                        filtered_dataset=filtered_dataset,
                        frame_start=int(row.dataset_from_index),
                        frame_end=int(row.dataset_to_index),
                        task_description=args.task_description,
                    )
                    append_episode_initial_pose(
                        filtered_json_path,
                        len(report["valid_episode_indices"]) - 1,
                        {"Garment": initial_pose} if initial_pose is not None else None,
                        garment_name=garment_name,
                        scale=garment_record.get("scale"),
                    )
            else:
                report["invalid_episode_indices"].append(episode_index)
                report["per_garment"][garment_name]["invalid"].append(episode_index)

        report["num_valid"] = len(report["valid_episode_indices"])
        report["num_invalid"] = len(report["invalid_episode_indices"])
        report["num_invalid_config"] = len(report["invalid_config_episode_indices"])
        total = report["total_selected_episodes"]
        report["success_rate"] = report["num_valid"] / total if total else 0.0
        report["per_garment"] = dict(report["per_garment"])

        report_path = report_root / f"{dataset_root.name}_filter_report.json"
        _write_report(report_path, report)
        logger.info(f"Saved filter report to: {report_path}")
        logger.info(
            f"Dataset {dataset_root.name}: kept {report['num_valid']}/{total} episodes"
        )
        return report
    finally:
        if filtered_dataset is not None:
            filtered_dataset.stop_image_writer()
            filtered_dataset.finalize()
        env.close()


def filter_datasets(args) -> None:
    dataset_roots = _resolve_dataset_roots(args)
    logger.info(f"Resolved {len(dataset_roots)} dataset(s) to filter")

    summaries = []
    for dataset_root in dataset_roots:
        summaries.append(filter_single_dataset(args, dataset_root))

    summary_root = Path(args.report_root).resolve()
    summary_path = summary_root / "filter_summary.json"
    summary = {
        "datasets": [
            {
                "dataset_root": item["dataset_root"],
                "total_selected_episodes": item["total_selected_episodes"],
                "num_valid": item["num_valid"],
                "num_invalid": item["num_invalid"],
                "num_invalid_config": item["num_invalid_config"],
                "success_rate": item["success_rate"],
            }
            for item in summaries
        ]
    }
    _write_report(summary_path, summary)
    logger.info(f"Saved global summary to: {summary_path}")
