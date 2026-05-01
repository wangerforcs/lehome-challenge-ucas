"""Dataset processing utilities for augmenting and merging LeRobot datasets."""

import copy
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING, Any
import json
import shutil
import traceback
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from lerobot.datasets.dataset_tools import merge_datasets as lerobot_merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.utils import RobotKinematics, compute_ee_pose_single_arm
from lehome.utils.logger import get_logger
from lehome.utils.record import append_episode_initial_pose

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


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


def _load_episode_records(
    dataset_root: Path, expected_total_episodes: Optional[int] = None
) -> dict[int, dict[str, Any]]:
    garment_info_path = dataset_root / "meta" / "garment_info.json"
    if not garment_info_path.exists():
        raise FileNotFoundError(f"garment_info.json not found: {garment_info_path}")

    with garment_info_path.open("r", encoding="utf-8") as f:
        garment_info = json.load(f)

    episode_records: dict[int, dict[str, Any]] = {}
    duplicate_global_keys = False
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

    if not garment_info:
        raise ValueError(f"No episode records found in {garment_info_path}")

    if not duplicate_global_keys:
        if expected_total_episodes is not None and len(episode_records) != expected_total_episodes:
            logger.warning(
                "garment_info.json uses global episode indices, but count does not match "
                f"meta episodes: garment_info={len(episode_records)}, meta={expected_total_episodes}"
            )
        return episode_records

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
            "garment_info.json appears to use per-garment local indices, but reconstructed "
            f"episode count does not match meta episodes: reconstructed={next_global_idx}, "
            f"meta={expected_total_episodes}"
        )

    logger.warning(
        "Detected repeated local episode indices in garment_info.json. "
        "Reconstructed global episode mapping by iterating garments in file order."
    )
    return sequential_records


def _export_episode(
    source_dataset: LeRobotDataset,
    target_dataset: LeRobotDataset,
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
        key for key in target_dataset.features if key not in auto_managed_keys
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
                value = value.permute(1, 2, 0)
            frame[key] = value
        frame["task"] = item.get("task", task_description)
        target_dataset.add_frame(frame)
    target_dataset.save_episode()


def _normalize_episode_index(raw_index: int, total_episodes: int) -> int:
    index = raw_index
    if index < 0:
        index += total_episodes
    if index < 0 or index >= total_episodes:
        raise IndexError(
            f"Episode index {raw_index} resolves to {index}, out of range for total_episodes={total_episodes}"
        )
    return index


def _resolve_selected_episode_indices(spec: dict[str, Any], total_episodes: int) -> list[int]:
    if "episode_indices" in spec and spec["episode_indices"] is not None:
        resolved = [
            _normalize_episode_index(int(idx), total_episodes)
            for idx in spec["episode_indices"]
        ]
        return resolved

    start_raw = int(spec.get("start_episode", 0))
    end_raw = spec.get("end_episode", total_episodes)
    end_raw = total_episodes if end_raw is None else int(end_raw)

    start = start_raw + total_episodes if start_raw < 0 else start_raw
    end = end_raw + total_episodes if end_raw < 0 else end_raw
    start = max(0, start)
    end = min(total_episodes, end)

    if end < start:
        raise ValueError(
            f"Invalid episode slice for {spec.get('root')}: start={start_raw}, end={end_raw}"
        )
    return list(range(start, end))


def compute_ee_pose_batch(
    solver: RobotKinematics,
    joint_batch: np.ndarray,
    state_unit: str,
    is_bimanual: bool,
) -> np.ndarray:
    """Compute end-effector poses for a batch of joint configurations.

    Returns:
        - Single-arm: shape (N, 8) - [x, y, z, qx, qy, qz, qw, gripper]
        - Dual-arm: shape (N, 16) - [left_8D, right_8D]
    """
    poses = []
    for idx, joints in enumerate(joint_batch):
        joints = np.asarray(joints, dtype=np.float32)
        try:
            if is_bimanual:
                left_joints = joints[:6]
                right_joints = joints[6:12]
                left_pose = compute_ee_pose_single_arm(solver, left_joints, state_unit)
                right_pose = compute_ee_pose_single_arm(
                    solver, right_joints, state_unit
                )
                poses.append(np.concatenate([left_pose, right_pose], axis=0))
            else:
                poses.append(compute_ee_pose_single_arm(solver, joints, state_unit))
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute EE pose for frame {idx} (joints: {joints}): {e}"
            ) from e

    return np.stack(poses, axis=0)


def add_ee_pose_to_parquet(
    parquet_path: Path,
    solver: RobotKinematics,
    state_unit: str,
    is_bimanual: bool,
    output_path: Path,
) -> None:
    """Add end-effector pose columns to a Parquet file."""
    table = pq.read_table(parquet_path)
    if "observation.state" not in table.column_names:
        raise KeyError(f"'observation.state' not found in {parquet_path}")
    if "action" not in table.column_names:
        raise KeyError(f"'action' not found in {parquet_path}")

    pose_dim = 16 if is_bimanual else 8

    obs_joint_batch = np.stack(table["observation.state"].to_pylist(), axis=0)
    obs_ee_pose = compute_ee_pose_batch(
        solver, obs_joint_batch, state_unit, is_bimanual
    )
    obs_ee_col = pa.array(obs_ee_pose.tolist(), type=pa.list_(pa.float32(), pose_dim))

    action_joint_batch = np.stack(table["action"].to_pylist(), axis=0)
    action_ee_pose = compute_ee_pose_batch(
        solver, action_joint_batch, state_unit, is_bimanual
    )
    action_ee_col = pa.array(
        action_ee_pose.tolist(), type=pa.list_(pa.float32(), pose_dim)
    )

    new_table = table.append_column("observation.ee_pose", obs_ee_col)
    new_table = new_table.append_column("action.ee_pose", action_ee_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, output_path)


def update_info_json(meta_path: Path, is_bimanual: bool, overwrite: bool) -> None:
    """Update dataset metadata (info.json) to include ee_pose feature definitions."""
    info_path = meta_path / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)
    feats = info.get("features", {})

    if ("observation.ee_pose" in feats or "action.ee_pose" in feats) and not overwrite:
        raise RuntimeError(
            "ee_pose features already exist; use --overwrite to replace."
        )

    if is_bimanual:
        ee_pose_feature = {
            "dtype": "float32",
            "shape": [16],
            "names": [
                "left_x",
                "left_y",
                "left_z",
                "left_qx",
                "left_qy",
                "left_qz",
                "left_qw",
                "left_gripper",
                "right_x",
                "right_y",
                "right_z",
                "right_qx",
                "right_qy",
                "right_qz",
                "right_qw",
                "right_gripper",
            ],
        }
    else:
        ee_pose_feature = {
            "dtype": "float32",
            "shape": [8],
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        }

    feats["observation.ee_pose"] = ee_pose_feature
    feats["action.ee_pose"] = ee_pose_feature

    info["features"] = feats
    with info_path.open("w") as f:
        json.dump(info, f, indent=4)


def augment_ee_pose(
    dataset_root: Path,
    urdf_path: Path,
    state_unit: str = "rad",
    output_root: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    """Add end-effector pose to existing datasets."""
    dataset_root = dataset_root.resolve()
    urdf_path = urdf_path.resolve()
    output_root = output_root.resolve() if output_root else dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF path not found: {urdf_path}")

    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    joint_names = info["features"]["observation.state"]["names"]
    num_joints = len(joint_names)

    if num_joints == 6:
        is_bimanual = False
        solver_joint_names = joint_names[:5]
        print("✓ Detected single-arm dataset (6 DoF)")
    elif num_joints == 12:
        is_bimanual = True
        solver_joint_names = [n.replace("left_", "") for n in joint_names[:5]]
        print("✓ Detected dual-arm dataset (12 DoF)")
    else:
        raise ValueError(
            f"Unsupported number of joints: {num_joints}. "
            f"Only 6 (single-arm) or 12 (dual-arm) are supported."
        )

    solver = RobotKinematics(
        str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=solver_joint_names,
    )

    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_root}")

    total_files = len(parquet_files)
    print(f"📦 Processing {total_files} parquet file(s)...")

    for idx, src in enumerate(parquet_files, 1):
        rel = src.relative_to(dataset_root)
        dst = output_root / rel

        if dst.exists() and not overwrite:
            raise FileExistsError(
                f"{dst} exists; use --overwrite or set --output_root to new dir."
            )

        print(f"  [{idx}/{total_files}] {src.name}")
        try:
            add_ee_pose_to_parquet(src, solver, state_unit, is_bimanual, dst)
        except Exception as e:
            raise RuntimeError(f"Failed to process {src}: {e}") from e

    if output_root != dataset_root:
        print("📋 Copying meta, videos, and images...")
        for sub in ["meta", "videos", "images"]:
            src_dir = dataset_root / sub
            dst_dir = output_root / sub
            if src_dir.exists():
                if dst_dir.exists() and not overwrite:
                    raise FileExistsError(f"{dst_dir} exists; use --overwrite.")
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    update_info_json(output_root / "meta", is_bimanual, overwrite=overwrite)

    pose_dim = 16 if is_bimanual else 8
    print(f"✅ Done! Added ee_pose features ({pose_dim}D) to dataset.")


def _fix_depth_data_format(dataset_root: Path) -> None:
    """Ensure observation.top_depth has a stable Arrow schema for merging."""
    dataset_root = dataset_root.resolve()
    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))

    if not parquet_files:
        return

    try:
        first_table = pq.read_table(parquet_files[0])
    except Exception as e:
        logger.warning(f"Failed to read parquet file {parquet_files[0]}: {e}")
        return

    if "observation.top_depth" not in first_table.column_names:
        return

    logger.info(
        f"Found observation.top_depth in {dataset_root.name}, "
        f"normalizing depth column schema in {len(parquet_files)} parquet file(s)..."
    )

    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            if "observation.top_depth" not in table.column_names:
                continue

            depth_col = table["observation.top_depth"]
            depth_list = depth_col.to_pylist()

            fixed_list = []
            for item in depth_list:
                if item is None:
                    fixed_list.append(None)
                    continue

                if isinstance(item, np.ndarray):
                    item = item.tolist()

                # Item should be a 2D list: H x W
                if isinstance(item, list):
                    new_rows = []
                    for row in item:
                        if isinstance(row, np.ndarray):
                            new_rows.append(row.astype(np.float32).tolist())
                        elif isinstance(row, list):
                            new_rows.append([float(v) for v in row])
                        else:
                            # Unexpected format, convert to float list
                            new_rows.append([float(row)])
                    fixed_list.append(new_rows)
                else:
                    # Fallback: scalar/1D, convert to single-row list
                    fixed_list.append([[float(item)]])

            # Infer H, W from first non-None item
            height = width = None
            for item in fixed_list:
                if item is not None and isinstance(item, list) and len(item) > 0:
                    height = len(item)
                    width = len(item[0]) if isinstance(item[0], list) else None
                    break

            if height is None or width is None:
                logger.warning(f"Skip depth normalization for {pf}: cannot infer shape.")
                continue

            # Auto-detect dtype from first non-None item
            sample_value = None
            for item in fixed_list:
                if item is not None and isinstance(item, list) and len(item) > 0:
                    if isinstance(item[0], list) and len(item[0]) > 0:
                        sample_value = item[0][0]
                        break
            
            # Determine Arrow type based on sample value
            if sample_value is not None and isinstance(sample_value, (int, np.integer)):
                depth_type = pa.list_(pa.list_(pa.uint16(), width), height)
            else:
                depth_type = pa.list_(pa.list_(pa.float32(), width), height)
            
            new_depth_array = pa.array(fixed_list, type=depth_type)

            col_idx = table.column_names.index("observation.top_depth")
            table = table.remove_column(col_idx)
            table = table.add_column(col_idx, "observation.top_depth", new_depth_array)

            pq.write_table(table, pf)
        except Exception as e:
            logger.warning(f"Failed to normalize depth column in {pf}: {e}")
            continue

    logger.info(f"Depth column normalization completed for {dataset_root.name}.")


def merge_datasets(
    source_roots: List[Path],
    output_root: Path,
    output_repo_id: str = "merged_dataset",
    merge_custom_meta: bool = True,
) -> None:
    """Merge multiple LeRobot datasets including custom meta files.

    Args:
        source_roots: List of source dataset root directories
        output_root: Output dataset root directory
        output_repo_id: Repository ID for the merged dataset
        merge_custom_meta: Whether to merge custom meta files (garment_info.json)
    """
    # Validate source datasets
    for source_root in source_roots:
        if not source_root.exists():
            raise ValueError(f"Source dataset not found: {source_root}")
        if not (source_root / "meta").exists():
            raise ValueError(f"Meta directory not found in {source_root}")

    logger.info(f"Merging {len(source_roots)} datasets:")
    for i, root in enumerate(source_roots, 1):
        logger.info(f"  {i}. {root}")
    logger.info(f"Output: {output_root}")

    # Normalize depth column schema if needed (to avoid ArrowTypeError)
    for source_root in source_roots:
        try:
            _fix_depth_data_format(source_root)
        except Exception as e:
            logger.warning(f"Depth format normalization failed for {source_root}: {e}")

    # Load all source datasets
    datasets = []
    for source_root in source_roots:
        repo_id = source_root.name
        try:
            dataset = LeRobotDataset(repo_id=repo_id, root=source_root)
            datasets.append(dataset)
            logger.info(
                f"Loaded dataset: {repo_id} ({dataset.meta.total_episodes} episodes) from {dataset.root}"
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {repo_id}: {e}")
            logger.error(f"  Source root: {source_root}")
            traceback.print_exc()
            raise

    # Merge datasets
    logger.info("Starting dataset merge...")
    merged_dataset = lerobot_merge_datasets(
        datasets=datasets,
        output_repo_id=output_repo_id,
        output_dir=output_root,
    )

    logger.info(f"Merged dataset created:")
    logger.info(f"  Total episodes: {merged_dataset.meta.total_episodes}")
    logger.info(f"  Total frames: {merged_dataset.meta.total_frames}")
    logger.info(f"  Location: {output_root}")

    # Merge custom meta files
    if merge_custom_meta:
        logger.info("Merging custom meta files...")
        merge_garment_info(source_roots, output_root)
        logger.info("Custom meta files merged successfully")

    logger.info("Dataset merge completed!")


def subset_merge_datasets(
    source_specs: List[dict[str, Any]],
    output_root: Path,
    output_repo_id: str = "subset_merged_dataset",
) -> Path:
    """Compose a new dataset from selected episode subsets of source datasets."""
    if not source_specs:
        raise ValueError("source_specs must not be empty")

    normalized_specs: list[dict[str, Any]] = []
    for spec in source_specs:
        if not isinstance(spec, dict):
            raise TypeError(f"Each source spec must be a dict, got: {type(spec)}")
        if "root" not in spec:
            raise ValueError(f"Each source spec must include 'root': {spec}")
        normalized = copy.deepcopy(spec)
        normalized["root"] = Path(normalized["root"]).resolve()
        normalized_specs.append(normalized)

    first_root = normalized_specs[0]["root"]
    if not (first_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Dataset meta/info.json not found: {first_root}")

    first_dataset = LeRobotDataset(repo_id=first_root.name, root=first_root)
    features = dict(first_dataset.meta.features)
    fps = first_dataset.fps

    output_root = output_root.resolve()
    if output_root.exists():
        logger.warning(f"Output path already exists, removing: {output_root}")
        shutil.rmtree(output_root)

    logger.info(f"Creating subset-merged dataset at: {output_root}")
    target_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        root=output_root,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )

    manifest: dict[str, Any] = {
        "output_root": str(output_root),
        "output_repo_id": output_repo_id,
        "sources": [],
        "total_exported_episodes": 0,
    }
    garment_info_path = output_root / "meta" / "garment_info.json"
    target_episode_index = 0

    for spec in normalized_specs:
        source_root = spec["root"]
        logger.info(f"Loading source dataset: {source_root}")
        source_dataset = LeRobotDataset(repo_id=source_root.name, root=source_root)

        source_feature_keys = set(source_dataset.meta.features.keys())
        target_feature_keys = set(features.keys())
        if source_feature_keys != target_feature_keys:
            raise ValueError(
                f"Feature mismatch between {first_root} and {source_root}: "
                f"{sorted(source_feature_keys ^ target_feature_keys)}"
            )

        episode_table = _load_episode_table(source_root)
        episode_records = _load_episode_records(
            source_root, expected_total_episodes=source_dataset.num_episodes
        )
        episode_rows = {
            int(row.episode_index): row for row in episode_table.itertuples(index=False)
        }
        selected_indices = _resolve_selected_episode_indices(
            spec, total_episodes=source_dataset.num_episodes
        )
        logger.info(
            f"Selected {len(selected_indices)} episodes from {source_root.name}: {selected_indices}"
        )

        manifest_entry = {
            "root": str(source_root),
            "selected_episode_indices": selected_indices,
        }
        manifest["sources"].append(manifest_entry)

        for source_episode_index in selected_indices:
            if source_episode_index not in episode_rows:
                raise KeyError(
                    f"Episode {source_episode_index} not found in episodes table for {source_root}"
                )
            if source_episode_index not in episode_records:
                raise KeyError(
                    f"Episode {source_episode_index} not found in garment_info.json for {source_root}"
                )

            row = episode_rows[source_episode_index]
            episode_record = episode_records[source_episode_index]
            task_description = "fold the garment on the table"
            source_item = source_dataset[int(row.dataset_from_index)]
            if "task" in source_item:
                task_description = source_item["task"]

            logger.info(
                f"Exporting source episode {source_episode_index} from {source_root.name} "
                f"-> target episode {target_episode_index}"
            )
            _export_episode(
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                frame_start=int(row.dataset_from_index),
                frame_end=int(row.dataset_to_index),
                task_description=task_description,
            )
            append_episode_initial_pose(
                garment_info_path,
                target_episode_index,
                episode_record.get("object_initial_pose"),
                garment_name=episode_record.get("garment_name"),
                scale=episode_record.get("scale"),
            )
            target_episode_index += 1

    manifest["total_exported_episodes"] = target_episode_index
    manifest_path = output_root / "meta" / "subset_merge_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Subset merge completed: {target_episode_index} episodes written to {output_root}"
    )
    logger.info(f"Manifest written to: {manifest_path}")
    return output_root


def merge_garment_info(source_roots: List[Path], output_root: Path) -> int:
    """Merge garment_info.json files from multiple datasets.

    Format:
    {
      "Top_Long_Unseen_0": {
        "0": {"object_initial_pose": [...], "scale": [...]},
        "1": {"object_initial_pose": [...], "scale": [...]}
      }
    }

    Args:
        source_roots: List of source dataset root directories
        output_root: Output dataset root directory

    Returns:
        Total number of episodes merged
    """
    output_path = output_root / "meta" / "garment_info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_data = {}
    total_merged = 0

    for source_root in source_roots:
        source_path = source_root / "meta" / "garment_info.json"

        if not source_path.exists():
            logger.warning(f"garment_info.json not found in {source_root}, skipping...")
            continue

        logger.info(f"Merging garment_info.json from {source_root}")
        count = 0

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                source_data = json.load(f)

            for garment_name, episodes in source_data.items():
                if garment_name not in merged_data:
                    merged_data[garment_name] = {}

                # Each garment's episodes start from 0, appending to existing
                next_idx = len(merged_data[garment_name])
                for episode_key, episode_data in sorted(episodes.items(), key=lambda x: int(x[0])):
                    try:
                        new_key = str(next_idx)
                        merged_data[garment_name][new_key] = episode_data.copy()
                        next_idx += 1
                        count += 1
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid episode key '{episode_key}' in {source_path}: {e}"
                        )
                        continue

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to parse {source_path}: {e}")
            continue

        total_merged += count
        logger.info(f"  Merged {count} episodes from {source_root}")

    # Sort by garment_name and episode indices
    sorted_data = {}
    for garment_name in sorted(merged_data.keys()):
        episodes = merged_data[garment_name]
        sorted_episodes = {
            str(k): episodes[str(k)]
            for k in sorted(int(key) for key in episodes.keys())
        }
        sorted_data[garment_name] = sorted_episodes

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Total merged {total_merged} episodes from {len(sorted_data)} garments to {output_path}"
    )
    return total_merged
