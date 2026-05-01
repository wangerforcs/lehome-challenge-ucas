#!/usr/bin/env python3
"""Inspect a LeRobot dataset: parquet data, garment_info, videos, and metadata.

Usage:
    python scripts/dataset_inspect.py <dataset_root> [options]

Options:
    --check-consistency   Check consistency between garment_info and parquet
    --episodes [N ...]    Print per-episode video file paths and time ranges.
                          No args = all episodes; otherwise list specific indices.
"""

import json
import sys
import argparse
from pathlib import Path

import pandas as pd


def check_info_json(root: Path):
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print("  [MISSING] meta/info.json")
        return None
    with open(info_path) as f:
        info = json.load(f)
    print(f"  codebase_version: {info.get('codebase_version')}")
    print(f"  total_episodes:   {info.get('total_episodes')}")
    print(f"  total_frames:     {info.get('total_frames')}")
    print(f"  fps:              {info.get('fps')}")
    print(f"  chunks_size:      {info.get('chunks_size')}")
    splits = info.get("splits", {})
    print(f"  splits:           {splits}")
    features = info.get("features", {})
    print(f"  features ({len(features)}):")
    for name, feat in features.items():
        dtype = feat.get("dtype", "?")
        shape = feat.get("shape", "?")
        print(f"    {name}: dtype={dtype}, shape={shape}")
    return info


def check_parquet_data(root: Path):
    data_dir = root / "data"
    if not data_dir.exists():
        print("  [MISSING] data/ directory")
        return None
    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        print("  [MISSING] no parquet files in data/")
        return None

    print(f"  files ({len(parquet_files)}):")
    all_dfs = []
    for pf in parquet_files:
        rel = pf.relative_to(root)
        try:
            df = pd.read_parquet(pf)
            all_dfs.append(df)
            ep_range = f"{df['episode_index'].min()}-{df['episode_index'].max()}"
            print(f"    {rel}: {len(df)} rows, episodes {ep_range}")
        except Exception as e:
            print(f"    {rel}: ERROR - {e}")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  total rows: {len(combined)}")
    print(f"  episode_index: {combined['episode_index'].min()} to {combined['episode_index'].max()}")
    print(f"  unique episodes: {combined['episode_index'].nunique()}")
    print(f"  columns:")
    for col in combined.columns:
        print(f"    {col}: dtype={combined[col].dtype}")
    return combined


def check_episodes_parquet(root: Path):
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not ep_path.exists():
        print("  [MISSING] meta/episodes/chunk-000/file-000.parquet")
        return None
    df = pd.read_parquet(ep_path)
    print(f"  episodes: {len(df)}")
    print(f"  episode_index: {df['episode_index'].min()} to {df['episode_index'].max()}")
    print(f"  total frames (sum length): {df['length'].sum()}")
    print(f"  length range: {df['length'].min()} - {df['length'].max()}")
    if "dataset_from_index" in df.columns:
        mono = df["dataset_from_index"].is_monotonic_increasing
        print(f"  dataset_from_index monotonic: {mono}")
    return df


def check_garment_info(root: Path):
    gi_path = root / "meta" / "garment_info.json"
    if not gi_path.exists():
        print("  [MISSING] meta/garment_info.json")
        return None
    with open(gi_path) as f:
        garment_info = json.load(f)

    total_episodes = 0
    has_duplicate_keys = False
    garment_summary = []
    for garment_name, episodes in garment_info.items():
        indices = sorted(int(k) for k in episodes.keys())
        count = len(indices)
        total_episodes += count
        expected = list(range(count))
        is_sequential = indices == expected
        if not is_sequential:
            has_duplicate_keys = True
        garment_summary.append((garment_name, indices[0], indices[-1], count, is_sequential))

    print(f"  garments: {len(garment_info)}")
    print(f"  total episodes: {total_episodes}")
    print(f"  per-garment sequential from 0: {not has_duplicate_keys}")
    print(f"  garment list:")
    for name, start, end, count, seq in garment_summary:
        status = "OK" if seq else "NON-SEQ"
        print(f"    {name}: {start}-{end} ({count} eps) [{status}]")
    return garment_info


def check_videos(root: Path):
    videos_dir = root / "videos"
    if not videos_dir.exists():
        print("  [MISSING] videos/ directory")
        return
    camera_dirs = sorted(d for d in videos_dir.iterdir() if d.is_dir())
    if not camera_dirs:
        print("  [EMPTY] videos/ directory")
        return
    print(f"  cameras ({len(camera_dirs)}):")
    for cam_dir in camera_dirs:
        mp4_files = sorted(cam_dir.glob("chunk-*/file-*.mp4"))
        total_size = sum(f.stat().st_size for f in mp4_files)
        size_mb = total_size / (1024 * 1024)
        print(f"    {cam_dir.name}: {len(mp4_files)} files, {size_mb:.1f} MB")


def check_episodes_detail(root: Path, ep_df, fps):
    """Print per-episode video file paths and time ranges."""
    if ep_df is None:
        # Try loading episodes.parquet directly
        ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if ep_path.exists():
            ep_df = pd.read_parquet(ep_path)
        else:
            print("  [SKIP] no episodes.parquet available")
            return

    # Identify video columns: pattern like "videos/<camera_key>/chunk_index|file_index|from_timestamp|to_timestamp"
    video_keys = set()
    for col in ep_df.columns:
        if col.startswith("videos/") and col.endswith("/from_timestamp"):
            # e.g. "videos/observation.images.top_rgb/from_timestamp"
            cam = col[len("videos/"):-len("/from_timestamp")]
            video_keys.add(cam)
    video_keys = sorted(video_keys)

    if not video_keys:
        print("  [SKIP] no video metadata columns in episodes.parquet")
        return

    print(f"  cameras: {video_keys}")
    print()

    for _, row in ep_df.iterrows():
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        time_s = length / fps if fps else 0
        print(f"  Episode {ep_idx} ({length} frames, {time_s:.1f}s @ {fps}fps):")
        for cam in video_keys:
            chunk_idx = int(row[f"videos/{cam}/chunk_index"])
            file_idx = int(row[f"videos/{cam}/file_index"])
            from_ts = float(row[f"videos/{cam}/from_timestamp"])
            to_ts = float(row[f"videos/{cam}/to_timestamp"])
            video_path = f"videos/{cam}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
            print(f"    {cam}:")
            print(f"      file: {video_path}")
            print(f"      time: {from_ts:.3f}s - {to_ts:.3f}s (duration {to_ts - from_ts:.3f}s)")
        print()


def check_consistency(root: Path, data_df, garment_info):
    if data_df is None or garment_info is None:
        print("  [SKIP] missing parquet data or garment_info")
        return

    parquet_eps = set(data_df["episode_index"].unique())

    # Reconstruct global indices from garment_info (same logic as _load_episode_records)
    if garment_info:
        first_key_counts = {}
        for garment_name, episodes in garment_info.items():
            for k in episodes:
                idx = int(k)
                first_key_counts[idx] = first_key_counts.get(idx, 0) + 1
        has_duplicate_keys = any(v > 1 for v in first_key_counts.values())
    else:
        has_duplicate_keys = False

    if has_duplicate_keys:
        # Reconstruct global indices
        reconstructed = {}
        next_idx = 0
        for garment_name, episodes in garment_info.items():
            for ep_key in sorted(episodes, key=lambda x: int(x)):
                reconstructed[next_idx] = (garment_name, int(ep_key))
                next_idx += 1
        gi_global_eps = set(reconstructed.keys())
    else:
        gi_global_eps = set()
        for garment_name, episodes in garment_info.items():
            for k in episodes:
                gi_global_eps.add(int(k))

    # Compare
    print(f"  parquet unique episodes: {len(parquet_eps)}")
    print(f"  garment_info episodes:   {len(gi_global_eps)}")

    only_parquet = parquet_eps - gi_global_eps
    only_gi = gi_global_eps - parquet_eps
    matching = parquet_eps & gi_global_eps

    print(f"  matching:    {len(matching)}")
    if only_parquet:
        sorted_missing = sorted(only_parquet)
        preview = sorted_missing[:10]
        suffix = f" ... ({len(sorted_missing)} total)" if len(sorted_missing) > 10 else ""
        print(f"  in parquet but NOT in garment_info: {preview}{suffix}")
    if only_gi:
        sorted_missing = sorted(only_gi)
        preview = sorted_missing[:10]
        suffix = f" ... ({len(sorted_missing)} total)" if len(sorted_missing) > 10 else ""
        print(f"  in garment_info but NOT in parquet: {preview}{suffix}")

    if not only_parquet and not only_gi:
        print("  CONSISTENT: all episode indices match")
    else:
        print("  INCONSISTENT: mismatches found")

    # Check episodes.parquet if available
    ep_meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if ep_meta_path.exists():
        ep_df = pd.read_parquet(ep_meta_path)
        ep_meta_eps = set(ep_df["episode_index"].unique())
        ep_only = ep_meta_eps - parquet_eps
        parquet_only = parquet_eps - ep_meta_eps
        if ep_only or parquet_only:
            print(f"  episodes.parquet vs data parquet: MISMATCH")
            if ep_only:
                print(f"    in episodes.parquet but not data: {sorted(ep_only)[:10]}")
            if parquet_only:
                print(f"    in data but not episodes.parquet: {sorted(parquet_only)[:10]}")
        else:
            print(f"  episodes.parquet vs data parquet: CONSISTENT")


def main():
    parser = argparse.ArgumentParser(description="Inspect a LeRobot dataset")
    parser.add_argument("dataset_root", type=str, help="Path to dataset root directory")
    parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="Check consistency between garment_info and parquet episode indices",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        metavar="IDX",
        help="Print per-episode video details. No args = all episodes; otherwise list specific indices.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)

    print(f"=== Dataset: {root.name} ===")
    print(f"Path: {root.resolve()}\n")

    print("--- info.json ---")
    info = check_info_json(root)

    print("\n--- parquet data ---")
    data_df = check_parquet_data(root)

    print("\n--- episodes.parquet ---")
    check_episodes_parquet(root)

    print("\n--- garment_info.json ---")
    garment_info = check_garment_info(root)

    print("\n--- videos ---")
    check_videos(root)

    if args.episodes is not None:
        print("\n--- episode details ---")
        ep_df = check_episodes_parquet(root)
        fps = info.get("fps", 30) if info else 30
        if args.episodes:
            # Filter to requested episode indices
            requested = set(int(x) for x in args.episodes)
            ep_df = ep_df[ep_df["episode_index"].isin(requested)]
            if ep_df.empty:
                print(f"  [SKIP] none of the requested episodes found")
            else:
                check_episodes_detail(root, ep_df, fps)
        else:
            check_episodes_detail(root, ep_df, fps)

    if args.check_consistency:
        print("\n--- consistency check ---")
        check_consistency(root, data_df, garment_info)

    print("\nDone!")


if __name__ == "__main__":
    main()
