# OpenPI PI0.5 on LeHome

This project can now evaluate checkpoints trained with official `openpi` directly inside LeHome simulation.

## 0. Modified files

This integration touched the following files.

### In `openpi`

- `/home/wzb/vla/openpi/src/openpi/policies/lehome_policy.py`
- `/home/wzb/vla/openpi/src/openpi/training/config.py`
- `/home/wzb/vla/openpi/src/openpi/training/data_loader.py`
- `/home/wzb/vla/openpi/examples/lehome/README.md`

### In `lehome-challenge`

- `/home/wzb/challenges/lehome-challenge/scripts/eval_policy/openpi_policy.py`
- `/home/wzb/challenges/lehome-challenge/scripts/eval_policy/__init__.py`
- `/home/wzb/challenges/lehome-challenge/scripts/utils/parser.py`
- `/home/wzb/challenges/lehome-challenge/scripts/utils/evaluation.py`
- `/home/wzb/challenges/lehome-challenge/docs/openpi_lehome.md`

## 1. Train or export the checkpoint in `openpi`

The new config name is `pi05_lehome_top_short`.

Dataset assumptions:

- local dataset root: `/data/datasets/datasets-hf/dataset_challenge_merged/top_short_merged`
- observation keys: `observation.state`, `observation.images.top_rgb`, `observation.images.left_rgb`, `observation.images.right_rgb`
- action key: `action`
- task prompt comes from the dataset `task` field

Useful commands from the `openpi` repo:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_lehome_top_short
uv run scripts/train.py pi05_lehome_top_short --exp-name top_short_pi05 --overwrite
```

If you need inference from a converted PyTorch checkpoint, point LeHome at the checkpoint directory containing `model.safetensors`.

## 2. Run LeHome evaluation with the openpi adapter

From the `lehome-challenge` repo:

```bash
python -m scripts.eval \
  --policy_type openpi \
  --policy_path /path/to/openpi/checkpoint_dir \
  --openpi_config_name pi05_lehome_top_short \
  --openpi_repo_root /home/wzb/vla/openpi \
  --task_description "fold the garment on the table" \
  --garment_type top_short \
  --num_episodes 5
```

Notes:

- `--policy_path` should point to an `openpi` checkpoint directory, not a LeRobot `pretrained_model` directory.
- The adapter maps LeHome sim observations to the `openpi` format as:
  - `top_rgb -> base_0_rgb`
  - `left_rgb -> left_wrist_0_rgb`
  - `right_rgb -> right_wrist_0_rgb`
- Returned `openpi` actions are sliced back to the 12-dim joint command used by LeHome.
