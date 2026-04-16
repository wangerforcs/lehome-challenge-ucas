#!/usr/bin/env bash

set -euo pipefail

# Fill these two paths before running the script.
# POLICY_ROOT: the uploaded `policies` directory on the local machine.
# REPO_ROOT: the local repository root that should consume those files.
POLICY_ROOT="/path/to/policies"
REPO_ROOT="/path/to/lehome-challenge"

link_path() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "${src}" && ! -L "${src}" ]]; then
    echo "Missing source: ${src}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${dst}")"
  ln -sfn "${src}" "${dst}"
  echo "linked: ${dst} -> ${src}"
}

if [[ "${POLICY_ROOT}" == "/path/to/policies" || "${REPO_ROOT}" == "/path/to/lehome-challenge" ]]; then
  echo "Please edit POLICY_ROOT and REPO_ROOT in policies/setup_local_links.sh before running." >&2
  exit 1
fi

link_path "${POLICY_ROOT}/outputs/train/top_long_best" \
          "${REPO_ROOT}/outputs/train/top_long_best"

link_path "${POLICY_ROOT}/outputs/train/top_short_best" \
          "${REPO_ROOT}/outputs/train/top_short_best"

link_path "${POLICY_ROOT}/outputs/train/pant_long_best" \
          "${REPO_ROOT}/outputs/train/pant_long_best"

link_path "${POLICY_ROOT}/outputs/classifier/garment_classifier_resnet18.pth" \
          "${REPO_ROOT}/outputs/classifier/garment_classifier_resnet18.pth"

link_path "${POLICY_ROOT}/Datasets/example" \
          "${REPO_ROOT}/Datasets/example"

echo "All local links are ready."
