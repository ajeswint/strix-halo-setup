#!/usr/bin/env bash
# Reproduces the torch env as built 2026-10-04 (default target /data/envs/torch).
# AMD ROCm 10.0.0 gfx1151 wheels (bundled user-space ROCm runtime) + HF training stack.
# Usage: install-torch-env.sh [target-dir]
set -euo pipefail
TARGET="${1:-/data/envs/torch}"
HERE="$(cd "$(dirname "$0")" && pwd)"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
AMD=https://stable.repo.amd.com/rocm/whl-next/

"$UV" venv --python /usr/bin/python3.14 "$TARGET"
"$UV" pip install --python "$TARGET/bin/python" \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" "torchvision==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0" \
  --index-url "$AMD"
# unsafe-best-match is REQUIRED: the AMD index also hosts fsspec (one version only) and uv's
# first-index rule otherwise backtracks datasets/trl to 2020-era releases.
"$UV" pip install --python "$TARGET/bin/python" \
  --constraint "$HERE/torch-constraints.txt" \
  --index "$AMD" --index-strategy unsafe-best-match \
  transformers peft trl datasets accelerate
grep -q '^export HF_HOME=' "$TARGET/bin/activate" || \
  printf '\n# model/dataset cache on the data drive\nexport HF_HOME=/data/hf-cache\n' >> "$TARGET/bin/activate"
