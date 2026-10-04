#!/usr/bin/env bash
# Reproduces /data/envs/torch exactly as built 2026-10-04.
# AMD ROCm 10.0.0 gfx1151 wheels (bundled user-space ROCm runtime) + HF training stack.
set -euo pipefail
mkdir -p /data/envs && cd /data/envs
uv venv --python /usr/bin/python3.14 torch
uv pip install --python /data/envs/torch/bin/python \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" "torchvision==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0" \
  --index-url https://stable.repo.amd.com/rocm/whl-next/
# unsafe-best-match is REQUIRED: the AMD index also hosts fsspec (one version only) and uv's
# first-index rule otherwise backtracks datasets/trl to 2020-era releases.
uv pip install --python /data/envs/torch/bin/python \
  --constraint "$(dirname "$0")/torch-constraints.txt" \
  --index https://stable.repo.amd.com/rocm/whl-next/ --index-strategy unsafe-best-match \
  transformers peft trl datasets accelerate
printf '\n# model/dataset cache on the data drive\nexport HF_HOME=/data/hf-cache\n' >> /data/envs/torch/bin/activate
