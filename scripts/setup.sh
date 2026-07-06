#!/usr/bin/env bash
# Training environment (venv): CUDA-matched torch + gepard[train] + optional flash-attn.
# Orchestration lives in scripts/lib/env_common.sh; the safe deps in pyproject [train].
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root
# shellcheck disable=SC1091
. scripts/lib/env_common.sh

TORCH_VER="2.8.0"; TV_VER="0.23.0"; TA_VER="2.8.0"

make_venv venv

echo "=== [1/3] CUDA-matched PyTorch (pinned ${TORCH_VER}) ==="
TAG="$(cuda_wheel_tag)"
if [[ -z "$TAG" ]]; then
  uv_install_torch "" "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}"
else
  INDEX="https://download.pytorch.org/whl/${TAG}"
  if ! uv_install_torch "${INDEX}" \
        "torch==${TORCH_VER}+${TAG}" "torchvision==${TV_VER}+${TAG}" "torchaudio==${TA_VER}+${TAG}"; then
    echo "WARN: ${TAG} wheels not found — falling back to cu121."
    uv_install_torch "https://download.pytorch.org/whl/cu121" \
      "torch==${TORCH_VER}+cu121" "torchvision==${TV_VER}+cu121" "torchaudio==${TA_VER}+cu121"
  fi
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== [2/3] gepard[train] (transformers 5.3.0 + accelerate + datasets + wandb + CLIs) ==="
uv_install_package train

echo "=== [3/3] Optional: flash-attn 2.8.3 (non-fatal) ==="
ARCH="$(python - <<'PY'
import torch
if torch.cuda.is_available():
    a, b = torch.cuda.get_device_capability(); print(f"{a}.{b}")
PY
)"
[[ -n "${ARCH}" ]] && export TORCH_CUDA_ARCH_LIST="${ARCH}" && echo "INFO: TORCH_CUDA_ARCH_LIST=${ARCH}"
# flash-attn builds from source with --no-build-isolation (so it sees the installed
# torch). A uv venv is minimal, so its build tools must be present first — otherwise
# the build dies with "No module named 'wheel'". nvcc comes from `make system-deps`.
uv pip install wheel setuptools packaging ninja
set +e
uv pip install "flash-attn==2.8.3" --no-build-isolation
[[ $? -ne 0 ]] && echo "WARN: flash-attn build failed — using PyTorch SDPA."
set -e

python - <<'PY'
import torch, transformers, accelerate, datasets, hydra, omegaconf
print("ENV READY ✓ (train)")
print("torch       :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__, "| accelerate:", accelerate.__version__,
      "| datasets:", datasets.__version__)
print("hydra-core  :", hydra.__version__, "| omegaconf:", omegaconf.__version__)
PY
echo "🎉 === Training env ready: source venv/bin/activate === 🎉"
