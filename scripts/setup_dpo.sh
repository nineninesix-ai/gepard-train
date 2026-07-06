#!/usr/bin/env bash
# DPO data-pipeline environment (venv_dpo): NeMo nano-codec + Whisper ASR + WER.
# Install order matters (see below); orchestration in scripts/lib/env_common.sh.
#
#   1. CUDA-matched torch/torchaudio (newest wheel the driver can run)
#   2. nemo-toolkit[tts]  (NeMo codec stack; pins an older transformers)
#   3. gepard[dpo]     (re-pins transformers==5.3.0 LAST, after NeMo)
#   4. torchcodec ABI-match, then self-heal torch if churn broke it
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. scripts/lib/env_common.sh

make_venv venv_dpo

echo "=== [1/5] CUDA-matched PyTorch (newest for tag, coherent) ==="
TAG="$(cuda_wheel_tag)"
[[ -n "$TAG" ]] && INDEX="https://download.pytorch.org/whl/${TAG}" || INDEX=""
uv_install_torch "${INDEX}" torch torchaudio
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== [2/5] NeMo codec stack (nemo-toolkit[tts], not a meta-package) ==="
install_codec_stack

echo "=== [3/5] gepard[dpo] — re-pins transformers 5.3.0 after NeMo + scoring deps ==="
uv_install_package dpo

echo "=== [4/5] torchcodec ABI-match ==="
fix_torchcodec

echo "=== [5/5] Self-heal CUDA after dependency churn ==="
verify_cuda_selfheal "${INDEX}" torch torchaudio

python - <<'PY'
import torch, torchaudio, transformers, soundfile, jiwer  # noqa: F401
import importlib.metadata as md
print("ENV READY ✓ (dpo)")
print("torch       :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("nemo-toolkit:", md.version("nemo-toolkit"))
from nemo.collections.tts.models import AudioCodecModel  # noqa: F401  (heavy import = real smoke test)
print("nemo codec import ✓")
PY
echo "🎉 === DPO env ready: source venv_dpo/bin/activate === 🎉"
