#!/usr/bin/env bash
# Lean inference environment (venv_infer): just enough to run the TTSRunner and
# decode tokens → waveform. No Whisper/WER, no accelerate/datasets/wandb — that
# keeps it light for demos and the Colab notebook.
#
# Shares the DPO stack's hard parts (NeMo codec + the transformers-after-NeMo
# re-pin + torchcodec ABI match), just without the scoring/training extras.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. scripts/lib/env_common.sh

make_venv venv_infer

echo "=== [1/5] CUDA-matched PyTorch (newest for tag, coherent) ==="
TAG="$(cuda_wheel_tag)"
[[ -n "$TAG" ]] && INDEX="https://download.pytorch.org/whl/${TAG}" || INDEX=""
uv_install_torch "${INDEX}" torch torchaudio
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== [2/5] NeMo codec stack (nemo-toolkit[tts], not a meta-package) ==="
install_codec_stack

echo "=== [3/5] gepard[inference] — re-pins transformers 5.3.0 after NeMo ==="
uv_install_package inference

echo "=== [4/5] torchcodec ABI-match ==="
fix_torchcodec

echo "=== [5/5] Self-heal CUDA after dependency churn ==="
verify_cuda_selfheal "${INDEX}" torch torchaudio

python - <<'PY'
import torch, transformers, soundfile  # noqa: F401
print("ENV READY ✓ (inference)")
print("torch       :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
from nemo.collections.tts.models import AudioCodecModel  # noqa: F401
print("nemo codec import ✓")
PY
echo "🎉 === Inference env ready: source venv_infer/bin/activate === 🎉"
