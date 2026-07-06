#!/usr/bin/env bash
# Shared environment-setup helpers for Gepard's three venvs (train / dpo / inference).
#
# Why bash and not a pure pyproject/lockfile: the hard parts here are inherently
# imperative and machine-specific — (1) detect the driver's CUDA at runtime and
# pick a matching torch wheel index, (2) install NeMo then deliberately RE-PIN
# transformers over NeMo's older bound, (3) trial-match torchcodec against the
# installed torch, (4) self-heal if a dependency silently swapped torch. A static
# resolver cannot express any of these. pyproject stays for packaging + the safe
# declarative extras (see pyproject.toml); this file orchestrates the rest.
#
# Source it after `set -euo pipefail`:  . "$(dirname "$0")/lib/env_common.sh"

# ── uv: the installer. Fast, pip-compatible, and can provision Python 3.12 itself
# (no reliance on the host having the right python). Bootstrap if missing.
ensure_uv() {
  # Make an already-installed uv visible even if it's only under ~/.local/bin
  # (a prior run may have installed it there without re-login).
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    echo "OK: uv $(uv --version 2>/dev/null | awk '{print $2}') at $(command -v uv)"
    return
  fi
  echo "INFO: uv not found — installing (https://astral.sh/uv)."
  # Astral standalone installer (installs to ~/.local/bin), with fallbacks for a
  # bare box that lacks curl. Last resort: pip. No apt package exists for uv.
  if   command -v curl >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then wget -qO- https://astral.sh/uv/install.sh | sh
  else python3 -m pip install --user uv || pip3 install --user uv
  fi
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv install failed. Install it manually and re-run:"
    echo "       https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  }
  echo "OK: uv installed → $(command -v uv)"
}

# Create a Python 3.12 venv with uv (uv fetches a managed 3.12 if the host lacks
# one) and activate it. $1 = venv dir.
make_venv() {
  local dir="$1"
  ensure_uv
  echo "=== Create & activate venv (${dir}, python 3.12) ==="
  uv venv --python 3.12 "${dir}"
  # shellcheck disable=SC1090
  source "${dir}/bin/activate"
  echo "OK: $(python --version) @ ${VIRTUAL_ENV}"
}

# Newest CUDA the driver supports (e.g. "12.8"). Empty if no GPU.
detect_cuda() {
  local ver=""
  if command -v nvidia-smi &>/dev/null; then
    ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -n 1 || true)
  elif command -v nvcc &>/dev/null; then
    ver=$(nvcc --version | grep -o 'release [0-9]\+\.[0-9]\+' | awk '{print $2}')
  fi
  echo "$ver"
}

# Map driver CUDA → a wheel tag the driver can run. Ceiling is cu128 for ALL three
# envs: it runs on any driver >= 12.8 (incl. CUDA 13.x, Blackwell), is the most
# portable, and — crucially — FSDP/accelerate + dpo-train only ever run in the
# TRAIN venv (torch 2.8.0+cu128), so there is no reason to chase cu130 anywhere.
# The DPO/inference codec venvs land on a newer torch (NeMo needs ~2.11) but still
# on cu128, and never touch distributed training.
cuda_wheel_tag() {
  local ver major minor
  ver="$(detect_cuda)"
  [[ -z "$ver" ]] && { echo ""; return; }
  major="$(echo "$ver" | cut -d. -f1)"
  minor="$(echo "$ver" | cut -d. -f2)"
  if   [[ ${major} -ge 13 ]]; then echo "cu128"
  elif [[ ${major} -eq 12 && ${minor} -ge 8 ]]; then echo "cu128"
  elif [[ ${major} -eq 12 && ${minor} -ge 6 ]]; then echo "cu126"
  else echo "cu121"; fi
}

# Install the torch stack. Args after the index are the specs
# (e.g. "torch==2.8.0+cu128" ... pinned, or bare "torch" "torchaudio" for newest).
# $1 = wheel index url ("" → CPU wheels).
#
# INDEX-ONLY on purpose: the pytorch index self-hosts torch + torchvision +
# torchaudio AND their deps (numpy, sympy, networkx, …) as a coherent set. Adding
# `--extra-index-url pypi` is what breaks it — uv would pull a mismatched generic
# torch from PyPI (e.g. torch 2.12 CPU alongside torchaudio 2.11+cu128). The rest
# of the deps come later from PyPI via `uv pip install -e .[extra]`.
uv_install_torch() {
  local index="$1"; shift
  [[ -z "$index" ]] && index="https://download.pytorch.org/whl/cpu" && echo "WARN: no CUDA — CPU torch (slow)."
  echo "INFO: torch from ${index} (index-only)"
  uv pip install --index-url "${index}" "$@"
}

# NeMo TTS codec stack — install nemo-toolkit[tts] directly (not a meta-package).
# nemo-toolkit pins an OLD transformers; the subsequent
# `uv pip install -e .[dpo|inference]` re-pins transformers==5.3.0 on top. torch/
# torchaudio are already installed (kept — nemo only floors torch>=2.8); scipy /
# librosa ride in the pyproject extras; torchcodec is pulled here and ABI-checked
# by fix_torchcodec.
install_codec_stack() {
  echo "=== NeMo codec stack (nemo-toolkit[tts]==2.4.0) ==="
  uv pip install "nemo-toolkit[tts]==2.4.0"
}

# Re-assert the pinned transformers on top of whatever NeMo installed. Runs as the
# editable-extra install so the pyproject pin (transformers==5.3.0) is the source
# of truth — call this LAST, after the codec stack.  $1 = extra name (dpo|inference).
uv_install_package() {
  local extra="$1"
  echo "=== Install gepard[${extra}] (re-pins transformers==5.3.0 after NeMo) ==="
  uv pip install -e ".[${extra}]"
}

# torchcodec ships per-torch ABI; NeMo only floors it, so pip may grab a
# build that needs a NEWER torch and fails to load. Trial newest→older until the
# shared lib imports. transformers' ASR pipeline imports it, so it must load.
fix_torchcodec() {
  local tc
  for tc in "" "0.13.0" "0.12.0" "0.11.0"; do
    if [[ -n "$tc" ]]; then
      echo "WARN: torchcodec import failed — trying torchcodec==${tc}"
      uv pip install -q "torchcodec==${tc}"
    fi
    if python -c "import torchcodec" 2>/dev/null; then
      echo "OK: torchcodec $(python -c 'import torchcodec; print(torchcodec.__version__)')"
      return 0
    fi
  done
  python -c "import torchcodec" || { echo "ERROR: no working torchcodec"; exit 1; }
}

# Self-heal: if a dependency replaced torch with a build this driver can't run,
# reinstall the CUDA-matched torch. $1 = wheel index, remaining args = specs.
verify_cuda_selfheal() {
  local index="$1"; shift
  set +e
  python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
  local ok=$?
  set -e
  if [[ ${ok} -ne 0 && -n "${index}" ]]; then
    echo "WARN: torch/CUDA broke after dependency churn — reinstalling."
    uv pip install --index-url "${index}" --force-reinstall --no-deps "$@"
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA still broken'"
  fi
}
