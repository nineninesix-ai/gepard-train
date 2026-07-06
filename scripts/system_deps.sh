#!/usr/bin/env bash
# System-level dependencies (apt) — the part no Python packager can do.
# Opt-in: run once per machine before the venv setups, on hosts that need it
# (e.g. fresh AWS GPU instances). Idempotent: checks first, installs only what's
# missing, and does nothing (no apt-get update, no sudo action) when all present.
#
#   make system-deps        # or: sudo scripts/system_deps.sh
#
# Provides: nvidia-cuda-toolkit (nvcc — flash-attn build / codec), python3.12
# venv/dev headers, git-lfs, curl (the uv installer fetches over curl), and ffmpeg
# (torchcodec needs FFmpeg 4-8 shared libs; the DLAMI ships none — see
# docs/AWS_nanocodec_deps.md). uv can provision its own Python 3.12, so python3.12-*
# is a belt-and-suspenders fallback for host-python venvs.

set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "INFO: not an apt system — install nvidia-cuda-toolkit + python3.12 dev headers manually."
  exit 0
fi

# Is an apt package already installed?
pkg_installed() { dpkg -s "$1" >/dev/null 2>&1; }

WANT=(nvidia-cuda-toolkit python3.12-venv python3.12-dev git-lfs curl ffmpeg)
MISSING=()
for p in "${WANT[@]}"; do
  # nvidia-cuda-toolkit: accept any nvcc already on PATH (may come from a non-apt CUDA).
  if [[ "$p" == "nvidia-cuda-toolkit" ]] && command -v nvcc >/dev/null 2>&1; then
    echo "OK: nvcc present ($(nvcc --version | grep -o 'release [0-9.]*' | head -1)) — skipping ${p}"
    continue
  fi
  if pkg_installed "$p"; then
    echo "OK: ${p} already installed"
  else
    MISSING+=("$p")
  fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo "🎉 === All system deps already present — nothing to do === 🎉"
  exit 0
fi

echo "INFO: missing → ${MISSING[*]}"
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi

echo "=== apt update ==="
${SUDO} apt-get update -y

# python3.12 isn't in the default repos on older Ubuntu (22.04); add deadsnakes if
# we actually need a python3.12-* package and it isn't available.
need_py312=false
for p in "${MISSING[@]}"; do [[ "$p" == python3.12-* ]] && need_py312=true; done
if ${need_py312} && ! apt-cache show python3.12-venv >/dev/null 2>&1; then
  echo "INFO: python3.12 not in default repos — adding deadsnakes PPA."
  ${SUDO} apt-get install -y software-properties-common
  ${SUDO} add-apt-repository -y ppa:deadsnakes/ppa
  ${SUDO} apt-get update -y
fi

echo "=== Installing: ${MISSING[*]} ==="
${SUDO} apt-get install -y "${MISSING[@]}"

echo "🎉 === System deps ready === 🎉"
