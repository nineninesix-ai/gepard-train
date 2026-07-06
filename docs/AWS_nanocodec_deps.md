# AWS DLAMI — torchcodec / NanoCodec native-dependency fix

**Scope:** why `make setup_inference` (and `make setup_dpo`) fail at the
`torchcodec ABI-match` step **on this specific AWS image**, what the real root
cause is, and exactly how to fix an environment by hand.

This is an AMI quirk, not a bug in our code. On a normal CUDA box the same setup
just works.

---

## 1. The affected image

```
AMI Name : Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04)
OS       : Ubuntu 24.04.4 LTS (Noble)
GPU      : NVIDIA RTX PRO 6000 Blackwell Server Edition
Driver   : 595.71.05  →  CUDA 13.2 (driver level)
```

Two properties of this DLAMI combine to break `torchcodec`:

1. **The driver is CUDA 13.x (Blackwell), but the apt CUDA libraries are stock
   Ubuntu Noble — CUDA *12.0*.** In particular the NVIDIA Performance Primitives
   (NPP) shipped by `nvidia-cuda-toolkit` are:

   ```
   libnppicc12  12.0.1.104~12.0.1-4build4   →  /usr/lib/x86_64-linux-gnu/libnppicc.so.12
   ```

   That is a *very* old NPP (CUDA 12.0). It is missing symbols that newer
   torchcodec builds link against.

2. **The DLAMI ships no FFmpeg.** There is not a single `libavutil.so.*` on the
   box. `torchcodec` requires FFmpeg 4–8 shared libraries to load at all.

Our stack lands on `torch 2.x+cu128` (see `cuda_wheel_tag` in
`scripts/lib/env_common.sh`; the driver reports CUDA ≥ 13 → we pick the portable
`cu128` wheels). NanoCodec (NeMo `nemo-toolkit[tts]`) and `transformers`' ASR
pipeline both hard-import `torchcodec`, so if `torchcodec` cannot load, setup
aborts.

---

## 2. The symptom

`make setup_inference` dies in `fix_torchcodec`:

```
=== [4/5] torchcodec ABI-match ===
...
RuntimeError: Could not load libtorchcodec.
    OSError: libnppicc.so.13: cannot open shared object file: No such file or directory
ERROR: no working torchcodec
make: *** [Makefile:112: setup_inference] Error 1
```

The reported `libnppicc.so.13` message is **misleading**. The dynamic loader
stops at the *first* dependency it cannot resolve, and `libnppicc` happens to be
listed before FFmpeg in the ELF `DT_NEEDED` order — so an FFmpeg-missing box and
an NPP-missing box produce the *same* first error. There are actually **three**
independent problems stacked on top of each other.

---

## 3. Root cause — three layered problems

Peel them one at a time (`ldd`, then the real import error):

### (A) Wrong NPP *major* version
A bare `pip install torchcodec` (what NeMo pulls, and what the old
`fix_torchcodec` trial-loop installed from PyPI) grabs a build linked against
**CUDA 13** NPP:

```
$ ldd .../torchcodec/libtorchcodec_core8.so | grep npp
        libnppicc.so.13 => not found        # ← CUDA 13; our box only has .so.12
```

Our `torch` is `+cu128` (CUDA 12). Mixing a CUDA-13-linked torchcodec with a
CUDA-12 torch is the wrong ABI. **Fix:** install the torchcodec built for the
*same* CUDA major as torch, from the PyTorch wheel index (`+cu128` → links
`libnppicc.so.12`).

### (B) The system NPP (`.so.12`) is too old
Once you use the `cu128` torchcodec it links `libnppicc.so.12` — which *exists*
on the box — but loading still fails with:

```
OSError: .../libtorchcodec_core6.so: undefined symbol:
         nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx, version libnppicc.so.12
```

The apt NPP (12.0.1.104) predates that symbol. **Fix:** install a newer CUDA-12
NPP from PyPI: `nvidia-npp-cu12` (12.4.x has the symbol).

### (C) The new NPP is not on the loader search path
`torchcodec`'s `libtorchcodec_core*.so` carries **no `RPATH`/`RUNPATH`**, and
`torch`'s own `RPATH` lists `nvidia/cublas`, `nvidia/cudnn`, … but **not**
`nvidia/npp` (NPP isn't a torch dependency). So even with `nvidia-npp-cu12`
installed into the venv, the loader still resolves `libnppicc.so.12` against the
stale `/usr/lib` copy and fails exactly as in (B). **Fix:** preload the wheel NPP
with `RTLD_GLOBAL` at interpreter startup (see §4, step 4) so its symbols win.

### (D) No FFmpeg at all
Independently, the box has no `libavutil.so.*`. **Fix:** `apt install ffmpeg`
(Ubuntu 24.04 → FFmpeg 6.1 → `libavutil.so.58` → torchcodec `core6`).

---

## 4. The fix (per environment)

Run these against the venv that failed (`venv_infer`, and/or `venv_dpo`). Values
below are what worked on 2026-07-02.

```bash
cd ~/gepard-train

# 0. FFmpeg — system-wide, once per machine (now also done by `make system-deps`).
sudo apt-get install -y ffmpeg          # → 6.1.1, libavutil.so.58

source venv_infer/bin/activate          # this venv uses `uv`; `pip` is not on PATH

# 1. torchcodec built for the SAME CUDA major as torch (cu128 → links .so.12).
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    --reinstall --no-deps torchcodec    # → torchcodec 0.11.1+cu128

# 2. A new-enough CUDA-12 NPP (the apt 12.0 one lacks the needed symbol).
uv pip install nvidia-npp-cu12          # → 12.4.1.87

# 3. Preload that NPP at interpreter startup so it beats /usr/lib's stale copy.
#    Written into site-packages so it applies to EVERY launch of this venv's
#    python — no LD_LIBRARY_PATH, independent of how the runner is started.
PURELIB="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cat > "$PURELIB/_gepard_npp_preload.py" <<'PY'
# Preload nvidia-npp-cu12's libs with RTLD_GLOBAL so torchcodec's RPATH-less core
# lib resolves nppi* symbols against them, not /usr/lib's stale libnppicc.so.12.
try:
    import ctypes, glob, os, nvidia
    _d = os.path.join(os.path.dirname(nvidia.__file__), "npp", "lib")
    # libnppc (the NPP core) first: the other libnpp* link against it.
    _libs = sorted(
        glob.glob(os.path.join(_d, "libnpp*.so*")),
        key=lambda p: ("libnppc.so" not in os.path.basename(p), p),
    )
    for _p in _libs:
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
PY
echo "import _gepard_npp_preload" > "$PURELIB/gepard_npp_preload.pth"

# 4. Verify — must import cleanly in a FRESH process (so the .pth actually runs).
python -c "import torchcodec; from torchcodec.decoders import AudioDecoder; \
           print('OK', torchcodec.__version__)"
```

Expected:

```
OK 0.11.1+cu128
```

### Why the preload (not `LD_LIBRARY_PATH`)
`LD_LIBRARY_PATH` only helps processes that inherit it — it breaks the moment the
runner is launched some other way (a wrapper, a notebook kernel, a service). The
site `.pth` runs at *interpreter* startup, before any `import torchcodec`, for
**every** invocation of this venv's python. A Python-level
`os.environ["LD_LIBRARY_PATH"] = ...` would **not** work here: glibc reads the
search path once at exec time, so setting it after startup has no effect on later
`dlopen`s — hence we `ctypes.CDLL(..., RTLD_GLOBAL)` the libraries directly.

---

## 5. What is now automated vs. still manual

| Problem | Fixed automatically? |
|---|---|
| (D) FFmpeg missing | ✅ `make system-deps` now installs `ffmpeg` (only if absent) — `scripts/system_deps.sh` |
| (A) torchcodec CUDA major | ⚠️ **Not yet** — `fix_torchcodec` in `scripts/lib/env_common.sh` still installs torchcodec from **PyPI** (cu13 build) |
| (B) NPP too old | ⚠️ **Not yet** — no `nvidia-npp-cu12` install in setup |
| (C) NPP preload | ⚠️ **Not yet** — `.pth` is currently written only into the live venv (step 3 above) |

> **⚠️ Re-running `make setup_inference` / `make setup_dpo` will re-break a fixed
> venv.** `fix_torchcodec` reinstalls torchcodec from PyPI (the cu13 build), which
> the cu12 preload cannot satisfy. After any re-run of setup, re-apply steps 1–4
> of §4. (Alternatively, fold steps 1–4 into `fix_torchcodec` so setup is
> self-healing — intentionally left out here per request to not modify the
> setup scripts beyond the ffmpeg system dep.)

---

## 6. Quick diagnosis cheatsheet

```bash
# Which NPP major does the installed torchcodec want?
ldd venv_infer/lib/python3.12/site-packages/torchcodec/libtorchcodec_core8.so | grep npp

# Is any FFmpeg present at all?
find / -name 'libavutil.so*' 2>/dev/null ; ffmpeg -version 2>/dev/null | head -1

# What NPP does the system have, and does it have the symbol torchcodec needs?
ls -l /usr/lib/x86_64-linux-gnu/libnppicc.so.*
nm -D venv_infer/lib/python3.12/site-packages/nvidia/npp/lib/libnppicc.so.12 \
   | grep nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx      # present → new enough

# See the REAL per-FFmpeg-version load errors (past the misleading first one):
venv_infer/bin/python -c "import torchcodec" 2>&1 | grep -A11 'FFmpeg version 6:'
```
