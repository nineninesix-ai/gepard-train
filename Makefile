.PHONY: help system-deps setup setup_dpo setup_inference dataset train resume finetune finetune-resume merge upload clean test check-env login \
        dpo-sample dpo-sample-sharded dpo-score dpo-pairs dpo-dataset dpo-train \
        test-baseline test-baseline-real

# Hydra overrides for the DPO pipeline (e.g. make dpo-sample DPO="run_name=round4 dpo/sampling.checkpoint=<ckpt>")
DPO ?=

# Use bash instead of sh for source command
SHELL := /bin/bash

# Gepard palette — cheetah coat (sand / rust / orange), grass green, red
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
ORANGE := \033[38;2;245;129;12m      # rusty orange (cheetah coat)
RUST   := \033[38;2;178;58;12m       # deep rust (spots)
SAND   := \033[38;2;224;178;92m      # tan fur
GRASS  := \033[38;2;104;159;56m      # grass green
BOLD   := \033[1m
RESET  := \033[0m
# Legacy aliases — remapped to the Gepard palette so existing help lines recolor
# without touching each one (cyan→sand, blue→rusty orange).
CYAN   := $(SAND)
BLUE   := $(ORANGE)

define BANNER
	@printf "\n"
	@printf "  $(ORANGE)$(BOLD) ██████╗ ███████╗██████╗  █████╗ ██████╗ ██████╗ $(RESET)\n"
	@printf "  $(ORANGE)$(BOLD)██╔════╝ ██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗$(RESET)\n"
	@printf "  $(ORANGE)$(BOLD)██║  ███╗█████╗  ██████╔╝███████║██████╔╝██║  ██║$(RESET)\n"
	@printf "  $(ORANGE)$(BOLD)██║   ██║██╔══╝  ██╔═══╝ ██╔══██║██╔══██╗██║  ██║$(RESET)\n"
	@printf "  $(ORANGE)$(BOLD)╚██████╔╝███████╗██║     ██║  ██║██║  ██║██████╔╝$(RESET)\n"
	@printf "  $(ORANGE)$(BOLD) ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ $(RESET)\n"
	@printf "\n\n"
endef

# Ephemeral NVMe cache (optional): make dataset NVME=/opt/dlami/nvme
NVME ?=
HF_HOME_ARG  = $(if $(NVME),HF_HOME=$(NVME)/.cache/huggingface,)
# Read token from current HF_HOME (env var), fallback to default location
HF_TOKEN_VAL = $(shell cat "$${HF_HOME:-$$HOME/.cache/huggingface}/token" 2>/dev/null || echo "")
HF_TOKEN_ARG = $(if $(HF_TOKEN_VAL),HF_TOKEN=$(HF_TOKEN_VAL),)
HF_ENV       = $(HF_HOME_ARG) $(HF_TOKEN_ARG)

# Default target
help:
	$(BANNER)
	@printf "\n"
	@printf "$(BOLD)$(GREEN)╔══════════════════════════════════════════════════╗$(RESET)\n"
	@printf "$(BOLD)$(GREEN)║                  NineNineSix AI                  ║$(RESET)\n"
	@printf "$(BOLD)$(GREEN)╚══════════════════════════════════════════════════╝$(RESET)\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)Available commands:$(RESET)\n"
	@printf "  $(GREEN)make login$(RESET)              - Authenticate HuggingFace & Weights & Biases\n"
	@printf "  $(GREEN)make system-deps$(RESET)        - Install apt deps (nvidia-cuda-toolkit + python3.12) — sudo, run once\n"
	@printf "  $(GREEN)make setup$(RESET)              - Setup training env (venv: torch + transformers + accelerate)\n"
	@printf "  $(GREEN)make setup_dpo$(RESET)          - Setup DPO env (venv_dpo: codec + whisper + wer)\n"
	@printf "  $(GREEN)make setup_inference$(RESET)    - Setup lean inference env (venv_infer: codec + runner)\n"
	@printf "  $(GREEN)make dataset$(RESET)            - Prepare training dataset from HuggingFace\n"
	@printf "  $(GREEN)make train$(RESET)              - Start pretraining with Flash Attention 2 (4 GPU)\n"
	@printf "  $(GREEN)make resume$(RESET) CHECKPOINT=path - Resume pretraining from checkpoint\n"
	@printf "  $(GREEN)make finetune$(RESET)           - LoRA short-phrase fine-tune: backbone adapters only (1 GPU)\n"
	@printf "  $(GREEN)make finetune-resume$(RESET) CHECKPOINT=path - Resume LoRA fine-tune from checkpoint\n"
	@printf "  $(GREEN)make merge$(RESET) CHECKPOINT=path   - Fold LoRA adapters of a checkpoint-N into base weights (CPU)\n"
	@printf "  $(GREEN)make upload$(RESET) REPO=user/model CHECKPOINT=path - Upload model to HF Hub (bf16, no optimizer)\n"
	@printf "  $(GREEN)make upload$(RESET) ... $(YELLOW)TRAINING_STATE=1$(RESET)   - Resume snapshot: + optimizer state, weights stay fp32\n"
	@printf "  $(GREEN)make clean$(RESET)              - Remove generated files (checkpoints, datasets, cache)\n"
	@printf "  $(GREEN)make test$(RESET)               - Test Flash Attention installation\n"
	@printf "  $(GREEN)make check-env$(RESET)          - Check environment and dependencies\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)DPO pipeline (Phase 3, config: conf/dpo.yaml):$(RESET)\n"
	@printf "  $(GREEN)make dpo-sample$(RESET)         - 1. Batched rollouts, saves tokens (venv_dpo)\n"
	@printf "  $(GREEN)make dpo-sample-sharded$(RESET) - 1. Same, N parallel processes (SHARDS=4)\n"
	@printf "  $(GREEN)make dpo-progress$(RESET)       - 1. Live progress bar over all shards (2nd terminal)\n"
	@printf "  $(GREEN)make dpo-score$(RESET)          - 2. Codec decode + Whisper + reward (venv_dpo)\n"
	@printf "  $(GREEN)make dpo-pairs$(RESET)          - 3. Preference pairs + ref logprobs (venv)\n"
	@printf "  $(GREEN)make dpo-dataset$(RESET)        - Stages 1-3 in sequence\n"
	@printf "  $(GREEN)make dpo-train$(RESET)          - 4. DPO training: LoRA + stop_head (venv)\n"
	@printf "  Hydra overrides: $(YELLOW)make dpo-sample DPO=\"run_name=round4\"$(RESET)\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)NVMe cache (ephemeral machines):$(RESET)\n"
	@printf "  Set $(BOLD)NVME$(RESET) to redirect HF cache to fast ephemeral storage.\n"
	@printf "  Token is read automatically from ~/.cache/huggingface/token\n"
	@printf "  $(GREEN)make dataset$(RESET) $(YELLOW)NVME=/opt/dlami/nvme$(RESET)\n"
	@printf "\n"
	@printf "$(BOLD)$(BLUE)Examples:$(RESET)\n"
	@printf "  $(CYAN)make train$(RESET)\n"
	@printf "  $(CYAN)make dataset NVME=/opt/dlami/nvme$(RESET)\n"
	@printf "  $(CYAN)make resume CHECKPOINT=./checkpoints/checkpoint-5000$(RESET)\n"
	@printf "  $(CYAN)make upload REPO=username/model-name CHECKPOINT=./checkpoints/checkpoint-5000$(RESET)\n"
	@printf "\n"

# System-level apt deps (nvidia-cuda-toolkit + python3.12 headers). Opt-in, sudo,
# run once per machine (e.g. fresh AWS GPU box) BEFORE the venv setups.
system-deps:
	@printf "🔧 Installing system deps (apt, sudo)...\n"
	@sudo bash scripts/system_deps.sh

# Setup training environment (venv). User-owned; run `make system-deps` first if needed.
setup:
	@printf "🔧 Setting up training environment (venv)...\n"
	@bash scripts/setup.sh
	@printf "✅ Training env ready! Activate: source venv/bin/activate\n"

# Setup DPO data-pipeline environment (venv_dpo: NeMo codec + whisper + wer)
setup_dpo:
	@printf "🔧 Setting up DPO environment (venv_dpo)...\n"
	@bash scripts/setup_dpo.sh
	@printf "✅ DPO environment ready! Activate: source venv_dpo/bin/activate\n"

# Setup lean inference environment (venv_infer: codec + runner, no whisper/train)
setup_inference:
	@printf "🔧 Setting up inference environment (venv_infer)...\n"
	@bash scripts/setup_inference.sh
	@printf "✅ Inference env ready! Activate: source venv_infer/bin/activate\n"

login:
	@printf "\n"
	@printf "$(BOLD)$(CYAN)╔══════════════════════════════════════════════════╗$(RESET)\n"
	@printf "$(BOLD)$(CYAN)║              Authentication Setup                ║$(RESET)\n"
	@printf "$(BOLD)$(CYAN)╚══════════════════════════════════════════════════╝$(RESET)\n"
	@printf "\n"
	@if [ ! -d "venv" ]; then \
		printf "$(RED)$(BOLD)✗ Error:$(RESET) $(RED)Virtual environment not found (venv/).$(RESET)\n"; \
		printf "$(YELLOW)  Run '$(BOLD)make setup$(RESET)$(YELLOW)' first.$(RESET)\n"; \
		printf "\n"; \
		exit 1; \
	fi
	@printf "$(BLUE)━━━ Step 1/2: HuggingFace ━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "$(CYAN)  Configuring git credential helper...$(RESET)\n"
	@git config --global credential.helper store
	@printf "$(CYAN)  Launching HuggingFace login...$(RESET)\n"
	@printf "\n"
	@source venv/bin/activate && hf auth login
	@printf "\n"
	@printf "$(GREEN)  ✓ HuggingFace authentication done$(RESET)\n"
	@printf "\n"
	@printf "$(BLUE)━━━ Step 2/2: Weights & Biases ━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "$(YELLOW)  Login to Weights & Biases? [y/N]: $(RESET)"; \
	read -r answer; \
	if [ "$$answer" = "y" ] || [ "$$answer" = "Y" ]; then \
		printf "$(CYAN)  Launching wandb login...$(RESET)\n"; \
		printf "\n"; \
		source venv/bin/activate && wandb login; \
		printf "\n"; \
		printf "$(GREEN)  ✓ Weights & Biases authentication done$(RESET)\n"; \
	else \
		printf "$(YELLOW)  ⚠ Skipped W&B login$(RESET)\n"; \
	fi
	@printf "\n"
	@printf "$(BOLD)$(GREEN)╔══════════════════════════════════════════════════╗$(RESET)\n"
	@printf "$(BOLD)$(GREEN)║            ✓ Authentication complete!            ║$(RESET)\n"
	@printf "$(BOLD)$(GREEN)╚══════════════════════════════════════════════════╝$(RESET)\n"
	@printf "\n"

# Prepare dataset
dataset:
	@printf "📊 Preparing training dataset...\n"
	@source venv/bin/activate && $(HF_ENV) python -m gepard.cli.prepare
	@printf "✅ Dataset prepared (data.train_dataset_path)\n"

# Train model
train:
	@printf "🚀 Starting training with Flash Attention 2...\n"
	@source venv/bin/activate && accelerate launch \
		--config_file accelerate/pretrain_fsdp.yaml \
		-m gepard.cli.train $(EXP)

# LoRA short-phrase fine-tune — frozen model, backbone adapters only, 1 GPU.
# Composes conf/sft.yaml (its own entry, not an experiment preset).
# EXP passes extra Hydra overrides, e.g. an experiment preset:
#   make finetune EXP="experiment=ft_wider_lora"
EXP ?=
finetune:
	@printf "🎯 Starting LoRA short-phrase fine-tune (frozen model, adapters only, 1 GPU)...\n"
	@source venv/bin/activate && accelerate launch \
		--config_file accelerate/finetune_single.yaml \
		-m gepard.cli.sft $(EXP)

# Resume LoRA fine-tune from checkpoint
finetune-resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		printf "❌ Error: Please specify CHECKPOINT=path\n"; \
		printf "Example: make finetune-resume CHECKPOINT=/opt/dlami/nvme/checkpoints-lora/checkpoint-2000\n"; \
		exit 1; \
	fi
	@if [ ! -d "$(CHECKPOINT)" ]; then \
		printf "❌ Error: Checkpoint directory not found: $(CHECKPOINT)\n"; \
		exit 1; \
	fi
	@printf "🔄 Resuming LoRA fine-tune from $(CHECKPOINT)...\n"
	@source venv/bin/activate && accelerate launch \
		--config_file accelerate/finetune_single.yaml \
		-m gepard.cli.sft \
		run.resume_from=$(CHECKPOINT) $(EXP)

# ── DPO pipeline (Phase 3) ──────────────────────────────────────────────
# Stages 1-2 need the codec/Whisper stack → venv_dpo (make setup_dpo).
# Stages 3-4 only need the model stack → training venv.

dpo-sample:
	@printf "🎲 DPO stage 1/4: sampling rollouts (venv_dpo)...\n"
	@source venv_dpo/bin/activate && python -m gepard.data.dpo.sample $(DPO)

# Parallel sampling: N independent processes on one GPU (AR decoding at batch 8
# leaves the GPU mostly idle, so shards scale near-linearly until VRAM/compute
# saturates). Each process needs ~3 GB VRAM. Resume-safe like dpo-sample.
SHARDS ?= 4
dpo-sample-sharded:
	@printf "🎲 DPO stage 1/4: sampling with $(SHARDS) parallel shards...\n"
	@source venv_dpo/bin/activate && \
	trap 'kill 0' INT TERM; \
	for i in $$(seq 0 $$(($(SHARDS)-1))); do \
		python -m gepard.data.dpo.sample --shard $$i/$(SHARDS) $(DPO) \
			> /tmp/dpo_sample_shard_$$i.log 2>&1 & \
	done; \
	printf "  logs:     tail -f /tmp/dpo_sample_shard_*.log\n"; \
	printf "  progress: $(GREEN)make dpo-progress SHARDS=$(SHARDS)$(RESET)  (run in another terminal)\n"; \
	wait
	@printf "✅ All $(SHARDS) shards finished\n"

# Read-only live progress across all sampling shards. Run in a second terminal
# (the sharded launcher holds the first). SHARDS must match dpo-sample-sharded.
dpo-progress:
	@source venv/bin/activate && python -m gepard.data.dpo.progress --shards $(SHARDS) $(DPO)

dpo-score:
	@printf "🎯 DPO stage 2/4: scoring rollouts (venv_dpo)...\n"
	@source venv_dpo/bin/activate && python -m gepard.data.dpo.score $(DPO)

dpo-pairs:
	@printf "🔗 DPO stage 3/4: building preference pairs (venv)...\n"
	@source venv/bin/activate && python -m gepard.data.dpo.pairs $(DPO)

dpo-dataset: dpo-sample dpo-score dpo-pairs
	@printf "✅ DPO dataset ready!\n"

dpo-train:
	@printf "🚀 DPO stage 4/4: training (venv)...\n"
	@source venv/bin/activate && python -m gepard.training.dpo $(DPO)

# Resume training from checkpoint
resume:
	@if [ -z "$(CHECKPOINT)" ]; then \
		printf "❌ Error: Please specify CHECKPOINT=path\n"; \
		printf "Example: make resume CHECKPOINT=./checkpoints/checkpoint-5000\n"; \
		exit 1; \
	fi
	@if [ ! -d "$(CHECKPOINT)" ]; then \
		printf "❌ Error: Checkpoint directory not found: $(CHECKPOINT)\n"; \
		exit 1; \
	fi
	@printf "🔄 Resuming training from $(CHECKPOINT)...\n"
	@source venv/bin/activate && accelerate launch \
		--config_file accelerate/pretrain_fsdp.yaml \
		-m gepard.cli.train run.resume_from=$(CHECKPOINT) $(EXP)

# Merge LoRA adapters of a *periodic* checkpoint-N into base weights (CPU-only).
# Not needed for the final export or DPO merged/ — those merge automatically.
LORA_ALPHA ?= 32
LORA_RANK  ?= 16
merge:
	@if [ -z "$(CHECKPOINT)" ]; then \
		printf "❌ Error: Please specify CHECKPOINT=path (a checkpoint-N dir with un-merged LoRA)\n"; \
		printf "Example: make merge CHECKPOINT=/opt/dlami/nvme/checkpoints-lora/checkpoint-2000\n"; \
		exit 1; \
	fi
	@if [ ! -d "$(CHECKPOINT)" ]; then \
		printf "❌ Error: Checkpoint directory not found: $(CHECKPOINT)\n"; \
		exit 1; \
	fi
	@printf "🔀 Merging LoRA adapters (alpha=$(LORA_ALPHA), rank=$(LORA_RANK)) → $(CHECKPOINT)-merged...\n"
	@source venv/bin/activate && python scripts/merge_lora_checkpoint.py $(CHECKPOINT) \
		--alpha $(LORA_ALPHA) --rank $(LORA_RANK)

# Upload to Hugging Face Hub.
# Default: inference artifact — weights cast to gepard_config's model_dtype
# (bf16), training state skipped. TRAINING_STATE=1: resume snapshot — optimizer/
# scheduler/RNG included and weights stay fp32 (dtype conversion is ignored).
TRAINING_STATE ?=
upload:
	@if [ -z "$(REPO)" ] || [ -z "$(CHECKPOINT)" ]; then \
		printf "❌ Error: Please specify REPO and CHECKPOINT\n"; \
		printf "Example: make upload REPO=username/model-name CHECKPOINT=./checkpoints/checkpoint-5000\n"; \
		exit 1; \
	fi
	@if [ ! -d "$(CHECKPOINT)" ]; then \
		printf "❌ Error: Checkpoint directory not found: $(CHECKPOINT)\n"; \
		exit 1; \
	fi
	@printf "📤 Uploading $(CHECKPOINT) to $(REPO)...\n"
	@source venv/bin/activate && python scripts/upload_to_hf.py --repo $(REPO) --checkpoint $(CHECKPOINT) --private \
		$(if $(TRAINING_STATE),--with-training-state,)

# Clean generated files
clean:
	@printf "🧹 Cleaning generated files...\n"
	@rm -rf checkpoints/
	@rm -rf train_dataset/
	@rm -rf wandb/
	@rm -rf __pycache__/
	@rm -rf utils/__pycache__/
	@rm -rf data/__pycache__/
	@rm -rf .pytest_cache/
	@rm -f training_log*.txt
	@printf "✅ Cleaned!\n"

# Test Flash Attention installation
test:
	@printf "🔍 Testing Flash Attention installation...\n"
	@source venv/bin/activate && python -c "import torch; import flash_attn; print(f'✅ Flash Attention {flash_attn.__version__} installed'); print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Stage-0 baseline suite (ROADMAP): fast tiny-model + pure-logic regression tests.
# Pins current behaviour so the refactor stages stay behavior-preserving.
test-baseline:
	@printf "🧪 Running Stage-0 baseline tests (fast, CPU)...\n"
	@source venv/bin/activate && python -m pytest

# Golden-loss checks against the real 513M backbone (needs GPU + ~1GB download).
test-baseline-real:
	@printf "🧪 Running real-backbone golden-loss checks (GPU)...\n"
	@source venv/bin/activate && python -m pytest -m real_model

# Check environment and dependencies
check-env:
	@printf "🔍 Checking environment...\n"
	@printf "\n"
	@printf "Python version:\n"
	@source venv/bin/activate && python --version
	@printf "\n"
	@printf "CUDA devices:\n"
	@nvidia-smi --query-gpu=index,name,memory.total --format=csv
	@printf "\n"
	@printf "Installed packages:\n"
	@source venv/bin/activate && pip list | grep -E "torch|transformers|accelerate|flash-attn|wandb|datasets"
	@printf "\n"
	@printf "Disk space:\n"
	@df -h . | tail -1
	@printf "\n"
	@printf "Memory:\n"
	@free -h | grep Mem
