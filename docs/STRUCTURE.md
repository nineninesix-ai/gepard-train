# Target project structure

> **Status: reached.** The repo matches this layout as of the Stage 5–7
> closing session (see ROADMAP §J); this file is kept as the layout's
> rationale and map.

The authoritative end-state layout the ROADMAP refactor moves toward. Decisions
locked in discussion: a **nested `gepard/` package**, small committed inputs under
**`assets/`**, and the two inference runners **merged** into one base+CFG module.
Inference **stays in this repo** (its runner is a real DPO dependency); only the
Colab notebook is surfaced at the top level for users.

```
gepard-train/
│
├── gepard/                          # all importable code (the package)
│   ├── __init__.py
│   ├── config/                      # @dataclass schema  (← utils/config.py)
│   ├── model/
│   │   ├── modeling.py              # GepardModel        (← utils/model.py)
│   │   ├── configuration.py         # GepardConfig(PretrainedConfig)  [Stage 3]
│   │   ├── ref_compressor.py        # (← utils/ref_compressor.py)
│   │   ├── lora.py                  # (← utils/lora.py)
│   │   ├── codec_ops.py             # (← utils/codec_ops.py)
│   │   └── losses/
│   │       ├── supcon.py            # (← utils/supcon.py)
│   │       └── dpo.py               # trajectory logprob + dpo_loss  (← utils/dpo.py)
│   ├── data/
│   │   ├── collator.py              # DataCollator            (← utils/data.py)
│   │   ├── sampling.py              # ReferenceSamplingDataset, SpeakerBucketBatchSampler,
│   │   │                            #   prepare_dataset       (← utils/data.py)
│   │   ├── preprocessing/
│   │   │   ├── prepare.py           # (← data/prepare.py)
│   │   │   ├── processor.py         # (← data/processor.py)
│   │   │   ├── text_repetition.py   # (← data/text_repetition.py)
│   │   │   └── build_keep_index.py  # (← data/build_short_keep_index.py)
│   │   └── dpo/
│   │       ├── sample.py            # (← data/dpo_sample.py)
│   │       ├── score.py             # (← data/dpo_score.py)
│   │       └── pairs.py             # (← data/dpo_pairs.py)
│   ├── training/
│   │   ├── base.py                  # shared build/ckpt/lora/save/export   [Stage 2]
│   │   ├── sft.py                   # pretrain + finetune trainer  (← utils/trainer.py)
│   │   ├── dpo.py                   # DPO training loop            (← train_dpo.py loop)
│   │   ├── callbacks.py             # (← utils/callbacks.py)
│   │   └── checkpoint_io.py         # unified ckpt resolve/load    [Stage 2]
│   ├── inference/                   # stays in repo, "hidden" inside the package
│   │   ├── runner.py                # TTSRunner (+ GepardRunner)  (← model_run.py + model_run_cfg.py)
│   │   └── codec_wrapper.py         # nemo codec, lazy import     (← codec_wrapper.py)
│   ├── logging/                     # (← logging_utils/)
│   │   ├── core.py                  # logger hierarchy, file routing, train logging
│   │   ├── dashboard.py             # rich.live UI + startup banners
│   │   └── model_card.py            # training_metadata.json + HF model card  [publish]
│   └── cli/                         # thin entry points (Hydra apps)
│       ├── train.py                 # (← train.py)
│       ├── train_dpo.py             # (← train_dpo.py entry)
│       └── prepare.py               # (← python -m data.prepare)
│
├── conf/                            # runtime YAML configs   (← configs/;  Hydra layout in Stage 5)
├── scripts/                         # shell ops + standalone python utilities
│   ├── setup.sh  setup_dpo.sh
│   ├── download_aws_dataset.sh  upload_aws_dataset.sh
│   ├── merge_lora_checkpoint.py     # (← merge_lora_checkpoint.py)
│   └── upload_to_hf.py              # (← upload_to_hf.py)
├── assets/                          # small committed inputs (not code, not big data)
│   ├── ref_audio/                   # 2–3 demo voices (heavy clips dropped: ~20 MB → <2 MB)
│   └── dpo_seed/short_focused_v2.jsonl   # (← dpo_data/short_focused_v2.jsonl)
├── notebooks/
│   └── inference_demo.ipynb         # Colab, user-facing   (← inference.ipynb)
├── docs/                            # MODEL_GUIDE.md, STRUCTURE.md
├── tests/                           # Stage-0 baseline suite (guards every move below)
├── README.md  ROADMAP.md  ROADMAP.ru.md
├── Makefile  pyproject.toml  pytest.ini
└── (removed on publish) text_docs/  # internal R&D — folds into docs/MODEL_GUIDE.md [Stage 1]
```

## Generated / gitignored (never committed)

`venv/`, `checkpoints/`, `dpo_checkpoints/`, `train_dataset/`, `dpo_dataset/round*/`
(tokens, scores.jsonl, pairs.jsonl), `wandb/`, `logs/`, `*.pytest_cache`.

## Amendments to the ROADMAP this locks in

- **Inference is NOT split to a separate repo** (cancels the split half of Stage 7).
  `gepard/inference/runner.py` holds the base `TTSRunner`; DPO imports it. Only
  `inference.ipynb` → `notebooks/` is surfaced for users.
- **The two runners merge.** `model_run.py` (base) + `model_run_cfg.py` (CFG) become
  one `runner.py`: base `TTSRunner` + subclass `GepardRunner`. This removes the
  near-duplicate file instead of deleting `model_run.py`.
- **Stage 4 becomes an import redirect**, not a decouple: `gepard/training/dpo.py`
  imports `gepard.inference.runner` (clean intra-package dependency) rather than a
  top-level `model_run` module. `codec_wrapper` (nemo) is never imported by the
  runner, so DPO stays importable in the training venv.

## Migration order (each step behavior-preserving, guarded by `make test-baseline`)

The package move pulls ROADMAP Stage 6 forward as the *skeleton*, because the
Stage-0 tests already pin behaviour and make a mechanical move safe. Suggested
slices, smallest-blast-radius first:

1. **assets + notebooks + scripts** (no code imports change): move `ref_audio/`
   (keep 2–3), `dpo_data/` → `assets/`; `inference.ipynb` → `notebooks/`; shell +
   util scripts → `scripts/`. Update Makefile paths.
2. **`gepard/model/` + `gepard/config/`**: move leaf modules (config, modeling,
   ref_compressor, lora, codec_ops, losses/*), rewrite their imports.
3. **`gepard/data/`**: split `utils/data.py` → collator/sampling; move
   `data/*` → preprocessing/ + dpo/.
4. **`gepard/training/` + `gepard/logging/`**: trainer→sft, callbacks, logging_utils.
5. **`gepard/inference/` + `gepard/cli/`**: merge runners → runner.py; thin CLIs;
   redirect DPO import; update tests + Makefile.

Then the remaining ROADMAP stages run *inside* the skeleton: Stage 1 renames
`KaniTTS3*` → `Gepard*` (a pure symbol rename now), Stage 2 extracts
`training/base.py` + `checkpoint_io.py`, Stage 3 adds `GepardConfig`, Stage 5
reshapes `conf/` for Hydra.
```
