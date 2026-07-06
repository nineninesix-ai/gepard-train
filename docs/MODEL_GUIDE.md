# Gepard Design Guide and Model Reference

This document serves as the canonical technical reference and engineering manual for the **Gepard** autoregressive speech model. It describes the design philosophy, serving architecture, system components, dataset construction, configuration details, and execution logic of the pipeline.

## Contents

- [1. Introduction and Serving Architecture](#1-introduction-and-serving-architecture)
  - [1.1. Overarching Design Principle](#11-overarching-design-principle)
  - [1.2. Concurrency and Latency Benchmarks](#12-concurrency-and-latency-benchmarks)
- [2. Acoustic Backbone and Codec Interface](#2-acoustic-backbone-and-codec-interface)
  - [2.1. Stock Transformer Backbone](#21-stock-transformer-backbone)
  - [2.2. Codec: GroupFSQ Orthogonality](#22-codec-groupfsq-orthogonality)
  - [2.3. Mixed-Radix Unfolding to 32 Heads](#23-mixed-radix-unfolding-to-32-heads)
- [3. Modality Scale Alignment and Gradient Attenuation](#3-modality-scale-alignment-and-gradient-attenuation)
  - [3.1. Projection Pipeline](#31-projection-pipeline)
  - [3.2. Gradient Starvation Mitigation](#32-gradient-starvation-mitigation)
- [4. Voice Cloning and Prefix Conditioning](#4-voice-cloning-and-prefix-conditioning)
  - [4.1. Compressor Architecture](#41-compressor-architecture)
  - [4.2. CFG-Dropout and Sentinel Masking](#42-cfg-dropout-and-sentinel-masking)
  - [4.3. Representation Leakage Regularization](#43-representation-leakage-regularization)
- [5. Short Register Failure Mode and Text Repetition](#5-short-register-failure-mode-and-text-repetition)
  - [5.1. Onset Entropy Probe](#51-onset-entropy-probe)
  - [5.2. Text Repetition Layout](#52-text-repetition-layout)
  - [5.3. Token Budget Calibration](#53-token-budget-calibration)
- [6. Output Heads and Stop Predictor](#6-output-heads-and-stop-predictor)
  - [6.1. Multihead Loss Formulation](#61-multihead-loss-formulation)
  - [6.2. Architecture-Level NaN-Guard](#62-architecture-level-nan-guard)
- [7. Direct Preference Optimization (DPO)](#7-direct-preference-optimization-dpo)
  - [7.1. Trajectory Log-Likelihood](#71-trajectory-log-likelihood)
  - [7.2. Length-Normalized DPO](#72-length-normalized-dpo)
  - [7.3. Two-Sided Length Reward](#73-two-sided-length-reward)
  - [7.4. Offline CFG Distillation](#74-offline-cfg-distillation)
- [8. Detailed Configuration System Guide](#8-detailed-configuration-system-guide)
  - [8.1. Config Tree Architecture](#81-config-tree-architecture)
  - [8.2. Composition and Overrides](#82-composition-and-overrides)
  - [8.3. Parameter Group Reference Dictionary](#83-parameter-group-reference-dictionary)
  - [8.4. Configuration Validation Engine](#84-configuration-validation-engine)
- [9. The Dataset: Sources, Construction, and Loading](#9-the-dataset-sources-construction-and-loading)
  - [9.1. Source Dataset Contract](#91-source-dataset-contract)
  - [9.2. Per-Source Configuration: the `hf_datasets` Item](#92-per-source-configuration-the-hf_datasets-item)
  - [9.3. Build Pipeline, Stage by Stage](#93-build-pipeline-stage-by-stage)
  - [9.4. Speaker Identity and the Singleton Policies](#94-speaker-identity-and-the-singleton-policies)
  - [9.5. Row Transform: Sequence and Label Layout](#95-row-transform-sequence-and-label-layout)
  - [9.6. Text Repetition Is a **Prepare-Time** Decision](#96-text-repetition-is-a-prepare-time-decision)
  - [9.7. The Fine-Tune Keep-Index (Duration Reweighting)](#97-the-fine-tune-keep-index-duration-reweighting)
  - [9.8. Training-Side Loading: `prepare_dataset`](#98-training-side-loading-prepare_dataset)
  - [9.9. `ReferenceSamplingDataset`: What Voice Cloning Adds at Load Time](#99-referencesamplingdataset-what-voice-cloning-adds-at-load-time)
  - [9.10. Two Batching Regimes](#910-two-batching-regimes)
  - [9.11. `DataCollator`: the Batch Contract](#911-datacollator-the-batch-contract)
  - [9.12. The DPO Dataset: Why It Is Different](#912-the-dpo-dataset-why-it-is-different)
  - [9.13. Cross-Config Dependency Map (Data Edition)](#913-cross-config-dependency-map-data-edition)
- [10. The Model: Architecture, Configuration, and Checkpoints](#10-the-model-architecture-configuration-and-checkpoints)
  - [10.1. Anatomy: Stock Core, Thin Overlay](#101-anatomy-stock-core-thin-overlay)
  - [10.2. The Backbone](#102-the-backbone)
  - [10.3. Audio Interface — Input Side](#103-audio-interface--input-side)
  - [10.4. Output Side](#104-output-side)
  - [10.5. Forward Contract and Runtime Attributes](#105-forward-contract-and-runtime-attributes)
  - [10.6. Static Model Configuration (`model/` group)](#106-static-model-configuration-model-group)
  - [10.7. Three Construction Paths](#107-three-construction-paths)
  - [10.8. Checkpoint Formation](#108-checkpoint-formation)
  - [10.9. Self-Describing Checkpoints, End to End](#109-self-describing-checkpoints-end-to-end)
- [11. Voice Cloning: the Cross-Cutting Subsystem](#11-voice-cloning-the-cross-cutting-subsystem)
  - [11.1. Input Representation: Codec Codes, Not Waveforms](#111-input-representation-codec-codes-not-waveforms)
  - [11.2. `RefCompressor` Architecture](#112-refcompressor-architecture)
  - [11.3. The K-Query Bottleneck Is the Central Design Bet](#113-the-k-query-bottleneck-is-the-central-design-bet)
  - [11.4. The Unconditional Path: `null_prefix`](#114-the-unconditional-path-null_prefix)
  - [11.5. Training Strategy Across Phases (and Its Openness)](#115-training-strategy-across-phases-and-its-openness)
  - [11.6. Generation-Time Usage](#116-generation-time-usage)
  - [11.7. Config Surface and the Enable/Disable Contract](#117-config-surface-and-the-enabledisable-contract)
- [12. Training: Pretrain, Fine-Tune, DPO](#12-training-pretrain-fine-tune-dpo)
  - [12.1. Command → What Actually Runs](#121-command--what-actually-runs)
  - [12.2. The Two Accelerate Profiles](#122-the-two-accelerate-profiles)
  - [12.3. Phase I — Pretrain](#123-phase-i--pretrain)
  - [12.4. Phase II — LoRA Fine-Tune](#124-phase-ii--lora-fine-tune)
  - [12.5. Phase III — DPO Training](#125-phase-iii--dpo-training)
  - [12.6. Checkpointing and Resume](#126-checkpointing-and-resume)
  - [12.7. wandb Observability — What We Log and Why](#127-wandb-observability--what-we-log-and-why)
  - [12.8. Running Experiments](#128-running-experiments)
- [13. Inference](#13-inference)
  - [13.1. The Classes](#131-the-classes)
  - [13.2. Loading](#132-loading)
  - [13.3. Generation Anatomy](#133-generation-anatomy)
  - [13.4. CFG Is Optional — and Off by Default](#134-cfg-is-optional--and-off-by-default)
  - [13.5. The Notebook Flow](#135-the-notebook-flow)
- [14. Environment and Operations](#14-environment-and-operations)
  - [14.1. Three Virtualenvs](#141-three-virtualenvs)
  - [14.2. Setup Logic (why bash, not a lockfile)](#142-setup-logic-why-bash-not-a-lockfile)
  - [14.3. Ephemeral NVMe (the AWS Case)](#143-ephemeral-nvme-the-aws-case)
  - [14.4. Dataset and Checkpoint Transfer](#144-dataset-and-checkpoint-transfer)
  - [14.5. Makefile](#145-makefile)

---

## 1. Introduction and Serving Architecture

Gepard is an autoregressive decoder-only model designed for real-time spoken dialogue, generating speech tokens directly from text prompts. The primary design goal is **standard engine serving compatibility** (such as with vLLM): the model runs on stock inference engines without custom CUDA kernels or custom layers in the autoregressive loop.

### 1.1. Overarching Design Principle
To ensure compatibility with continuous batching and PagedAttention mechanisms, Gepard enforces a strict separation:
*   **Prefill Phase (Offline/Static):** All custom sequence preparations—such as speaker-profile extraction via a Q-Former compressor and adaptive text repetition—are computed once during prompt prefill and are excluded from the step-by-step autoregressive decode loop.
*   **Generation Phase (Online/Autoregressive):** The model operates as a flat, single-pass decoder generating speech frames sequentially, preserving a clean KV-caching loop. Complex two-pass generation techniques like Classifier-Free Guidance (CFG) are distilled into the weights via Direct Preference Optimization (DPO).

### 1.2. Concurrency and Latency Benchmarks
Under serving workloads on server-class hardware running in 16-bit precision, Gepard streams audio chunk-by-chunk using the SSE protocol:
*   **Single Stream:** Achieves a Real-Time Factor (RTF) of approximately `0.067` (about 15 times faster than real-time) with a Time-to-First-Audio (TTFA) of ~0.046 seconds.
*   **Concurrent Scaling:** Aggregate throughput scales linearly, reaching system-level speeds of over `200x` real-time throughput at 256 concurrent streams on a single server-class GPU.
*   **Interactive Limit:** The optimal operating range is 64 to 128 simultaneous streams per GPU, beyond which compute saturation causes individual streams to fall behind real-time.

---

## 2. Acoustic Backbone and Codec Interface

### 2.1. Stock Transformer Backbone
The core transformer is a stock decoder-only architecture (based on Qwen3.5 with 14 layers, a hidden dimension of 1024, and 8 attention heads). All custom linear block overrides are removed to maintain standard FlashAttention-2 compatibility. The backbone's parameters are defined and instantiated in `gepard/model/modeling.py`.

### 2.2. Codec: GroupFSQ Orthogonality
Gepard tokenizes speech using a neural codec operating at a frame rate of 21.5 Hz with a low bitrate (~1.89 kbps). 

Instead of Residual Vector Quantization (RVQ), which introduces hierarchical dependencies (where each codebook layer quantizes the residual of the previous layer), Gepard is built around **GroupFSQ (Finite Scalar Quantization)**. The latent space is divided into 8 independent groups, each quantized by its own FSQ grid of levels (yielding 2,016 code capacities per group). 

Because the channels are orthogonal and independent by design, the mutual information between channels given the hidden state is negligible. This allows **factorized parallel sampling** across all channels in a single autoregressive step, rendering vertical depth-transformers or codebook-by-codebook sampling loops unnecessary.

### 2.3. Mixed-Radix Unfolding to 32 Heads
The 8 packed codebook tokens (values 0 to 2015) are unfolded into 32 independent channels per frame with repeating alphabet capacities of `[8, 7, 6, 6]`. This is done using a little-endian mixed-radix decomposition implemented in `gepard/model/codec_ops.py`.

Predicting these 32 channels directly via 32 tiny linear heads (totaling 216 logits) instead of 8 heads of size 2016 significantly reduces classification layer overhead, bypasses redundant decompression steps during inference, and aligns the backbone outputs directly with the codec's internal dequantization stage.

---

## 3. Modality Scale Alignment and Gradient Attenuation

Connecting pretrained text embeddings and from-scratch audio embeddings in a single decoder-only model introduces representational scale mismatches and gradient starvation.

### 3.1. Projection Pipeline
The frame embedding for the 32 discrete audio channels is constructed by looking up values in 32 corresponding lookup tables (each of dimension `L_k x 32`), concatenating them into a 1024-dimensional vector, projecting it through a two-layer GELU MLP, applying an affine-free LayerNorm, and scaling the result by a fixed standard deviation buffer matching the pretrained text embeddings.

### 3.2. Gradient Starvation Mitigation
In early configurations, placing hard scale multipliers (like `0.02`) and intermediate normalization layers between the audio lookup tables and the backbone created a gradient barrier: gradients reaching the audio embeddings were thousands of times smaller than text and backbone gradients.

To balance training dynamics:
1.  The intermediate normalizer and manual scale barriers are bypassed, using a direct MLP coupling.
2.  The lookup tables are initialized with unit variance, and the MLP projections are initialized inversely proportional to their input sizes.
3.  The final frame scale is matched to text standard deviation via a non-learnable coefficient.
4.  Split learning rates are applied during optimization (e.g., text embedding learning rate is scaled down to protect pretrained weights, while the audio interface is scaled up).

Weight-space diagnostics show that this keeps FSQ codes linearly distinguishable and free from collapse. While the modalities start orthogonal, hidden-state monitoring shows that the causal full-attention layers act as a cross-modal alignment mechanism, aligning activations in the deep hidden space of the network (achieving near-perfect similarity in the final layers).

---

## 4. Voice Cloning and Prefix Conditioning

Voice cloning extracts speaker characteristics from a reference clip once during prompt prefill and prepends them as sequence prefix tokens.

### 4.1. Compressor Architecture
The reference stack of dequantized codes is compressed into a fixed set of 8 query tokens using a 2-block Q-Former (self-attention followed by masked cross-attention to reference features and SwiGLU FFN). The output speaker prefix is normalized and scaled to prevent it from dominating the text embeddings.

### 4.2. CFG-Dropout and Sentinel Masking
At training time, the compressor queries are replaced with a learnable `null_prefix` parameter (defining the unconditional path) via an OR gate combining:
1.  Stochastic CFG-dropout at a rate of 15%.
2.  Forced substitution for low-frequency speakers (with few clips in the dataset) and speaker-less rows.

This ensures the model is sufficiently exposed to the unconditional prefix state, which is required for Classifier-Free Guidance (CFG) at inference.

### 4.3. Representation Leakage Regularization
Because the reconstruction objective incentives the compressor to copy the exact spectral sequence of the reference clip rather than abstract timbre, two regularizers are applied during pretraining (implemented in `gepard/model/losses/supcon.py`):
1.  **Diversity Loss (hinge-variance):** Prevents the query vectors from collapsing into a single average representation by enforcing a variance threshold across queries.
2.  **Supervised Contrastive Loss (SupCon):** Enforces content invariance. Average speaker query vectors are projected and normalized. The loss maximizes the similarity of representations of different clips from the same speaker while minimizing similarity to other speakers in the batch.

---

## 5. Short Register Failure Mode and Text Repetition

On short text inputs (1 to 2 words), the speaker prefix (8 tokens) dominates the causal self-attention states of the transformer, weakening the text conditioning. The model fails to lock onto text alignment, resulting in high-entropy frame predictions and infinite generation loops (runaway).

### 5.1. Onset Entropy Probe
We track frame-by-frame token negative log-likelihood (NLL) and belief entropy over the 32 heads during generation to diagnose derailment:
*   **Clean Generation:** Belief entropy drops rapidly within the first 50 frames as the model locks onto speech generation.
*   **Derailed Loop:** Belief entropy remains flat and elevated throughout the generation.
*   The minimum word error rate (WER) across multiple temperature-scaled rollouts remains zero, confirming that the model knows the vocabulary but fails due to generation instability.

### 5.2. Text Repetition Layout
To strengthen text conditioning on short prompts, the input text is repeated multiple times before the canonical copy (implemented in `gepard/data/preprocessing/text_repetition.py`):

```
[ (SOT text EOT) x (R-1) | SOT text EOT SOS | audio ... ]
```

*   **SOS Gating:** The Start of Speech (SOS) token is attached only to the final copy, which triggers audio generation; the context copies are read by self-attention but are never voiced.
*   **Supervision Masking:** Context text tokens are masked with `-100` in the labels to prevent double-voicing supervision.
*   **Mixed Keep:** A fraction of short inputs (25%) is kept without repetition during training, so the model learns that repetition is optional and remains stable in its absence.

### 5.3. Token Budget Calibration
Sweeping the failure rate against the text-token budget reveals three regimes:
1.  **Cliff Zone (≤ 6 tokens):** High failure rates (60–96%).
2.  **Transition Zone (7–12 tokens):** Failure rate decreases.
3.  **Plateau Zone (≥ 13 tokens):** Failure rate stabilizes below 5% for familiar voices.

This sweep justifies a target text token budget of `16` and a threshold of `13` tokens for applying repetition.

---

## 6. Output Heads and Stop Predictor

### 6.1. Multihead Loss Formulation
Output projections are applied only to the audio region (the speaker prefix states are discarded). The joint loss combines:
1.  **Shifted Cross-Entropy Loss:** Summed across the 32 categorical codebook heads.
2.  **Weighted Binary Cross-Entropy Loss:** Applied to the stop predictor head.

Because the stop event occurs only once per audio sequence (approx. 1 frame out of 150), standard BCE collapses to predicting "always 0". To counter this class imbalance, BCE is calculated with a positive class weight (e.g., `pos_weight = 25.0`), and the stop head loss is scaled.

### 6.2. Architecture-Level NaN-Guard
If a training microbatch contains no valid labels for a head (due to masking), standard mean reduction division results in a NaN. Gepard guards against this in `gepard/model/modeling.py` by substituting the loss with a zeroed scale (`logits.sum() * 0`), keeping the output head in the compute graph with zero gradient.

---

## 7. Direct Preference Optimization (DPO)

To prevent runaway loops at serving time without deploying two-pass Classifier-Free Guidance (CFG), we perform offline CFG distillation using Direct Preference Optimization (DPO).

### 7.1. Trajectory Log-Likelihood
Generation is modeled as a Bernoulli stop decision followed by independent categorical channel predictions. The trajectory log-likelihood combines the stop probabilities and the 32 channel log-probabilities. Because the stop head is highly saturated during training, predicted stop probabilities are clipped from below at `10^-4` to prevent vanishing gradients.

### 7.2. Length-Normalized DPO
We optimize the model with standard reference-anchored DPO on **length-normalized** trajectory log-likelihoods (per-frame average; the normalization idea follows SimPO, but unlike SimPO the frozen-reference terms are kept — precomputed once at the pair-building stage, so training holds only one model in memory):

$$\mathcal{L} = -\log\sigma\left(\beta\left[\frac{\log\pi_\theta(y_w \mid x) - \log\pi_\text{ref}(y_w \mid x)}{T_w} - \frac{\log\pi_\theta(y_l \mid x) - \log\pi_\text{ref}(y_l \mid x)}{T_l}\right]\right)$$

with $\beta = 2.0$; $y_w$/$y_l$ are the chosen/rejected trajectories of lengths $T_w$/$T_l$ (implementation: `gepard/model/losses/dpo.py::dpo_loss`). Optionally, per-pair losses are re-weighted by the reward gap of the pair (`reward_weight_mode`, §9.12): weights are normalized to keep the batch loss scale, redistributing gradient toward high-contrast pairs.

### 7.3. Two-Sided Length Reward
Programmatic rewards evaluate rollouts using an ASR model:
*   **WER Penalty:** Deducts score based on transcription word error rate.
*   **Two-Sided Length Penalty:** Penalizes both overly long generations (runaway) and overly short generations (premature stops).
*   **Similarity Penalty:** Evaluates speaker similarity.

The two-sided length penalty is required; otherwise, preference optimization collapses into immediately ending generation at the first frame.

### 7.4. Offline CFG Distillation
During offline DPO rollout generation, a two-pass text CFG is applied at a scale of `3.0` over the first 20 frames for difficult voices, generating high-quality positive trajectories that are otherwise unreachable by random sampling. DPO trains the model on these pairs, baking the CFG alignment benefit directly into the single-pass weights.

---

## 8. Detailed Configuration System Guide

The entire Gepard lifecycle—dataset compilation, training/SFT, DPO preference optimization, and inference runtime—is controlled by a single unified config system using [Hydra](https://hydra.cc). This system splits configuration facts into modular **groups** to eliminate code duplication and ensure synchronization across steps.

### 8.1. Config Tree Architecture
The configuration configurations are organized in the `conf/` directory. Rather than duplicating parameters across runs, Hydra composes a configuration dynamically.

#### Layout of the `conf/` Tree:
```
conf/
├── train.yaml                 # Entry config for PRETRAIN (make train)
├── sft.yaml                   # Entry config for the LoRA fine-tune (make finetune)
├── prepare.yaml               # Entry config for dataset compilation
├── dpo.yaml                   # Entry config for the DPO pipeline
├── codec/                     # Global FSQ codec geometry groups
├── tokens/                    # Special token vocab identity groups
├── text_layout/               # Adaptive text repetition layout groups
├── model/                     # Transformer backbone + audio interface groups
├── voice_cloning/             # Q-Former architecture + contrastive training groups
├── data/                      # Dataset sources and singleton routing rules
├── trainer/                   # Training arguments, optimizer & scheduler specs
├── finetune/                  # LoRA parameters and freeze maps
├── dpo/                       # DPO sampling, scoring, and preference loop groups
└── experiment/                # Optional ad-hoc preset overrides (currently empty)
```

The three training/data phases each have their **own entry file** — `train.yaml`
(pretrain), `sft.yaml` (LoRA fine-tune), `dpo.yaml` — rather than one entry plus
phase presets, so a phase reads top-to-bottom in a single file. `experiment/` is
kept only for one-off variations layered on top of an entry (§8.2); it ships empty.

### 8.2. Composition and Overrides
A top-level entry configuration file (such as `conf/train.yaml`) lists the default configuration group for each layer:

```yaml
defaults:
  - _self_
  - tokens: qwen3_5              # Special token IDs
  - codec: nano_21_5            # Global codec dimensions
  - text_layout: pretrain_stage # Repetition OFF for pretrain (§9.6)
  - model: gepard               # Model architecture config
  - voice_cloning: qformer_supcon  # Q-Former speaker prefix + SupCon (pretrain default)
  - data: full_corpus           # Dataset sourcing config
  - trainer: pretrain           # Trainer optimizer schedule
  - finetune: none              # Freeze & LoRA settings
  - optional experiment: null   # Presets that override any group

run:
  resume_from: null             # checkpoint dir to resume from
  stage: pretrain               # phase label frozen into the checkpoint's
                                #   training_metadata.json / model card (§12.3)
```

*   **Phase entries vs. experiment presets:** whole *phases* are separate entry files (`train.yaml` → pretrain, `sft.yaml` → LoRA fine-tune, `dpo.yaml`), each composing its own default groups — not experiment presets. `conf/experiment/` is reserved for optional one-off *variations* layered on top of an entry (`experiment=my_exp` swaps a few groups without copying invariant values); it ships two commented examples — `experiment/ft_wider_lora.yaml` (fine-tune) and `experiment/pretrain_no_vc.yaml` (pretrain) (§12.8).
*   **Command Line Overrides:** All parameters can be overridden at invocation time:
    `python -m gepard.cli.train trainer.learning_rate=1e-4 trainer.per_device_train_batch_size=16`

### 8.3. Parameter Group Reference Dictionary
Every parameter group is mapped to a Python `@dataclass` schema in `gepard/config/schema.py`. Below is the complete field reference:

#### 8.3.1. `tokens/` (Token Definition)
Defines special token IDs. These must remain synchronized across tokenization, model compilation, training, and inference decoding.

*   `tokenizer_name` (str): Hugging Face model repository ID of the base text tokenizer.
*   `tokeniser_length` (int): Text vocabulary size.
*   `start_of_text` (int): Token ID marking prompt start (SOT).
*   `end_of_text` (int): Token ID marking prompt end (EOT).
*   `start_of_speech` (int): Token ID marking the beginning of speech tokens (SOS).
*   `end_of_speech` (int): Token ID marking the end of speech tokens (EOS).
*   `tts_pad` (int): Pad token ID used for padding audio sequences.

#### 8.3.2. `codec/` (FSQ Codec Geometry)
Defines the structure of the audio tokenizer.
*   `num_layers` (int): Number of independent quantization layers (e.g. 8).
*   `fsq_levels` (List[int]): Number of quantization intervals for each dimension (e.g. `[8, 7, 6, 6]`).
*   `do_unfold` (bool): If True, unfolds the packed codebook tokens into orthogonal FSQ channels.
*   `frame_rate_hz` (float): Audio frame rate (e.g. 21.5 frames per second).
*   `codec_id` (str): Hugging Face ID of the pretrained neural codec.
*   `sample_rate` (int): Codec target audio sampling rate in Hz (e.g. 22050).

#### 8.3.3. `text_layout/` (Text Repetition Layout)
Configures repetition parameters to prevent runaway loops on short inputs.
*   `enabled` (bool): Enables adaptive text repetition.
*   `target_text_tokens` (int): Target text tokens required to lock speech alignment (default: 16).
*   `apply_below` (int): Maximum length threshold below which repetition applies (default: 13).
*   `max_repeats` (int): Hard cap on the number of repetitions (default: 8).
*   `mixed_keep_prob` (float): Fraction of short inputs to train without repetition (default: 0.25).
*   `seed` (int): Random seed for text layout shuffling.

#### 8.3.4. `model/` (Backbone and Projections)
Defines the core transformer layers and multi-head audio classifiers.
*   `backbone_id` (str): Identifier of the base text transformer.
*   `attn_implementation` (str): Attention mechanism (default: `"flash_attention_2"`).
*   `dtype` (str): Computational data precision (`"bfloat16"`, `"float16"`, or `"float32"`).
*   `audio_embed_dim` (int): Embedding projection dimension of individual FSQ channels (default: 32).
*   `partial_rotary_factor` (float): Percentage of the hidden dimension covered by Rotary Embeddings.
*   `stop_loss_weight` (float): Scalar loss multiplier for the stop predictor head (default: 2.0).
*   `stop_pos_weight` (float): Positive class weight in Binary Cross Entropy to balance stop prediction (default: 25.0).
*   `audio_heads` (Dict[str, int]): Derived dictionary mapping head names to their vocabulary classes (e.g. 32 heads of sizes 8, 7, 6, or 6). Computed dynamically.

#### 8.3.5. `voice_cloning/` (Q-Former and Regularizers)
Defines zero-shot cloning parameters and regularizer loss terms.
*   `enabled` (bool): Gaters voice cloning queries. If False, the Q-Former prefix path is bypassed.
*   `reference_sampling` (ReferenceSamplingConfig): Controls duration boundaries (`l_min_seconds` to `l_max_seconds`) and self-reference sampling flags for speaker cloning selection.
*   `compressor.num_queries` (int): Number of speaker prefix tokens prepended to text prompts (default: 8).
*   `compressor.num_layers` (int): Transformer layers in Q-Former (default: 2).
*   `compressor.num_heads` (int): Attention heads in Q-Former (default: 8).
*   `compressor.d_model` (Optional[int]): Q-Former hidden size. If null, inherits from the backbone dimension.
*   `compressor.ffn_hidden_size_multiplier` (int): FFN expansion factor (default: 4).
*   `compressor.dropout` (float): Dropout probability in cloning layers (default: 0.1).
*   `compressor.queries_init_std` (float): Init std of the K learnable query vectors (default: 0.02).
*   `compressor.lr_multiplier` (float): Learning rate multiplier for the `ref` optimizer group — `ref_compressor.*` + `null_prefix` + `supcon_head.*` (§12.3).
*   `training.cfg_dropout_prob` (float): CFG-dropout rate during SFT (default: 0.15).
*   `training.null_prefix_init_std` (float): Random initialization variance of the `null_prefix` parameter (default: 0.02).
*   `training.diversity_loss` (DiversityLossConfig): Controls diversity regularizer settings (e.g., `gamma = 0.5`, `weight = 1.0`, linear warmup and ramp parameters).
*   `training.supcon` (SupConConfig): Supervised Contrastive regularizer. Fields: `enabled`; batch shape `P`/`K`/`M` (§9.10); `weight`, `warmup_start`, `ramp_steps` (curriculum, §12.3); `temperature` (0.1); `use_projection` + `projection_hidden_dim`/`projection_dim` (the disposable SimCLR-style head — turning `use_projection` off also removes `supcon_head.*` from the checkpoint layout); `gather_across_ranks` (all-gather negatives across FSDP ranks; disable only for single-GPU debugging — it changes the effective contrastive batch).

#### 8.3.6. `data/` (Dataset Processing)
Configures dataset compilation and singleton speaker rules.
*   `train_dataset_path` (str): Authoritative path to output prepared datasets.
*   `max_duration_sec` (Optional[float]): Audio duration filter cap.
*   `add_row_id` (bool): Embeds raw indices for tracking source data records.
*   `add_speaker_id` (bool): Prepends speaker tags for voice cloning batches.
*   `singleton_policy` (str): Speaker routing policy (`"null_prefix"` or `"remove"`).
*   `min_clips_per_speaker` (int): Filter threshold to discard low-frequency speakers (default: 3).
*   `processing` (ProcessingConfig): Configures worker parallelisms for Arrow chunking (`num_shards`, `load_dataset_num_proc`, `filter_num_proc`).

#### 8.3.7. `trainer/` (Optimization Details)
Configures standard SFT training arguments.
*   `output_dir` (str): Path to write training checkpoints.
*   `save_steps` (int): Checkpoint logging frequency.
*   `num_train_epochs` (int): Total training epochs.
*   `per_device_train_batch_size` (int): Batch size per GPU device.
*   `gradient_accumulation_steps` (int): Steps before executing weight updates.
*   `learning_rate` (float): Maximum optimizer step size.
*   `lr_scheduler_type` (str): Learning rate decay schedule (default: `"cosine"`).
*   `warmup_steps` (int): Linear LR warmup steps before the schedule starts.
*   `max_grad_norm` (float): Gradient clipping threshold (3.0 pretrain / 1.0 fine-tune).
*   `optim` (str): Target optimizer (default: `"adamw_torch_fused"`).
*   `bf16` / `fp16` (bool): Training precision settings (must agree with `model.dtype` — validator inv. 5).
*   `weight_decay` (float): AdamW decay; biases/LayerNorm excluded per group (§12.3). Default 0.0.
*   `seed` (int): HF Trainer seed; also feeds the SupCon batch sampler's per-epoch RNG (§9.10).
*   `logging_steps` (int): wandb/console cadence. Kept at 1 — the per-layer cosine collection arming depends on it (§12.7).
*   `save_total_limit` (int): Checkpoint rotation depth. Rotation DELETES old `checkpoint-N` dirs — keep backups outside `output_dir` (§12.3).
*   `report_to` (list): `["wandb"]` or `[]` to disable telemetry.
*   `expensive_metrics_every` (int): Interval for the SVD/hidden-state diagnostics (§12.7).
*   `keep_index_path` (Optional[str]): Location of a prebuilt `.npy` row keep index. Used to apply dataset reweighting.
*   `audio_lr_multiplier` / `embed_lr_multiplier` (float): Per-group LR scaling (audio overlay / `embed_tokens`, §12.3).
*   `gradient_checkpointing` (bool): Reclaims VRAM by recalculating activation states on backward passes (forwarded to the backbone only, §10.2).
*   `dataloader_num_workers` / `dataloader_persistent_workers` / `dataloader_prefetch_factor` / `dataloader_pin_memory`: standard PyTorch DataLoader throughput knobs, passed through verbatim.
*   `remove_unused_columns` (bool): **must stay `false`.** HF Trainer prunes dataset columns not named in the model's `forward` signature; Gepard's audio channels arrive via `**kwargs` (§10.5), so pruning would silently drop every `level_audio_*` column.
*   `average_tokens_across_devices` (bool): kept `false` — per-device loss averaging matches the historical runs; flipping it changes gradient scale on multi-GPU.
*   `wandb` (project / name / entity): run identity; `entity` empty → wandb default.

#### 8.3.8. `finetune/` (LoRA Fine-tuning)
*   `checkpoint_path` (str): Base checkpoint to load before injection.
*   `freeze_backbone` / `freeze_ref_compressor` / `freeze_supcon_head` / `freeze_null_prefix` (bool): Frozen state targets (prefix-matched parameter groups, §10.1).
*   `lora.enabled` (bool): Injects LoRA adapters to adapt weights.
*   `lora.rank` (int): LoRA projection rank (e.g. 16).
*   `lora.alpha` (int): LoRA scaling coefficient (e.g. 32).
*   `lora.target_modules` (List[str]): Targets for adapter injection (default: `["q_proj", "k_proj", "v_proj", "o_proj"]`).
*   `lora.last_n_layers` (int): Limits LoRA adapter injection to the top N layers of the transformer backbone (default: 16).

#### 8.3.9. `dpo/` (Preference Optimization)
Top level: `run_name` (isolates a round; all derived paths hang off it, §9.12) and `p_floor` (the single logprob floor mirrored into `pairs`/`training` — validator inv. 9).

`sampling` (stage 1, `conf/dpo/sampling/`):
*   `checkpoint` (str): model to roll out (HF repo or local dir); also the default frozen reference for stage 3.
*   `texts_file` (str): seed-prompt JSONL (§9.12).
*   `ref_audios` (List[str]) / `speaker_pool` (str): reference voices — explicit wav paths, or an HF dataset with an `audio` column (rows become `pool_NN`); `holdout_speakers` are excluded and reserved for unseen-voice eval.
*   `null_prefix_prob` (float): fraction of rollouts generated with the null prefix instead of a speaker (keeps the unconditional path in the preference data).
*   `speakers_per_text` / `num_samples`: group fan-out (§9.12).
*   `temperature` / `top_k` / `stop_threshold`: sampling knobs, matched to production serving.
*   `cfg_scale` / `cfg_frames` / `cfg_uncond_mode`: offline text-CFG for positive-trajectory mining (§7.4).
*   `cap_expected_multiple` / `cap_min_frames` / `cap_max_frames`: the adaptive runaway cap (§9.12); `seed`.

`reward` (stage 2, `conf/dpo/reward/`):
*   `whisper_model` / `whisper_language` / `whisper_batch_size` / `decode_batch_size`: ASR + codec-decode throughput.
*   `sec_base` + `sec_per_word`: the linear expected-duration model; `dur_min_ratio` / `dur_max_ratio` scale it into the per-text acceptance window.
*   `w_wer` / `w_over` / `w_short` / `w_empty`: reward weights (§7.3 — the two-sided length term is mandatory).
*   `sim_enabled` / `w_sim` / `sim_model` / `sim_sr`: optional speaker-similarity term (WavLM x-vector cosine).

`pairs` (stage 3, `conf/dpo/pairs/`):
*   `chosen_max_wer` / `chosen_not_truncated` / `chosen_dur_in_bounds`: absolute quality gates for the chosen side (§9.12).
*   `min_reward_margin` / `max_pairs_per_group`: pair selection policy.
*   `ref_checkpoint` (str): frozen reference model; empty → `sampling.checkpoint`. `ref_logp_batch`: teacher-forcing batch.

`training` (stage 4, `conf/dpo/training/`): `out_dir`; the loss knobs `beta`, `length_normalize`, `stop_term_weight`, `p_floor`; reward-gap pair weighting `reward_weight_mode` (`none`/`linear`/`clip`) with `reward_weight_scale`/`reward_weight_max`; the `lora` block (its own dataclass — narrower `q,v` targets than SFT); `train_stop_head` / `stop_head_lr` and the (off by default) `train_audio_heads`; loop shape `num_epochs` / `batch_pairs` / `grad_accum` / `warmup_steps` / `max_grad_norm` / `seed`; cadence `save_steps` / `log_steps`; `save_merged` (export a runner-loadable merged dir, §12.5); wandb identity `wandb_project` / `wandb_name` / `entity` (team/org; empty → personal) / `report_to`.

---

### 8.4. Configuration Validation Engine
To prevent resource-heavy runs from failing mid-training due to configuration inconsistencies, Gepard runs a cross-field validation pass at startup. 

The validator (`gepard/config/validate.py`) check parameters across three levels:
*   **L1 (Hydra Level):** Checks that required fields (marked `MISSING`) are set during composition and validates field types.
*   **L2 (Intra-Group Level):** Checks local invariants during class `__post_init__` passes (e.g. positive learning rates, valid dtypes, and valid probability bounds).
*   **L3 (Cross-Group Level):** Enforces consistency rules across different configuration files.

#### The Cross-Field Validation Invariants (`gepard/config/validate.py`):
1.  **Codec sanity:** `codec.frame_rate_hz > 0` and `codec.fsq_levels` non-empty (all entry points).
2.  **Head layout parity:** `model.audio_heads` must have exactly `num_layers × len(fsq_levels)` entries (or `num_layers × 1` when `do_unfold` is disabled).
3.  **Dtype validity:** `model.dtype` ∈ {bfloat16, float16, float32}.
4.  **SupCon dependency:** `voice_cloning.enabled = true` whenever the supervised contrastive loss is enabled.
5.  **Precision agreement:** `trainer.bf16` and `trainer.fp16` are mutually exclusive, and `trainer.bf16` must equal `model.dtype == "bfloat16"`.
6.  **Voice cloning speaker labels:** `data.add_speaker_id = true` when voice cloning is active, so references can be grouped by speaker.
7.  **SupCon clip floor:** `data.min_clips_per_speaker >= supcon.K` — every eligible speaker can supply K positives.
8.  **Batch composition:** `per_device_train_batch_size == P·K + M` when SupCon is enabled.
9.  **DPO p_floor synchronization:** `dpo.p_floor == pairs.p_floor == training.p_floor` — pair-building and training must clip logprobs identically.
10. **DPO speaker allocation:** `sampling.speakers_per_text <= len(sampling.ref_audios)` (when no speaker pool is configured).
11. **Text-layout bounds:** `text_layout.mixed_keep_prob` ∈ [0, 1]; when repetition is enabled, `target_text_tokens >= 1` and `max_repeats >= 1` (all entry points).
12. **Data-group sanity (train + prepare):** `data.singleton_policy` ∈ {remove, null_prefix}, `min_clips_per_speaker >= 1`, and `max_duration_sec > 0` when set.
13. **Prepare sources:** `data.hf_datasets` must be non-empty for the prepare step — an empty list builds nothing.
14. **LoRA sanity:** when adapters are in play (`finetune.lora.enabled`, and always for DPO training), `rank >= 1`, `alpha > 0`, `dropout` ∈ [0, 1), non-empty `target_modules`, `last_n_layers >= 1`.
15. **DPO beta:** `dpo.training.beta > 0`.

Additionally, `require_dataset_built(cfg)` runs at the train entry points (kept out of `validate()` because it touches the filesystem) and fails fast on filesystem preconditions: `data.train_dataset_path` must be a saved HF dataset, `trainer.keep_index_path` (when set) must exist, and `finetune.checkpoint_path` must exist when it is explicitly path-like (starts with `/`, `./`, `../` or `~`) — HF repo ids are left for `resolve_checkpoint_file` to handle at load time.

Rules that need the actual backbone or checkpoint — compressor width vs hidden size, checkpoint token-map identity, LoRA depth vs layer count — are deliberately NOT config-level invariants; they are re-checked (or clamped) at model build time, where the real values exist.

Violating any validation check raises a `ConfigError` naming the offending keys, stopping execution before any GPU work.

---

## 9. The Dataset: Sources, Construction, and Loading

This chapter is the engineering reference for everything data-side: what a
source dataset must look like, how the build pipeline transforms it, what the
trained-on rows actually contain, how they are loaded/batched in each training
phase, and how the DPO preference dataset differs from all of the above.

The data path is split into two independent products:

| Product | Built by | Consumed by |
|---|---|---|
| **SFT corpus** (Arrow dataset on disk) | `python -m gepard.cli.prepare` → `gepard/data/preprocessing/processor.py` | Pretrain + LoRA fine-tune (`gepard.cli.train`) |
| **DPO preference pairs** (`pairs.jsonl` + token files) | The 4-stage pipeline in `gepard/data/dpo/` | DPO training (`gepard.training.dpo`) |

### 9.1. Source Dataset Contract

`prepare` does **not** read audio. Every source is a Hugging Face dataset (Hub
repo or local `save_to_disk` directory) whose rows already carry **pre-tokenized
codec codes** plus a transcript. The required per-row features are:

| Feature | Type | Meaning |
|---|---|---|
| transcript column | `str` | Raw text of the utterance. Mapped via `text_col_name`. |
| frame count column | `int` | Number of codec frames in the clip. Mapped via `encoded_len`. Used for duration filtering (`encoded_len / frame_rate_hz`) and later by the reference sampler. |
| `nano_layer_1` … `nano_layer_N` | `List[int]` per row | **Packed** FSQ codebook indices, one column per codec layer (N = `codec.num_layers`, 8 for the default codec). Each value is in `[0, 2016)` for the default `[8,7,6,6]` grid. Column names are mapped per source. |
| speaker column | `str` (optional) | Speaker identifier, unique *within the source*. Mapped via `speaker_id_col_name`; required only under `singleton_policy: remove`. |

The number of `nano_layer_*` columns is validated against `codec.num_layers` at
load time (`ItemDataset._validate_layer_columns`) — a mismatch aborts the build
rather than producing silently misaligned channels.

**Producing a compliant source dataset.** The open-source tokenization
pipeline that converts raw audio datasets into exactly this shape (NeMo
NanoCodec 21.5 fps encoding, per-layer packed indices, frame counts) lives at:

> https://github.com/nineninesix-ai/nano-codec-processing-pipeline.git

Run your corpus through it, publish/save the result, and add an entry under
`hf_datasets` (§9.2) — no other integration is needed.

**Try it on public data.** A ready-made, fully public example corpus — already
tokenized with the default codec (NanoCodec 21.5 fps, the exact `nano_layer_1..8`
layout above) and documented by its own dataset card — is:

> https://huggingface.co/datasets/nineninesix/emolia_filtered_nano_codec_21_dataset

It is the one open dataset that runs the whole pretrain pipeline end to end. It
ships pre-wired but commented in `conf/data/full_corpus.yaml`; uncomment its
`hf_datasets` block, then `make dataset` builds a corpus from it.

### 9.2. Per-Source Configuration: the `hf_datasets` Item

The `data` group (`conf/data/*.yaml`) carries a list `hf_datasets`; each entry
describes ONE source and its column mapping. Full key reference (several of
these exist only in code and were previously undocumented):

```yaml
- reponame: nineninesix/emolia_filtered_nano_codec_21_dataset
  local: false                  # true → load_from_disk(reponame) instead of the Hub
  name: null                    # HF dataset config name (rarely needed)
  split: train                  # any HF split expression, e.g. "train[:1000]"
  text_col_name: text           # → renamed to `text`
  encoded_len: encoded_len      # → renamed to `encoded_len`
  nano_layer_1: nano_layer_1    # → renamed per layer; source column names
  # ... nano_layer_2..8 ...     #   may differ per dataset
  speaker_id_col_name: speaker  # → renamed to `speaker_id` (omit if the source has none)
  language_tag: null            # optional; see below
  max_len: null                 # optional row cap AFTER processing (dev/debug)
  speaker_prefix: null          # optional pinned 4-char namespace tag; see §9.4
```

*   **`language_tag`** — when set, every transcript is prefixed as
    `"{tag.lower()}: {text}"` *before tokenization*
    (`TrainDataPreProcessor.create_input_ids`). This is the language-routing
    mechanism: a multilingual corpus can steer the model by language marker
    without any tokenizer change. The tag becomes ordinary text tokens inside
    the `[SOT … EOT]` frame — which means **inference must reproduce the same
    prefix for those languages**, or the model sees an out-of-distribution
    prompt. There is currently no inference-side helper that adds it
    automatically; the caller of `runner.generate(text)` owns this.
*   **`max_len`** — after the source is fully processed, it is shuffled with a
    fixed seed and truncated to `max_len` rows. Debug/dev tool; leave unset in
    production builds.
*   **`split`** — passed verbatim to `datasets.load_dataset`, so HF slice
    syntax (`train[:10%]`) works per source.

### 9.3. Build Pipeline, Stage by Stage

Orchestration is three nested layers (`gepard/data/preprocessing/processor.py`):

```
DatasetProcessor                     (whole run; one per `prepare` invocation)
 ├─ per source: ItemDataset          (load + rename + main-process passes)
 │    ├─ load_dataset / load_from_disk        (num_proc = processing.load_dataset_num_proc)
 │    ├─ validate nano_layer_* count, speaker column presence
 │    ├─ rename columns to canonical names
 │    ├─ synthesize sentinel speaker_id       (speaker-less source, null_prefix only)
 │    ├─ FILTER: non-empty codes AND duration ≤ max_duration_sec
 │    │          (main process, BEFORE sharding — see ordering note below)
 │    ├─ COUNT speakers → low-count set       (main process, AFTER the filter)
 │    └─ shard into processing.num_shards → ProcessPoolExecutor
 │         └─ per shard: TrainDataPreProcessor.transform_row   (one map pass)
 ├─ concatenate all processed sources
 ├─ append row_id                    (only when data.add_row_id: true)
 ├─ save_to_disk(data.train_dataset_path)
 └─ dump speaker_statistics.json     (when data.speaker_statistics: true)
```

**Ordering is load-bearing.** The duration filter runs in the main process
*before* speaker counting. If it ran after (as it historically did, per-shard),
a speaker counted with ≥K clips could drop below K post-filter — a real
speaker id the SupCon sampler would then reject, leaving its rows in no batch
at all. Counting on the already-filtered rows closes that gap. Likewise the
low-count set is computed globally before sharding because shard-local counts
are wrong by construction.

**Cost discipline.** The whole per-row work is fused into exactly two Arrow
passes per source: one combined `filter`, one combined `map(transform_row)`
(code unfolding + sequence assembly + speaker rewrite in a single row visit).
The singleton policy adds zero extra rewrites — `remove` folds into the filter,
`null_prefix` folds into the map.

### 9.4. Speaker Identity and the Singleton Policies

Speaker handling exists to serve voice cloning (§4): the reference sampler
needs to find *other clips by the same speaker*, and SupCon needs speakers
with at least `K` clips. Everything below is active only when
`data.add_speaker_id: true`.

**Cross-source namespacing.** The same string (`"speaker_1"`) means different
people in different sources. Every source therefore gets a 4-character prefix,
and each id is stored as `"{prefix}_{original_id}"`. By default the prefix is
generated from `uuid4()` at run start — which means **speaker ids are not
stable across rebuilds** unless you pin `speaker_prefix:` on the item config.
Pin it if anything downstream (analysis notebooks, cached indices) keys on
speaker id strings.

**The low-count classification.** Per source, speakers with strictly fewer
than `data.min_clips_per_speaker` clips (counted after the duration filter)
are classified low-count. What happens next is `data.singleton_policy`:

| | `remove` (legacy) | `null_prefix` (current default) |
|---|---|---|
| low-count speaker rows | dropped | kept; `speaker_id` rewritten to the sentinel `"__null_ref__"` (`NULL_SPEAKER_SENTINEL`, defined in `gepard/data/collator.py`) |
| source without a speaker column | build error | allowed; a sentinel-valued `speaker_id` column is synthesized for every row |
| training-time effect | — | sentinel rows take the **unconditional path**: the reference sampler emits `force_null=1`, the model swaps the compressor prefix with the learnable `null_prefix` (same branch CFG-dropout exercises, §4.2) |

The rationale: a speaker with 1–2 clips carries no learnable timbre prior, but
its text+audio is still valuable supervision — so instead of discarding it,
route it into the unconditional prefix state that CFG needs anyway. Sentinel
rows are deliberately **not** namespaced, so all of them across all sources
collapse into one bucket the sampler can index directly.

**The `min_clips_per_speaker` ↔ `supcon.K` contract.** The SupCon batch
sampler needs K clips per chosen speaker. The validator (§8.4, invariant 7)
enforces `data.min_clips_per_speaker >= voice_cloning.training.supcon.K` at
composition time — but note this checks *configs*, not the already-built
corpus. If you raise `K` above the threshold the corpus was built with, the
composition fails loudly; if you *rebuild* the corpus with a lower threshold
and keep old K, nothing detects it until the sampler finds under-filled
speakers. Keep the two in lockstep and rebuild when the threshold changes.

**`speaker_statistics.json`.** With `data.speaker_statistics: true`, the build
writes a report inside the output dataset directory: per-source and global
clip-count histograms, `n_supcon_eligible_speakers/rows`,
`n_low_count_speakers/rows`, and `pct_structural_null_exposure` — the fraction
of rows that will take the null path from the data side alone (before
stochastic CFG-dropout is added on top). Check this after every corpus build;
if structural null exposure drifts far from the pretrain corpus, the
conditional/unconditional balance of the model changes with it.

### 9.5. Row Transform: Sequence and Label Layout

`TrainDataPreProcessor.transform_row` produces the exact tensors the model
trains on. Per row:

1.  **Unfold** (`add_codes`): the N packed per-layer indices are decomposed
    little-endian mixed-radix into `num_layers × len(fsq_levels)` = 32
    per-dimension channels (§2.3, `unfold_tokens_np`). With
    `codec.do_unfold: false` the packed layers are kept as 8 channels and
    `fsq_levels` is ignored — supported, but the default model is built around
    the unfolded form.
2.  **Tokenize** the (optionally language-tagged) transcript with the
    configured HF tokenizer, then build the text region via the shared
    `TextRepeater` (§9.6).
3.  **Assemble** inputs and labels with the stop trick:

```
input  : [SOT, txt.., EOT, SOS,  frame_0, ..., frame_m, frame_m(dup)]
labels : [-100] * n_text,        ch_0,    ..., ch_m,    -100            (per channel)
stop   : [-100] * n_text,        0,       ...,  0,       1
```

The last real frame is **duplicated** as the final input position. After the
model's causal shift (`logits[:-1]` vs `labels[1:]`): SOS predicts `frame_0`
and `stop=0`; the last real frame predicts `stop=1` while its codebook label
is `-100` (no audio CE at the stop slot); the duplicated frame itself is
dropped by the shift. This is how a single pass supervises both the 32
codebook heads and the Bernoulli stop head (§6) without a separate EOS frame.

**Output schema** (columns of the saved Arrow dataset — everything else from
the source is dropped):

| Column | Content |
|---|---|
| `text_ids` | `[SOT, …text…, EOT, SOS]`, possibly with repetition copies (§9.6) |
| `level_audio_0..31` | per-channel input codes, length `n_audio + 1` (dup frame) |
| `labels_level_audio_0..31` | `[-100]*n_text + codes + [-100]` |
| `labels_stop` | `[-100]*n_text + [0]*n_audio + [1]` |
| `attention_mask` | all-ones over `n_text + n_audio + 1` |
| `encoded_len` | passthrough frame count (drives durations in the ref sampler) |
| `speaker_id` | namespaced id or `"__null_ref__"` (only when `add_speaker_id`) |
| `row_id` | global row index (only when `add_row_id`; off by default) |

### 9.6. Text Repetition Is a **Prepare-Time** Decision

The adaptive repetition of §5 is implemented once, in
`gepard/data/preprocessing/text_repetition.py` (`TextRepeater`), and consumed
by three parties that must agree byte-for-byte: the corpus build, the
keep-index length recovery (§9.7), and the inference runner. The config group
is `text_layout/`, with two variants:

*   `text_layout: pretrain_stage` — `enabled: false`, plain `[SOT text EOT SOS]`.
    The **pretrain** default (repetition off).
*   `text_layout: repeat_ft` — `enabled: true`, targets ~16 text tokens for
    prompts shorter than 13 tokens, capped at R=8, with a 25% mixed-keep. The
    **fine-tune** layout (`conf/sft.yaml`).

**Repetition is baked into the data, not applied at train time.** The R for
each row is sampled during `prepare` (`sample_R`, with a per-shard-seeded RNG:
`seed + shard_idx`, so builds are reproducible yet shards independent) and the
repeated layout is written into `text_ids`. Changing the `text_layout` group
after the corpus is built changes nothing about training until you rebuild.

**Repetition is a per-phase choice — the two phases:**

| Phase | Corpus built with | Why |
|---|---|---|
| **Pretrain** | repetition **OFF** (`pretrain_stage`) | The short-register defect is not addressed here; pretrain optimizes voice cloning, and batching is governed by the SupCon sampler (§9.9). |
| **LoRA fine-tune** (`conf/sft.yaml`, `make finetune`) | repetition **ON** (`repeat_ft`) | The short-register fix (§5) is the entire point of this phase; the corpus is rebuilt with repetition baked in, and the keep-index (§9.7) re-balances it toward short rows. |

Because the trained checkpoint's behavior depends on which layout it last saw,
the **effective `text_layout` values are stamped into `gepard_config.json`**
(§10.9) and the runner replays the same deterministic policy (`target_R`) at
inference. If you build a new corpus with different repetition parameters,
train on it — the checkpoint then carries the new values automatically. The
only unguarded seam is mixing phases by hand: fine-tuning on a
repetition-corpus from a checkpoint whose config says `enabled: false` (or
vice versa) is legal but silently trains a layout the stamped config will
misreport. Keep the group consistent between `prepare` and `train` invocations
of the same phase.

### 9.7. The Fine-Tune Keep-Index (Duration Reweighting)

To stabilize training on the short register without incurring excessive memory footprint or slow iteration times from extremely long sequences, Gepard implements a dataset weighting and index filtering mechanism.

#### Original Length Recovery
Because text-repetition is baked directly into the prepared dataset's tokenized records, the stored sequence lengths are inflated. The preprocessor recovers the original text token length for each row by scanning the token sequence:
1.  Count occurrences of the `SOT` (Start of Text) token to find the repetition multiplier `R`.
2.  Retrieve the true original text length using the relationship:
    `orig_len = (len(text_tokens) - 1) // R - 2` (accounting for service boundary tokens).

#### Keep-Index Selection
The row-keep index is constructed by `gepard/data/preprocessing/build_keep_index.py`:
*   All rows shorter than the threshold `short_t = 13` are kept.
*   Long rows exceeding `tail_cap = 150` are dropped to prevent out-of-memory errors and slow tail iteration.
*   The remaining eligible long rows are randomly subsampled so that the final subset matches a target fraction (e.g. `target_fraction = 0.5`) of the corpus.
*   An optional `short_oversample` factor can tile short row indices to increase their representational share in the training batch.

This process outputs a `keep_idx_*.npy` array containing the on-disk row indices to keep, and an `orig_len_*.npy` array representing their corresponding original lengths.

#### Loading & Shuffling
The index is wired in via `trainer.keep_index_path` and applied by
`prepare_dataset` **before** the fixed-seed shuffle (on-disk row order is the
index's reference frame) — the exact sequence is §9.8.

### 9.8. Training-Side Loading: `prepare_dataset`

`gepard/data/sampling.py::prepare_dataset` is the single loading path for both
training phases:

1.  `load_from_disk(data.train_dataset_path)`.
2.  Optional `dataset.select(keep_idx)` from `trainer.keep_index_path`
    (§9.7) — **before** the shuffle, since the index refers to on-disk order.
3.  `shuffle(seed=44)` — fixed seed so every distributed rank sees the same
    permutation (rank partitioning happens later, in the sampler/loader).
4.  If `voice_cloning.enabled`: wrap in `ReferenceSamplingDataset` (§9.9).
    Otherwise the plain HF dataset is returned and the legacy no-prefix path
    is used end-to-end.

### 9.9. `ReferenceSamplingDataset`: What Voice Cloning Adds at Load Time

When VC is on, every `__getitem__` returns the precomputed row **plus** a
reference clip for the compressor, selected on the fly:

**Index build (once, at construction).** One scan over `speaker_id` builds
`speaker_to_indices` (real speakers only), `null_ref_indices` (sentinel rows),
per-row durations from `encoded_len / frame_rate_hz`, and a deterministic
`speaker → int` map (sorted, so identical across ranks; sentinel rows map to
`NULL_SPEAKER_INT = -1`). Ref candidates per speaker are pre-filtered to clips
`≥ min_ref_duration_seconds` (3.0s default), falling back to all of the
speaker's clips when none qualify.

**Reference selection policy** (`_select_reference`), in priority order:

1.  **Self-reference fast path** — `reference_sampling.use_self_reference: true`
    (the *fine-tune* setting, `voice_cloning: qformer_frozen`): the target's own
    already-loaded audio is the reference. No random Arrow read, which matters
    for throughput; the compressor is frozen in this phase, and the K=8 query
    bottleneck limits copy-paste leakage exactly as in the singleton fallback.
2.  **Cross-recording** (the *pretrain* setting): a random *other* clip of the
    same speaker, from the duration-qualified candidates.
3.  **Singleton fallback**: same audio as the target.
4.  For null-sentinel rows: no reference at all — a **1-frame zero placeholder**
    is emitted (so the compressor's softmax has one valid key and cannot NaN)
    together with `force_null=1`; the model discards the compressor output and
    substitutes `null_prefix`.

**Stochastic slice.** Every selected reference (cases 1–3) is randomly cropped
to `L ∈ [l_min_seconds, l_max_seconds]` (3–15s → 64–322 frames at 21.5 Hz), at
a random offset — the model must be robust to arbitrary prompt lengths at
inference. Exception: a clip shorter than `singleton_min_target_for_slice`
(6s) in the self/singleton paths is used whole.

Emitted extra keys per sample: `ref_codes` `[T_ref, C]` int64, `ref_len`,
`force_null` (0/1), `speaker_int`.

A performance note: ref reads go through a **slim column view**
(`select_columns` on the `level_audio_*` columns only) so each random pickup
deserializes just the codec channels, not the labels — `select_columns` is
metadata-only on Arrow and preserves row order.

### 9.10. Two Batching Regimes

Batch composition is phase-dependent, and this is a deliberate design fact:

**Pretrain — batching is governed by voice cloning.** With
`supcon.enabled: true`, `MultiheadTTSTrainer.get_train_dataloader` bypasses the
HF Trainer sampler machinery entirely and installs
`SpeakerBucketBatchSampler`. Every batch has the fixed shape **P·K + M**
(16·3 + 16 = 64 = `per_device_train_batch_size`, validated twice — at config
composition and again at dataloader build):

*   `P=16` unique speakers × `K=3` clips each — the SupCon region (positives =
    same speaker, negatives = other speakers);
*   `M=16` rows drawn from the null pool (`force_null` rows) — so the CE and
    stop objectives keep seeing the unconditional path inside every batch;
    they are masked out of the SupCon loss via `speaker_int = -1`. If the null
    pool is empty, fillers come from the eligible pool instead.

Distribution: each rank draws from its own RNG
(`base_seed + epoch·1009 + rank·7`, re-seeded via `set_epoch`); ranks are
statistically disjoint rather than strictly partitioned (with
`N_speakers ≫ P·world_size` overlap is negligible). Batches per rank per
epoch = `eligible_rows // (P·K·world_size)`. Consequence worth knowing:
**epoch coverage is stochastic** — rows are sampled by speaker, not enumerated;
"epoch" is a step-count notion here, not a guarantee that every row was seen.

**Fine-tune — plain batching, weighted by the keep-index.** With
`supcon.enabled: false` (`voice_cloning: qformer_frozen`), the standard HF
random sampler is used; duration/length balance is controlled *offline* by the
keep-index (§9.7), not by the sampler. There is no online length-weighted
sampling anywhere in the loop (the `orig_len.npy` emitted by the index builder
is a hook for one, currently unconsumed).

### 9.11. `DataCollator`: the Batch Contract

`gepard/data/collator.py` pads a list of rows into model tensors. Details that
matter when debugging shapes or extending the pipeline:

*   **Text is LEFT-padded** with `tokens.tts_pad`, so `SOS` sits at position
    `max_text - 1` for every sample and `frame_0` at `max_text` — text and
    audio regions are contiguous for all rows regardless of per-row text
    length. (Right-padding would insert a PAD gap between a short text and its
    audio.)
*   Audio channels are right-padded with `0`; padded positions are excluded
    via `attention_mask` and labels `-100`, so the pad value never trains.
*   **Labels are re-aligned at collate time**: the stored per-row layout is
    `[-100]*text_len + audio_labels`, but the batch layout must place audio
    labels at `max_text`, not `text_len`. The collator strips the per-row text
    mask and re-bases the audio labels — if you ever consume rows without this
    collator, replicate that shift or every loss is off by the padding delta.
*   `attention_mask` marks real text (right-aligned) + real audio; everything
    else is 0.
*   With `vc_enabled=True` it additionally pads `ref_codes` to
    `[B, max_T_ref, C_ref]` (zero-pad) with boolean `ref_mask`, and passes
    through `force_null` `[B]` and `speaker_ints` `[B]`. With
    `vc_enabled=False` these keys are absent and the model takes the
    no-prefix path — the collator flag comes straight from
    `voice_cloning.enabled`.

### 9.12. The DPO Dataset: Why It Is Different

The SFT corpus teaches the model *what speech is*; it cannot teach *which of
the model's own behaviors to prefer* — the runaway/premature-stop defects (§7)
are properties of the trained sampler, so the data to fix them must come from
the model itself. Hence the DPO dataset is **self-generated**, built by a
4-stage pipeline (`gepard/data/dpo/`), all stages driven by the one composed
`conf/dpo.yaml` (`run_name` isolates a round):

```
dpo_data/<run_name>/
  tokens/*.pt          stage 1: per-(text,speaker) rollout token files
  prefixes.pt          stage 1: precomputed frozen Q-Former prefix per speaker
  manifest.jsonl       stage 1: one record per rollout (+ .shard* when sharded)
  scores.jsonl         stage 2: + WER/CER, duration, reward components
  pairs.jsonl          stage 3: chosen/rejected pairs + frozen-ref logprobs
dpo_checkpoints/<run_name>/   stage 4 output (adapters + optional merged export)
```

**Stage 1 — `dpo.sample` (venv_dpo).** Seed prompts come from a JSONL file
(`assets/dpo_seed/short_focused_v2.jsonl`; fields `id`, `uuid`, `text`,
`category`, `lang`, `source` — 2.5k deliberately *short* texts, because that
is the failure register). For each text, `speakers_per_text` reference
speakers are drawn (from `ref_audios` wav files or a `speaker_pool` HF dataset;
`holdout_speakers` are excluded and reserved for unseen-voice evaluation), and
`num_samples` rollouts are generated per (text, speaker) group **in one
batched pass** — prefix and text prefill shared, diversity from sampling.
Three properties are load-bearing:

*   **Runaways are kept, not killed**: generation is truncated at an adaptive
    frame cap (`cap_expected_multiple ×` the reward model's expected duration
    for the text, clamped to `[cap_min_frames, cap_max_frames]`) and enters
    the dataset as a *negative*. The defect must be present in the data to be
    optimized away.
*   **CFG distillation** (§7.4): rollouts are sampled with two-pass text CFG
    (`cfg_scale=3.0`, first 20 frames) — this manufactures positive
    trajectories the single-pass sampler rarely reaches; DPO then bakes the
    difference into single-pass weights.
*   **Text layout parity**: `encode_text_ids(..., repeater=runner.repeater)`
    applies the *same* deterministic repetition policy as production inference.
    If sampling, pair scoring, and the deployed runner disagree on the layout,
    DPO optimizes a different policy than the one being served.

Stage 1 is resumable — groups whose token file exists are skipped.

**Stage 2 — `dpo.score` (venv_dpo).** Decodes tokens to audio (NeMo codec,
batched), transcribes with Whisper, and computes the composite reward:

```
R = -w_wer·WER - w_over·max(0, dur - dur_max) - w_short·1[dur < dur_min]
    - w_empty·1[empty_asr] - w_sim·(1 - speaker_similarity)      (sim optional)
```

Duration bounds are *per-text*: `expected_sec(text)` is a linear word-count
model (`base + per_word · n_words`), scaled by `dur_min_ratio`/`dur_max_ratio`.
The two-sided length term is the critical piece (§7.3): with only an
over-length penalty, preference optimization reward-hacks into stopping at
frame 1 (silence beats an honest attempt, because empty ASR caps WER at 1.0).
Scoring is decoupled from sampling on purpose: reward weights can be changed
and stage 2 re-run without regenerating rollouts.

**Stage 3 — `dpo.pairs`.** Within each rollout group: `chosen` = highest
reward that passes *absolute* quality gates (WER ≤ `chosen_max_wer`, non-empty
ASR, not truncated, duration in bounds); `rejected` = lowest reward with
margin `R_w − R_l ≥ min_reward_margin`; up to `max_pairs_per_group` pairs
(best-vs-worst, then 2nd-vs-2nd). Groups with no gate-passing candidate are
dropped entirely — a "least bad" chosen would teach the wrong target. The
frozen reference model's trajectory logprobs are computed **here, once**
(teacher-forced, batched) and stored in `pairs.jsonl`, so stage 4 never holds
a second model in memory; raw component sums (tokens / stop / T) are stored so
the trainer can switch normalization schemes without recomputation.

**Stage 4** consumes the pairs through its own minimal loader
(`gepard/training/dpo.py::PairsDataset`): `pairs.jsonl` is read up front,
speaker prefixes come precomputed from `prefixes.pt`, and rollout token files
are loaded lazily with a one-group LRU cache (pairs of the same prompt group
sit adjacently, so the cache hit rate is high). Pair order is shuffled per
epoch and sliced into `training.batch_pairs`; both trajectories of a pair are
teacher-forced through the single policy model, and the stored frozen-ref
logprobs complete the (length-normalized) DPO margin
`β·[(π_w − ref_w) − (π_l − ref_l)]` — no second model in memory.

### 9.13. Cross-Config Dependency Map (Data Edition)

What must stay in agreement, and who enforces it:

| Invariant | Enforced by |
|---|---|
| source `nano_layer_*` count == `codec.num_layers` | build-time validation (hard error) |
| `model.audio_heads` == `codec.num_layers × fsq_levels` layout | derived resolver + validator inv. 2 |
| `data.min_clips_per_speaker ≥ supcon.K` | validator inv. 7 (configs only — rebuild corpus when changing either) |
| `P·K + M == per_device_train_batch_size` | validator inv. 8 + a second check at dataloader build |
| `add_speaker_id: true` when VC enabled | validator inv. 6 |
| `text_layout` at prepare == at train == in checkpoint | **not enforced across invocations** — stamped into `gepard_config.json` at export, but keeping prepare/train composition consistent within a phase is on the operator (§9.6) |
| `language_tag` prefix reproduced at inference | **not enforced** — caller-owned (§9.2) |
| keep-index rows refer to on-disk order | `prepare_dataset` applies select-before-shuffle (§9.7) |
| DPO text layout parity (sampling == pairs == serving) | shared `TextRepeater` via the runner; parity by construction as long as all stages load the same checkpoint/config |

---

## 10. The Model: Architecture, Configuration, and Checkpoints

This chapter is the engineering reference for `GepardModel`
(`gepard/model/modeling.py`): exactly which parts are the stock backbone and
which are the TTS overlay, how the model is configured and constructed, and
how checkpoints are written and read back without any training configs. Loss
mathematics is deferred to the TRAIN chapter; the voice-cloning compressor
gets its own chapter — here it appears only as a module in the parameter map.

### 10.1. Anatomy: Stock Core, Thin Overlay

`GepardModel` is a plain `nn.Module` (deliberately **not** a
`PreTrainedModel` — see §10.8 for the checkpointing consequences) wrapping one
stock module and a thin overlay. The parameter-name prefixes below are a
public contract: the per-group learning rates, the freeze maps
(`finetune.freeze_*`), LoRA targeting, and the checkpoint key layout all key
off them.

| Prefix | Module | Origin | Shape drivers |
|---|---|---|---|
| `model.*` | `Qwen3_5TextModel` | **stock transformers** — zero overrides | nested backbone config |
| `audio_embeddings.{0..31}.*` | 32 × `nn.Embedding(L_i, 32)` | overlay | `audio_heads`, `audio_embed_dim` |
| `audio_embed_proj.*` | `Linear(1024→d) → GELU → Linear(d→d) → LayerNorm(no affine)` | overlay | `audio_embed_dim × 32`, backbone `hidden_size` |
| `audio_embed_scale` | scalar **buffer** (not a parameter) | overlay | — (set from text-embedding std) |
| `codebook_heads.{0..31}.*` | 32 × `nn.Linear(d, L_i)` | overlay | `audio_heads` |
| `stop_head.*` | `nn.Linear(d, 1)` | overlay | — |
| `ref_compressor.*` | Q-Former (chapter TBD) | overlay, **only when VC enabled** | `voice_cloning.compressor`, codec geometry |
| `null_prefix` | `nn.Parameter[K, d]` | overlay, only when VC enabled | `compressor.num_queries` |
| `supcon_head.*` | 2-layer projection MLP | overlay, only when VC + SupCon + `use_projection` | `supcon.projection_*` |

What is deliberately **absent**: the backbone's `lm_head` (text generation is
not a task — it is discarded at load, §10.6), and any custom module *inside*
the decoder stack. Everything TTS-specific lives strictly before
`model.embed-level` inputs or after `model` outputs, which is what keeps the
autoregressive loop stock-engine-servable (§1.1).

Text tokens are embedded by the backbone's own `model.embed_tokens` — the
overlay adds no text path of its own, so pretrained text representations are
reused as-is.

### 10.2. The Backbone

*   **Class and variant.** `Qwen3_5TextModel` from `transformers`, built from
    the repo `nineninesix/qwen3_5-full-attn-only-14`: a Qwen3.5 export whose
    `layer_types` are all `"full_attention"` (14 layers, hidden 1024, 8 heads,
    GQA with 2 KV heads). Full-attention-only matters twice: FlashAttention-2
    applies to every layer, and every layer carries the `q/k/v/o_proj` set
    that LoRA targets (mixed linear-attention layers would not).
*   **`partial_rotary_factor` — read this before touching RoPE.** Since
    transformers 5.x the value lives in TWO places: the flat top-level config
    attribute and `config.rope_parameters["partial_rotary_factor"]`. The HF
    model computes RoPE **only from the nested copy**; vLLM reads **only the
    flat copy**; the constructor does not keep them in sync, and the stock
    backbone repo itself ships them diverged (flat 0.25 vs effective nested
    1.0). Gepard therefore treats `model.partial_rotary_factor` (default 1.0
    = full rotary coverage) as authoritative and forces it into **both**
    copies at every seam: model load
    (`GepardModel.from_pretrained(partial_rotary_factor=…)`, applied *before*
    the backbone is built so `inv_freq` is computed with it), checkpoint
    reconstruction (`reconcile_backbone_config` inside `build_model`), and
    every written `config.json` (§10.8). RoPE is parameter-free, so the
    override never conflicts with checkpoint weights. Helpers:
    `set_partial_rotary_factor` / `effective_partial_rotary_factor` in
    `gepard/model/configuration.py`.
*   **Attention implementation** is a runtime choice, not a weight property:
    `flash_attention_2` for training (composed default), `eager` for
    inference/DPO stages (`from_checkpoint(attn_implementation="eager")`
    strips any serialized attn setting and re-applies the override).
*   **Gradient checkpointing** is forwarded to the backbone only
    (`gradient_checkpointing_enable` delegates to `self.model`); the overlay
    stays outside the checkpointed region — its compute is trivial while
    activation memory is dominated by the FFN intermediates.

### 10.3. Audio Interface — Input Side

One audio frame is 32 discrete codes (§2.3). The frame embedding is:

```
level_audio_0..31  ──►  32 × Embedding(L_i, 32)   # per-channel lookup
                   ──►  concat → [B, T, 1024]
                   ──►  Linear(1024→1024) → GELU → Linear(1024→1024)
                   ──►  LayerNorm(elementwise_affine=False)   # unit-norm frame
                   ──►  × audio_embed_scale                    # buffer ≈ text-emb std
```

Design facts an engineer must not "fix" without understanding (§3 has the
full derivation):

*   **MLP, not sum.** A sum of per-codebook lookups is an additive (linear)
    function of the channels; the 2-layer GELU MLP models cross-codebook
    interactions within a frame.
*   **The LayerNorm is affine-free on purpose.** The backbone's input RMSNorm
    makes the audio-embedding *magnitude* a free direction — it would drift
    unbounded and push the GELU toward degenerating into a linear map. The
    affine-free LN pins the scale; do not add affine parameters back.
*   **`audio_embed_scale` is a buffer, not a parameter.** Set once in
    `_init_audio_embeddings()` to the pretrained text-embedding std
    (`embed_tokens.weight.std()`), it rescales the unit-norm frame so audio
    embeddings are in-distribution next to text (the backbone itself discards
    scale via RMSNorm; the match matters for the ref-compressor and
    diagnostics). Being a persisted buffer, it survives checkpoint/resume.
*   **Initialization** (`_init_audio_embeddings`, fresh-training path only):
    tables at unit std, MLP fan-in init (`std = in_features^-0.5`) with zero
    biases — every stage keeps activations ~O(1); the trailing LN makes
    absolute output calibration unnecessary. When a checkpoint is loaded, all
    of this is overwritten anyway.

### 10.4. Output Side

Applied to the backbone hidden states **after slicing off the K prefix
positions** (`hidden[:, K:, :]`): the labels built by the collator span
`[text | audio]` and know nothing about the prefix, so forgetting this slice
breaks the causal-shift alignment silently — it is the single most fragile
index in the model.

*   `codebook_heads` — 32 independent `Linear(1024, L_i)`, 216 logits total
    per position (§2.3: two orders of magnitude cheaper than 8×2016 heads).
*   `stop_head` — `Linear(1024, 1)`, a per-position Bernoulli "this frame is
    the last" predictor (§6.1). At inference it is thresholded
    (`stop_threshold`, default 0.5); in DPO its log-probabilities enter the
    trajectory likelihood (§7.1).

### 10.5. Forward Contract and Runtime Attributes

`forward(text_ids, attention_mask, labels_stop=None, **kwargs)`:

*   Audio inputs and labels arrive **by channel name** in `kwargs`
    (`level_audio_i`, `labels_level_audio_i`) — the dataset columns, the
    collator outputs, and `audio_heads` keys are the same namespace by
    construction, and head order is positional (hence the ordered-dict
    discipline in §10.9).
*   Sequence assembly: `[prefix? | text_embeds | audio_embeds]`, attention
    mask extended with ones over the prefix. Training always runs
    `use_cache=False`; KV-cache generation lives in the runner, not here.
*   Returns `MultiheadTTSOutput(loss, logits_audio: List[32], logits_stop)`.
    Loss composition (CE × 32 + weighted stop BCE + curriculum-gated
    regularizers) is TRAIN-chapter material; architecturally note only the
    NaN-guard (§6.2): a head whose shifted labels are all `-100` contributes
    `logits.sum() * 0` — in the graph, zero gradient.

Runtime attributes that form the model↔trainer interface:

| Attribute | Written by | Read by |
|---|---|---|
| `_global_step` | trainer, before each forward | warmup/ramp curricula of the VC regularizers |
| `_last_losses` | forward | wandb logging callback (per-head losses) |
| `_last_diagnostics` | forward (cheap, every step) | diagnostics callback |
| `_collect_layer_sims` | diagnostics callback (one step per logging interval) | forward — gates `output_hidden_states=True`, ~28 extra `[B,T,d]` tensors, hence gated |

These are plain attributes, not buffers: they do not enter the state_dict and
carry no cross-checkpoint meaning.

### 10.6. Static Model Configuration (`model/` group)

The full per-field reference lives in §8.3.4; only two mechanics are worth
restating here:

*   **`audio_heads` is never hand-written.** An OmegaConf resolver
    (`gepard/config/store.py::_derive_audio_heads`) expands
    `${gepard.audio_heads:${codec.num_layers},${codec.fsq_levels}}` into the
    ordered map `{level_audio_0: 8, level_audio_1: 7, … level_audio_31: 6}` —
    one source of truth with the codec group, re-checked by validator
    invariant 2 (§8.4). Head order is positional wiring (§10.9).
*   **`dtype` strings are alias-tolerant.** `resolve_dtype` accepts
    `bfloat16/bf16`, `float16/fp16/half`, `float32/fp32/float` and the
    `torch.` prefix; anything else raises with the allowed set.

### 10.7. Three Construction Paths

A `GepardModel` comes into existence exactly three ways; know which one you
are on when debugging shapes:

1.  **Fresh training** — `GepardModel.from_pretrained(backbone_id, …)`
    (called by the trainer): loads `AutoConfig`, applies the
    `partial_rotary_factor` override (§10.2), builds the model with a randomly
    initialized overlay, then loads the full `Qwen3_5ForCausalLM` and copies
    its `.model` weights in with `strict=False` (our model has no `lm_head`;
    the base has no audio modules — both key sets are expected mismatches).
    Finally `_init_audio_embeddings()` calibrates the audio stack. A TTS
    checkpoint (`finetune.checkpoint_path`) may then overwrite everything via
    `gepard/training/base.py::load_tts_checkpoint`.
2.  **From a self-describing checkpoint** — `build_model(gepard_config)`
    (called by the runner and the DPO stages): reconstructs the backbone from
    the **nested** config (no Hub access for the architecture), rebuilds the
    exact overlay shape from the serialized fields, reconciles
    `partial_rotary_factor`, returns fresh weights for the caller to fill
    from `model.safetensors`.
3.  **Direct constructor** — tests and the legacy no-`gepard_config.json`
    fallback (`from_checkpoint(fallback=…)`), where the composed config tree
    supplies what the checkpoint cannot.

### 10.8. Checkpoint Formation

Because `GepardModel` is not a `PreTrainedModel`, HF Trainer's own save covers
only weights — everything else is explicit code, in two places:

**Final export** (`GepardTrainer.save_model`, runs on *every* rank):

1.  Under FSDP the parameters are sharded in place, so the full state dict is
    gathered via `accelerator.get_state_dict` — a collective all ranks must
    join; only the main process writes files.
2.  LoRA runs first dump the raw adapter (`lora_adapter.pt`) and then
    **merge** every `LoRALinear` back into its base weight
    (`W += B@A · α/r`), so the exported `model.safetensors` has the stock key
    layout and loads anywhere without LoRA awareness.
3.  Files written: `model.safetensors` (real safetensors — a historical bug
    wrote a torch pickle under that name), `gepard_config.json` (§10.9),
    backbone `config.json` (serving engines and `AutoConfig` read this;
    `partial_rotary_factor` guaranteed in both flat and nested form —
    `patch_config_json_rotary`), and the tokenizer files.

**Periodic `checkpoint-N` dirs** (HF Trainer + two callbacks in
`gepard/training/callbacks.py`): Trainer saves weights/optimizer;
`TokenizerSaveCallback` adds the tokenizer; `GepardConfigSaveCallback` stamps
`gepard_config.json` + `config.json`. One deliberate exception: while
**un-merged LoRA adapters** are live, the config provider returns `None` and
periodic checkpoints are *not* stamped — their state_dict layout
(`*.base.weight`, `*.lora_A/B`) does not match what a vanilla config
describes, and a false self-description is worse than none. Merge first
(final export does), or use `scripts/merge_lora_checkpoint.py` — a CPU-only
state-dict surgery that folds adapters offline, writes `config.json` from the
base model, and carries the tokenizer + `gepard_config.json` over.

Checkpoint file inventory:

| File | Written by | Consumed by |
|---|---|---|
| `model.safetensors` | trainer / merge script | runner, DPO stages, vLLM export |
| `gepard_config.json` | `save_gepard_config` | `TTSRunner/GepardRunner.from_checkpoint`, `build_model` |
| `config.json` | `config.save_pretrained` + rotary patch | vLLM, `AutoConfig`, legacy fallback |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | tokenizer save | runner (`AutoTokenizer`) |
| `lora_adapter.pt` | LoRA export | reuse/inspection; merge script input is the *unmerged* checkpoint instead |

### 10.9. Self-Describing Checkpoints, End to End

The serialization contract in implementation detail — `gepard/model/configuration.py`
+ the `config ↔ model` bridge at the bottom of `modeling.py`:

Config fields split into two categories (tagged in `schema.py`): **model-build `[B]`** fields structurally define the architecture and are serialized into the checkpoint; **trainer/data-only `[T]`/`[D]`** fields (schedules, paths, logging) never leave the training run.

```
       Training Run (composed YAML)
                 │
                 ▼
       Gepard Model Instantiation
                 │
         [Training Phase]
                 │
        Checkpoints Saved
                 │
                 ├──────► model.safetensors (Weights)
                 └──────► gepard_config.json (Model-Build [B] fields only)
```

**Writing.** `config_from_model(model, …)` builds a `GepardConfig` by
**introspecting the live modules**: audio head names/vocabs from
`channel_names`/`vocab_sizes`, `audio_embed_dim` from the embedding tables,
compressor dims from the actual `RefCompressor`, SupCon-head presence from
the module being non-None, the codec shape facts from the compressor
(overriding whatever the caller passed — the config can never disagree with
the weights), and the **effective** `partial_rotary_factor` (nested-first —
recording the flat attribute would have persisted the stock repo's stale
0.25). Only what the model does not hold — the special-token map, the
`text_layout` values, the full codec identity — comes from the composed
config at save time. Two serialization quirks are load-bearing:

*   `GepardConfig.to_json_string` overrides the parent to serialize
    **without key sorting**: `audio_heads` order IS the head wiring
    (lexicographic sorting would put `level_audio_10` before `level_audio_2`
    and silently permute every head on reload).
*   Unknown keys from older files (e.g. the removed `aligner` block) are
    swallowed as plain attributes by `PretrainedConfig` — old
    `gepard_config.json` files keep loading after schema evolution; stale
    weight keys in old checkpoints are then dropped by `strict=False`.

**Reading.** `from_checkpoint` resolves files through
`gepard/model/checkpoint_io.py::resolve_checkpoint_file` (local file / local
dir / HF Hub, uniformly), then:

```
gepard_config.json found ──► build_model(cfg, attn_implementation=…)
                             └ backbone from nested config, rotary reconciled
     state_dict ◄── model.safetensors, load_state_dict(strict=False)
     tokenizer  ◄── AutoTokenizer(checkpoint)
     runner     ◄── special_tokens + text_repetition from the SAME config
no gepard_config.json      ──► fallback= composed config tree (legacy), or
                               FileNotFoundError with re-export guidance
```

`strict=False` semantics at this seam: *unexpected* keys (modules removed
since the checkpoint was written) are dropped silently by design; *missing*
keys stay at random init — harmless only for train-only modules (e.g. a
SupCon head absent from a fine-tune lineage checkpoint). Both lists are
printed at load; an unexpected key you cannot name is a red flag, not noise.

**`config.json` vs `gepard_config.json` — roles, not duplication.** The two
files coexist by design and never collide (different filenames, different
readers): `config.json` is the *backbone-only* HF-standard config for the
external ecosystem (vLLM serving, `AutoConfig`, Hub tooling) and knows nothing
of audio heads; `gepard_config.json` is the complete self-description
(backbone nested inside it) that lets the runner rebuild the model with zero
Hub access and zero training YAMLs. Deleting `config.json` breaks serving;
deleting `gepard_config.json` degrades the checkpoint to the legacy
fallback path.

---

## 11. Voice Cloning: the Cross-Cutting Subsystem

Voice cloning is the one feature that touches **every** stage of the project —
there is no "VC module" you can read in isolation. This chapter owns the
compressor itself and the design reasoning; everything else is cross-referenced
to where it lives:

| Piece | Where it lives | Reference |
|---|---|---|
| speaker labeling, singleton → null routing | dataset build | §9.4 |
| reference clip selection (cross-recording / self / null) | `ReferenceSamplingDataset` | §9.9 |
| **batch formation (P·K+M speaker buckets)** | `SpeakerBucketBatchSampler` | **§9.10** |
| ref tensor padding, `force_null`, `speaker_ints` | `DataCollator` | §9.11 |
| prefix path in the decoder, CFG-dropout OR-gate | `GepardModel.forward` | §10.5, §4.2 |
| SupCon / diversity loss mathematics | losses | TRAIN chapter (overview: §4.3) |
| prefix at generation time, CFG interplay | runner | §11.6 |
| per-speaker prefix precompute for rollouts | DPO stage 1 | §9.12 |

Worth stating plainly what makes this subsystem interesting as engineering:
speaker identity is carried **entirely in activation space** — K prefix vectors
computed on the fly from the same discrete codec codes the decoder already
speaks. There is no speaker embedding table, no enrollment step, no per-voice
fine-tune, no separate audio encoder (mel/SSL) at serving time, and the whole
path is prefill-only, so it costs nothing in the autoregressive loop and
survives stock-engine serving (§1.1). The flip side: it is the least mature
part of the recipe, and several of its current choices are pragmatic first
passes rather than validated optima — see the training-strategy note in §11.5.

### 11.1. Input Representation: Codec Codes, Not Waveforms

The compressor consumes a reference clip as `[B, T_ref, C]` **discrete FSQ
codes** — the exact currency of the rest of the model — not waveforms or
spectrograms. Codes are dequantized to their FSQ lattice values in `[-1, 1]`
(`dequantize_codes`), so the input is a compact float matrix
(`C_total = 32` channels at 21.5 fps) that is already speech-specific.
Consequences:

*   Any audio the codec can encode is a valid reference; at serving, encoding
    the user's clip through the codec is the only preprocessing.
*   No second audio frontend to ship, version, or keep in dtype/device sync.
*   The compressor tolerates both on-disk layouts: if the dataset carries
    packed per-layer codes (`codec.do_unfold: false`), it unfolds in-forward;
    with the default unfolded datasets it consumes them as-is
    (`do_unfold_in_forward = not codec.do_unfold`).

### 11.2. `RefCompressor` Architecture

`gepard/model/ref_compressor.py` — a Q-Former-style bottleneck:

```
ref_codes [B, T_ref, C] ─ dequantize ─► Linear(C → d) ─ + sinusoidal PE ─► ref_feats
queries   nn.Parameter[K=8, d=1024]  ─ batch-expand ──────────────────────► q

× L=2 blocks (pre-norm RMSNorm everywhere):
    q = q + SelfAttn(q)                        # bidirectional, queries only
    q = q + CrossAttn(q ← ref_feats, key_padding_mask=ref_mask)
    q = q + SwiGLU_FFN(q)

q_normed = RMSNorm(q)                          # RMS = 1 per token
prefix   = output_scale · q_normed             # what the decoder consumes
```

Facts with reasons:

*   **Queries are position-less**; order carries no meaning. The reference
    gets sinusoidal PE so the cross-attention can exploit temporal structure
    of the clip, but the *output* is a set, not a sequence.
*   **`output_scale` starts at `1/√d_model`.** RMSNorm alone gives per-token
    RMS = 1, i.e. L2 ≈ √d ≈ 32 — which would dwarf the text/audio embeddings
    (scale ~1) it sits next to in the decoder input. The learnable scalar
    starts the prefix at L2 ≈ 1 and lets training rescale if useful. This is
    the prefix-side mirror of `audio_embed_scale` (§10.3): every input stream
    into the backbone gets its scale pinned explicitly.
*   **Dual output `(prefix, q_normed)`.** The regularizers (diversity, SupCon)
    operate on `q_normed` — the pre-`output_scale` space where RMS = 1 — so
    their thresholds (`γ`) stay in natural units and are not silently rescaled
    as `output_scale` learns. The decoder consumes `prefix`.
*   Attention is plain SDPA with a key-padding mask over reference padding;
    self-attention over the 8 queries needs no mask.
*   `d_model = 1024` equals the backbone hidden size (config allows overriding,
    but the match is enforced at model build time, §8.4 — the prefix is
    injected directly into the decoder's embedding stream with no adapter).

### 11.3. The K-Query Bottleneck Is the Central Design Bet

`num_queries: 8` is not a capacity knob to casually raise. The reconstruction
objective actively incentivizes the compressor to smuggle *content* — copy the
reference's spectral sequence so the decoder can cheat — rather than abstract
timbre (§4.3). Three mechanisms hold that leakage down, and K is the
structural one:

1.  **K=8 tokens for a 64–322-frame reference** is an 8–40× temporal
    compression: there is simply no room to encode the frame sequence.
2.  The **stochastic slice** (§9.9) changes which part of the clip the
    compressor sees each epoch, so memorizing alignment does not pay.
3.  The **regularizers** (diversity + SupCon) shape the surviving capacity
    toward speaker-discriminative, content-invariant features.

The bottleneck is also what makes the *same-audio* reference paths safe: the
singleton fallback and the fine-tune self-reference mode (§9.9) both feed the
target's own audio as the reference, and tolerating that without the model
degenerating into copy-through relies on K being small.

### 11.4. The Unconditional Path: `null_prefix`

A learnable `nn.Parameter[K, d]` (init std 0.02) that **replaces** the
compressor output per sample via an OR of two triggers
(`_maybe_apply_cfg_dropout`, §10.5):

*   stochastic CFG-dropout at `cfg_dropout_prob = 0.15` (training mode only);
*   `force_null` — rows whose speaker is the data-side null sentinel (§9.4);
    applied unconditionally, eval included.

This single mechanism serves three consumers: classifier-free guidance needs a
*trained* unconditional branch (§5, §7.4); no-reference generation at serving
falls back to it implicitly; and it monetizes the low-count-speaker rows the
dataset would otherwise have to drop. The observed random/forced rates are
reported separately in diagnostics (`cfg_dropout_rate_observed`,
`forced_null_rate`) so drift in data composition is distinguishable from the
CFG coin flip. Note the interplay with batching: the M null rows per SupCon
batch (§9.10) guarantee the unconditional path is exercised in *every* batch,
not just on the 15% coin flip.

### 11.5. Training Strategy Across Phases (and Its Openness)

| Phase | Compressor | References | Regularizers | Batching |
|---|---|---|---|---|
| **Pretrain** | **trained jointly** with the backbone, at `lr_multiplier: 0.1` (compressor + `null_prefix` group, TRAIN chapter) | cross-recording, stochastic 3–15s slice | diversity + SupCon, warmup-ramped | governed by VC: P·K+M speaker buckets (§9.10) |
| **SFT / LoRA fine-tune** | **frozen** (`freeze_ref_compressor: true` via `voice_cloning: qformer_frozen`) | self-reference fast path | off | plain random + keep-index (§9.10) |
| **DPO** | frozen; prefix computed **once per speaker** and cached (`prefixes.pt`, §9.12) | fixed per-speaker clips / speaker pool | — | pair batches |

The joint-pretrain / frozen-SFT split is a deliberate simplification of the
first release, not a validated optimum: freezing at SFT protects the speaker
space from being warped by the short-phrase objective, at the cost of never
adapting the compressor to the fine-tune distribution (self-references,
repetition-heavy prompts). **This schedule is expected to be revisited**, and
it is exactly the kind of surface where outside contributions can move the
needle without touching the serving contract — the prefix interface
(`[K, d]` at prefill) stays fixed while everything about how it is produced
and trained is open. Directions we consider promising, in rough order of
effort:

*   unfreezing the compressor at SFT with a small LR (the `lr_multiplier`
    plumbing already exists);
*   multi-clip / longer references — the architecture already accepts
    arbitrary `T_ref`, only the sampling policy (§9.9) limits it;
*   a K sweep with leakage-vs-similarity measurement (K=8 was chosen by
    design reasoning as a leakage bottleneck, not swept empirically);
*   richer SupCon feature reduction than mean-over-queries;
*   cross-lingual reference robustness (reference language ≠ target language).

PRs are welcome — the baseline numbers in the tech report and the golden-loss
test harness (`tests/`) make regressions cheap to catch.

### 11.6. Generation-Time Usage

*   **Runner path.** `runner.generate(text, ref_codes=…)` →
    `_compute_ref_prefix`: one compressor call during prefill, prefix
    prepended to the text embeddings, and the autoregressive loop never sees
    the compressor again (§1.1). `ref_codes` at inference are `[1, T_ref, 32]`
    unfolded codes — produce them with `UnfoldedCodecModel.encode` +
    `unfold_tokens` (see `notebooks/inference_demo.ipynb`).
*   **CFG shares the prefix.** In `GepardRunner` both the conditioned and the
    unconditioned branch carry the **same** speaker prefix; only the text is
    removed from the uncond branch. The speaker prior is common mode and
    cancels in `logit_uncond + w·(logit_cond − logit_uncond)` — guidance
    amplifies the *text* direction specifically, which is why text-CFG fixes
    prefix dominance instead of fighting the voice (§5).
*   **No reference** → no prefix at all (legacy path), which works because the
    null/CFG training exposed the model to text-only conditioning; passing
    nothing is not the same as passing `null_prefix`, but both are
    in-distribution.
*   **DPO rollouts** reuse the frozen compressor once per speaker
    (`compute_speaker_prefix`) — trajectories within a group differ only by
    sampling, which is what makes the group's preference pairs attributable
    to generation behavior rather than prefix noise.

### 11.7. Config Surface and the Enable/Disable Contract

One group, three variants, mapping 1:1 to the phase table above:
`voice_cloning: qformer_supcon` (pretrain), `qformer_frozen` (SFT),
`disabled`. The `enabled` flag is honored consistently everywhere — flipping
it changes, in one move: whether `ReferenceSamplingDataset` wraps the dataset
(§9.8), whether the collator emits `ref_codes/ref_mask/force_null/speaker_ints`
(§9.11), whether the model constructs `ref_compressor`/`null_prefix`/
`supcon_head` (§10.1), which batching regime applies (§9.10), and what the
checkpoint self-description records (`voice_cloning` block in
`gepard_config.json`, §10.9 — including the compressor dims introspected from
the live module, so a VC checkpoint rebuilds byte-identical at inference).
Validator invariants 4, 6–8 (§8.4) hold the cross-group consistency.

---

## 12. Training: Pretrain, Fine-Tune, DPO

The operational reference for the three training phases: what each `make`
command actually launches, how the runs are distributed, how the losses and
their curricula are composed, and — in detail — what the wandb telemetry means
and why each metric exists. Checkpoint *contents* are §10.8; this chapter
covers when and how they are produced and resumed.

> **Hydra in one minute (for those who skipped §8).** Every run is configured
> by *composing* one entry file (`conf/train.yaml`, `conf/prepare.yaml`,
> `conf/dpo.yaml`) with one YAML per **group** (`tokens/`, `codec/`,
> `trainer/`, …) chosen in its `defaults:` list. The composed tree is then
> type-checked against dataclasses (`gepard/config/schema.py`) and
> cross-validated (§8.4). Anything can be overridden from the command line
> (`trainer.learning_rate=1e-4`), a whole group can be swapped
> (`voice_cloning=disabled`), and whole **phases** are separate entry files
> (`train.yaml` pretrain, `sft.yaml` fine-tune, `dpo.yaml`) rather than presets.
> No YAML is ever copied per run; a run differs from its entry by exactly its
> overrides — which makes every run reproducible from its command line.

### 12.1. Command → What Actually Runs

| Command | Launches | Env | Distribution |
|---|---|---|---|
| `make dataset` | `python -m gepard.cli.prepare` | venv | multiprocess shards (§9.3) |
| `make train` | `accelerate launch --config_file accelerate/pretrain_fsdp.yaml -m gepard.cli.train` | venv | FSDP, 4 GPU |
| `make resume CHECKPOINT=…` | same + `run.resume_from=…` | venv | FSDP, 4 GPU |
| `make finetune` | `accelerate launch --config_file accelerate/finetune_single.yaml -m gepard.cli.sft` | venv | single GPU |
| `make finetune-resume CHECKPOINT=…` | same + `run.resume_from=…` | venv | single GPU |
| `make dpo-sample [DPO="…"]` | `python -m gepard.data.dpo.sample` | **venv_dpo** | 1 process (see sharded) |
| `make dpo-sample-sharded SHARDS=4` | N parallel `dpo.sample --shard i/N` | venv_dpo | N processes, one GPU |
| `make dpo-progress SHARDS=4` | `python -m gepard.data.dpo.progress` | venv | read-only monitor (2nd terminal) |
| `make dpo-score` | `python -m gepard.data.dpo.score` | **venv_dpo** | 1 GPU |
| `make dpo-pairs` | `python -m gepard.data.dpo.pairs` | venv | 1 GPU |
| `make dpo-dataset` | stages 1→2→3 in sequence | mixed | — |
| `make dpo-train` | `python -m gepard.training.dpo` | venv | 1 GPU, manual loop |
| `make merge CHECKPOINT=…` | `scripts/merge_lora_checkpoint.py` | venv | CPU-only |
| `make upload REPO=… CHECKPOINT=…` | `scripts/upload_to_hf.py` (bf16, no optimizer) | venv | — |
| `make upload … TRAINING_STATE=1` | same, resume snapshot: + optimizer, weights stay fp32 | venv | — |

Conventions: `DPO="…"` passes Hydra overrides to any DPO stage
(`make dpo-sample DPO="run_name=round4"`); `NVME=/path` redirects `HF_HOME`
to fast ephemeral storage for `make dataset`; the HF token is picked up from
the standard cache location automatically. `dpo-sample-sharded` exploits the
fact that AR decoding at batch 8 leaves the GPU mostly idle — N independent
resume-safe processes (~3 GB VRAM each) scale near-linearly.

Sampling is instrumented with the shared Gepard logger (`gepard/logging/`):
each shard prints a rich config banner (shard 0) and writes its own structured
file under `logs/dpo_sample_<run_name>/shard{i}.log` (single-process runs →
`sample.log`) — NeMo / runner / wandb output is left untouched on stdout, only
our lines are filed. Because the sharded launcher holds the first terminal on
`wait`, run `make dpo-progress SHARDS=N` in a **second** terminal for one
aggregated live progress bar (overall + per-shard, with ETA); it is read-only
(polls token files + per-shard manifests) and safe to start/stop at any time.

`gepard.cli.train` itself is thin: compose config → `require_dataset_built` →
`GepardTrainer(cfg).setup()` → `train(resume_from_checkpoint=cfg.run.resume_from)`
→ final `save_model(output_dir/"final")`. It also sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch import
(fragmentation guard for long runs).

### 12.2. The Two Accelerate Profiles

*   **`accelerate/pretrain_fsdp.yaml`** — FSDP **v2**, 4 processes, bf16
    mixed precision. Sharding is `TRANSFORMER_BASED_WRAP` on
    `Qwen3_5DecoderLayer` — each decoder layer is a shard unit; the overlay
    (embeddings, heads, compressor) rides along. `FULL_STATE_DICT` keeps
    checkpoint tensors un-sharded on save; `reshard_after_forward: true`
    trades a re-gather per backward for peak-memory headroom;
    `cpu_ram_efficient_loading` streams weights at init. Note
    `fsdp_activation_checkpointing: false` — activation checkpointing is
    instead requested through the trainer (`trainer.gradient_checkpointing`,
    forwarded to the backbone only, §10.2).
*   **`accelerate/finetune_single.yaml`** — `distributed_type: NO`, one
    process. Deliberate: the 555M model plus LoRA adapters fit on one device,
    and sharding would only add overhead and complicate the plain checkpoint
    layout the LoRA export relies on.

Effective batch sizes: pretrain `64 × 2 GA × 4 GPU = 512`; fine-tune
`16 × 4 GA × 1 = 64` (the pretrain per-device 64 is itself pinned by the
SupCon shape `P·K+M = 16·3+16`, §9.10).

### 12.3. Phase I — Pretrain

Composition: the `conf/train.yaml` defaults as-is (`trainer: pretrain`,
`text_layout: pretrain_stage` → repetition off (§9.6), `voice_cloning:
qformer_supcon`, `finetune: none` → no checkpoint load, no freezing —
everything trains from the Qwen3.5 init).

**Provenance.** `run.stage: pretrain` is frozen — together with the full
resolved config, step and UTC timestamp — into `training_metadata.json` in
every checkpoint (final export and each periodic `checkpoint-N`, via
`TrainingMetadataCallback`). Because it is the *resolved* config, it captures
CLI/experiment overrides and never drifts when `conf/` is edited later. At
publish time `make upload` renders a Hugging Face README model card from that
file plus `gepard_config.json` (`gepard/logging/model_card.py`).

**Schedule** (`conf/trainer/pretrain.yaml`): LR `9e-4`, cosine, warmup 500;
8 epochs; `max_grad_norm 3.0`; `adamw_torch_fused`; bf16;
`weight_decay 0.0`; gradient checkpointing ON.

**Per-group learning rates** (`MultiheadTTSTrainer.create_optimizer`). One
cosine schedule, four parameter groups (each split decay/no-decay the same
way HF does — biases and LayerNorm excluded from decay):

| Group | Param prefixes | LR | Why |
|---|---|---|---|
| backbone | everything else (`model.layers.*`, norms) | `9e-4` | the reference rate |
| audio | `audio_embeddings. / audio_embed_proj. / codebook_heads. / stop_head.` | ×`0.5` | fresh-init modules sitting directly on the loss — full LR overshoots (§3.2) |
| embed | `model.embed_tokens.` | ×`0.2` | pretrained text table must adapt *gently*, not re-rotate away from Qwen3.5 semantics (see the `cos_to_init` telemetry, §12.7) |
| ref | `ref_compressor. / null_prefix / supcon_head.` | ×`0.1` (from `voice_cloning.compressor.lr_multiplier`) | the compressor learns a *representation*, not a task head; slow LR keeps it from chasing the CE objective into content leakage (§11.3) |

The group census is printed at startup (`[LR groups] backbone … @ lr=… | audio … | embed … | ref …`) —
check it after any module rename: **group membership is prefix-matched**, and
a renamed module silently falls into the backbone group.

**Loss composition** (formulas; forward-pass wiring in §10.5):

```
L = Σ_{c=0}^{31} CE_c                                # per-channel, ignore_index=-100
  + λ_stop · BCE(stop_logits, stop_labels; pos_weight=25)     # λ_stop = 2.0
  + w_div(t)  · mean(relu(γ − std_over_queries(q_normed)))    # γ = 0.5
  + w_sup(t)  · SupCon(feats, speaker_ints; τ = 0.1)
```

*   **32 × CE** — independent heads over one shared hidden state; the shift
    (`logits[:-1]` vs `labels[1:]`) plus the §9.5 label layout give every
    real frame a target and route the stop supervision correctly.
*   **Stop BCE** — the positive class occurs ~once per 150 frames;
    `pos_weight=25` and `λ_stop=2` keep the head from collapsing to
    "never stop" (§6.1).
*   **Diversity (hinge-variance)** — computed on `q_normed` (RMS=1 space,
    §11.2): penalizes per-dimension std across the 8 queries below `γ=0.5`;
    bounded in `[0, γ]`. Prevents query collapse into one averaged vector.
*   **SupCon** (`gepard/model/losses/supcon.py`, Khosla et al. 2020) —
    features = mean-pool of `q_normed` over queries → 2-layer projection head
    (SimCLR-style, ~260K params; keeps the contrastive pressure in a
    disposable space so the upstream representation stays rich) → L2
    normalize → temperature 0.1. Positives = same `speaker_int`, negatives =
    rest of the (gathered) batch; null rows (`speaker_int = −1`) and
    self-pairs are masked out. Under FSDP the features are **all-gathered
    across ranks** (effective contrastive batch = `world × P·K`), MoCo-style:
    remote shards detached, the local shard spliced back with gradient.

**Warmup-ramp curricula.** Both regularizers are OFF for the first
`warmup_start=1000` optimizer steps, then their weight ramps linearly to the
configured maximum (`diversity: weight 0.3 over 1500 steps`,
`supcon: weight 0.3 over 500 steps`):

```
w(t) = weight · clamp((t − warmup_start) / ramp_steps, 0, 1)
```

The rationale is ordering: CE must first establish coarse phoneme structure —
regularizing a compressor that has no signal yet just adds noise; SupCon in
particular would shape speaker geometry on features that don't encode
speakers yet. The step counter is `model._global_step`, written by the
trainer before every forward (§10.5) — the model itself has no notion of
optimizer time.

**Numerical safety nets** (all live in `training_step` / `forward`):
the per-head CE NaN-guard for all-masked microbatches (§6.2), and a
post-backward `torch.nan_to_num_` sweep over every `.grad` — a single bad
microbatch (bf16 logit overflow) must not poison the fp32 master weights.
Both guards exist because of a real incident: a healthy LoRA run collapsed to
NaN in one step at 135k steps and the clean weights were lost to checkpoint
rotation (tech report §4.4). Related discipline: keep backups outside
`output_dir` — HF Trainer's `save_total_limit` rotation deletes anything
matching `checkpoint-<N>`.

### 12.4. Phase II — LoRA Fine-Tune

The fine-tune is a **first-class entry**, `conf/sft.yaml` (a sibling of
`train.yaml`, **not** an experiment preset), launched by `make finetune`:

```
accelerate launch --config_file accelerate/finetune_single.yaml -m gepard.cli.sft
```

Same driver (`gepard/cli/train.py::main`, called with `config_name="sft"`),
`GepardTrainer` and TrainConfig schema as pretrain — only the composed groups
differ. `sft.yaml` selects: `trainer: finetune_lora`,
`voice_cloning: qformer_frozen`, `finetune: sft_lora`, `text_layout: repeat_ft`,
`run.stage: sft`. The single-GPU accelerate profile (no FSDP) keeps the LoRA
checkpoint layout plain. What each group does:

*   **Model init** (`conf/finetune/sft_lora.yaml`): weights loaded from the
    pretrain TTS checkpoint (`finetune.checkpoint_path`), then
    `inject_backbone_lora` (`gepard/training/base.py`): the *entire* model is
    frozen and LoRA adapters (r=16, α=32, dropout 0.05) are injected into
    `q/k/v/o_proj` of the last 16 backbone layers (capped at the model's 14 —
    the cap is defensive). Trainable: adapters only; audio heads, stop head,
    compressor and null_prefix stay frozen.
*   **Schedule** (`conf/trainer/finetune_lora.yaml`): LR `2e-4` (adapters train
    ~10× hotter than a full fine-tune), warmup 150, 3 epochs, `max_grad_norm
    1.0`, gradient checkpointing OFF (no backward through the frozen base →
    nothing to recompute), LR multipliers left at 1.0 (their groups are frozen).
*   **Voice cloning** (`qformer_frozen`): the speaker prefix stays present (input
    distribution unchanged) but the compressor is **frozen**, SupCon/diversity
    are OFF and references use the self-reference fast path → plain random
    batching (no `P·K+M` sampler; §9.10). Using `qformer_supcon` here would fail
    validation (`per_device_batch 16 ≠ P·K+M 64`) — it belongs to pretrain only.
*   **Text repetition** (`repeat_ft`, ON): the short-register fix is the whole
    point (§5). It is a **prepare-time** decision (§9.6), so the SFT corpus must
    be *built* with repetition — `prepare.yaml` defaults to pretrain (off), so
    build it with `make dataset text_layout=repeat_ft`. `text_layout: repeat_ft`
    in `sft.yaml` independently stamps `enabled: true` into the checkpoint's
    `gepard_config.json`, so inference replays the same layout. The keep-index
    (§9.7) reweights toward short rows.

**Provenance.** As in pretrain (§12.3), `run.stage: sft` plus the full resolved
config are frozen into each checkpoint's `training_metadata.json`.

**Publishing a fine-tune checkpoint (merge → upload).** LoRA adapters change the
state-dict layout, so a periodic `checkpoint-N` is **not** self-describing: it
holds raw `lora_A`/`lora_B` + frozen base and no `config.json` /
`gepard_config.json` (those are stamped only once adapters are merged, §10.8).
To publish one:

1.  `make merge CHECKPOINT=checkpoints/checkpoint-N` folds the adapters
    (`W = base + (B·A)·α/r`) into a stock `nn.Linear` layout in
    `checkpoint-N-merged/`, and rebuilds the sidecars from the **fine-tune base**
    — read from `training_metadata.finetuned_from`, **not** the stock backbone:
    `config.json` is copied verbatim from the base (preserving its
    `partial_rotary_factor`, which pretrain may have changed); `gepard_config.json`
    is the base's with `text_repetition` overridden to the fine-tune's `repeat_ft`;
    `training_metadata.json` is carried forward.
2.  `make upload REPO=… CHECKPOINT=checkpoints/checkpoint-N-merged` — the merged
    dir is self-describing, so the un-merged-LoRA guard passes and the README
    model card renders real provenance: stage `sft`, step, and `base_model` →
    the pretrain parent with `base_model_relation: finetune` (§10.9).

(The final `save_model` export at end of training merges + stamps automatically;
`make merge` is only for a mid-run `checkpoint-N`.)

### 12.5. Phase III — DPO Training

`gepard/training/dpo.py` is a **manual loop**, not an HF Trainer: pairs are
fixed-size records (§9.12), the model is mostly frozen, and the loop fits in
a page. Key facts (`conf/dpo/training/default.yaml`):

*   Model = sampling checkpoint + `inject_backbone_lora` (r=16, α=32,
    dropout 0.0, targets `q_proj, v_proj` — narrower than SFT) **plus the
    stop head unfrozen** at its own LR: two optimizer groups, LoRA at `1e-5`,
    `stop_head` at `1e-4`. Training the stop head directly is the point —
    the runaway/premature-stop behavior lives there (§7).
*   Loss: length-normalized reference-anchored DPO (§7.2), `β=2.0` (large
    because per-frame-normalized logprob differences are small numbers),
    stop Bernoulli terms inside the trajectory likelihood with
    `p_floor=1e-4` clipping (§7.1). Optional per-pair weighting by reward
    gap: `reward_weight_mode: linear`, `scale 3.5` — weights normalized to
    keep the batch loss scale, only redistributing gradient toward
    high-contrast pairs.
*   Batching: `batch_pairs 4 × grad_accum 4`, 2 epochs, warmup 100,
    clip 1.0, per-epoch pair shuffle (seed 17).
*   Saving: adapter snapshots every 500 steps
    (`dpo_checkpoints/<run>/adapter-step-N.pt`: LoRA + stop_head + step),
    `adapter-final.pt`, and with `save_merged: true` a fully merged
    `merged/` dir that `GepardRunner.from_checkpoint` loads directly — this is
    what becomes the next round's sampling checkpoint. The merged dir is
    **self-describing**: `model.safetensors`, `config.json` (rotary duplication
    re-applied) and `gepard_config.json` + `tokenizer` are all copied from the
    **policy checkpoint** (`sampling.checkpoint`, i.e. the SFT model — never the
    stock backbone, so a pretrain-changed `partial_rotary_factor` survives), and
    `write_dpo_training_metadata` stamps `training_metadata.json` (stage `dpo`,
    step, full DPO recipe, and `finetuned_from` = the policy). So publishing is
    just `make upload REPO=… CHECKPOINT=dpo_checkpoints/<run>/merged` — the
    un-merged-LoRA guard passes and the README model card renders real
    provenance (stage `dpo`, `base_model_relation: finetune` → the SFT policy,
    the SFT→DPO lineage rather than a raw fine-tune of the base LM; §10.9).
*   wandb: project `kanitts3-multihead` (configurable), optional `entity`
    (team/org), run name defaults to `run_name`; logs
    `loss`, `acc` (fraction of pairs where the policy margin is positive),
    `margin` (`π_w − π_l`, length-normalized), `pi_chosen`/`pi_rejected`,
    and per-frame stop logprobs for both sides — `acc` rising while `margin`
    widens is the expected signature; `pi_rejected` collapsing toward the
    floor with `pi_chosen` flat signals reward hacking (§9.12).

Rounds are isolated by `run_name`; a new round = new seed prompts or new
sampling checkpoint + `make dpo-dataset && make dpo-train DPO="run_name=…"`,
each round sampling from the previous round's merged export. The shipped model
(`nineninesix/gepard-1.0`) is the output of the **second** DPO round: two
consecutive rounds run with **identical settings** (same reward, pairs and
training config — the second is a fresh sample→pairs→train cycle on top of
round 1, not a hyperparameter change). Later rounds were explored but did not
improve on round 2, so that is the production checkpoint.

### 12.6. Checkpointing and Resume

*   **Periodic**: every `save_steps` (1000 SFT / 500 DPO), rotation by
    `save_total_limit` (10 pretrain / 50 fine-tune). Contents and the
    self-description rules: §10.8.
*   **Resume**: `run.resume_from` → HF Trainer's `resume_from_checkpoint`
    (optimizer/scheduler/RNG restored). The SupCon batch sampler is re-seeded
    per epoch (`set_epoch`), so a resumed epoch reproduces its batch stream.
*   **Final export**: after `trainer.train()` returns, `save_model` writes
    `<output_dir>/final` (merged, stamped, servable — §10.8).

### 12.7. wandb Observability — What We Log and Why

Telemetry is not decoration here: three of the model's structural fixes
(gradient starvation §3.2, modality scale mismatch §3.1, prefix-dominance
§5) were *found* through these metrics, and the thresholds below are how we
distinguish "adapting" from "degrading" without waiting for audible failures.
Logging is per-step (`logging_steps: 1`); heavyweight metrics run every
`trainer.expensive_metrics_every` steps (100 pretrain / 200 fine-tune).
Callbacks: `MultiheadLossLogCallback` + `DiagnosticsCallback`
(`gepard/training/callbacks.py`).

**Losses (`loss/*`, every step).** Per-head CE (`loss/level_0..31`),
`loss/stop`, `loss/supcon`, `loss/diversity`, `loss/total`. Per-head curves
expose a lagging codebook channel immediately (one flat `level_k` among 31
falling ones = data or unfold bug in that channel). Note the aggregation
difference: HF's own `loss` is a running mean across accumulation steps,
`loss/total` is the last-microbatch snapshot — they legitimately disagree;
trust `loss/total` for "what is happening right now".

**Gradient norms (`grad_norm/*`, every step).** Read from `p.grad` at the
`on_pre_optimizer_step` event — after gradient accumulation and unscaling,
before clipping and the step, the only point where grads are guaranteed
present under every wrapper. (Backward tensor hooks — the previous
mechanism — never fire under FSDP2: autograd flows through `fully_shard`'s
unsharded proxies and grads land in `p.grad` directly.) Under FSDP2 each
rank holds the reduce-scattered shard, so norms are recovered with one
batched all-reduce as `sqrt(Σ_ranks ‖g_shard‖²)`; under DDP / single GPU the
full local gradient is used as-is. Tracked: every audio table (`emb_{k}`),
`text_emb`, `ref_compressor` (aggregate), `null_prefix`. This family exists because of
the gradient-starvation episode (§3.2): audio-interface gradients were
~1000× smaller than backbone gradients and nothing else made that visible.
Healthy: audio/text grad norms within an order of magnitude of each other.

**Embedding health (`norm/*` cheap; `cos_to_init`, `drift`,
`effective_rank` expensive).** The question these answer: *is the pretrained
text table adapting to the TTS domain or being destroyed?* A rising
`norm/text_emb` alone cannot distinguish the two. So at `on_train_begin` a
deterministic 8192-row sample of `embed_tokens` (seed 0, fp32, CPU, rank 0;
full weights materialized via `summon_full_params` on FSDP1 or DTensor
`full_tensor()` on FSDP2) is snapshotted, and every expensive step logs:

| Metric | Meaning | Healthy | Alarming → action |
|---|---|---|---|
| `cos_to_init/text_emb` | row-wise cosine to init (rotation) | smooth decline, plateau > 0.85 | < 0.75 in the first 10–20k steps → cut embed LR (the ×0.2 group exists because of this) |
| `drift/text_emb` | relative Frobenius drift | growth that decelerates | linear growth, no plateau → same |
| `effective_rank/text_emb` | entropy of singular values | stable, in the thousands | drop > 20% = subspace collapse → consider freezing |
| `effective_rank/emb_{k}` | audio-table specialization | **falling** = the table is specializing | flat near init = the table is not learning (starvation) |

Note the sign flip between the last two rows: for the pretrained text table a
falling rank is damage; for the fresh audio tables a falling rank is the
whole point.

**Cross-modal alignment (`cosine_sim/layer_{i}/text_audio`, expensive).**
Mean-pooled text-region vs audio-region hidden states per backbone layer,
collected via `output_hidden_states` on exactly one forward per expensive
interval (the `_collect_layer_sims` flag is armed one log-step early so the
*next* forward captures it; ~28 extra `[B,T,d]` tensors is why it is gated).
This metric produced a genuine finding: layers 12–13 reach cosine 0.98–0.99
by step ~2000 — the stock causal attention itself performs the cross-modal
alignment, no dedicated aligner needed (and the weight-space
`cosine_sim/text_audio`, which compares embedding *tables*, was shown to be
irrelevant for alignment questions — it is still logged, but read it as an
embedding-geometry stat, not an alignment one). A transient dip in middle
layers around steps 1–2k is normal (representation reshuffle), recovery is
what matters.

**Voice cloning (`vc/*`, cheap).** All measured on the **pre-CFG** compressor
output (post-CFG would mix in `null_prefix` rows and oscillate with the
per-step drop rate): `query_cos_sim_mean/max` and `query_std_across_k` —
query-collapse detection, the observable the diversity loss acts on;
`null_cos_sim_mean` — whether the learnable null is itself collapsing;
`norm/prefix_cond` vs `norm/prefix_null` — prefix scale sanity next to
text/audio norms (§11.2); `cfg_dropout_rate_observed` vs `forced_null_rate`
— the coin flip vs the data-driven sentinel share, separated so a drift in
corpus composition (§9.4) is not mistaken for RNG behavior; plus the SupCon
diagnostics (positive-pair counts, similarity gaps) under `vc/supcon_*`.

**Reading it as a dashboard**: `loss/level_*` uniform and falling +
`grad_norm/emb_*` within 10× of `grad_norm/text_emb` + `cos_to_init` on its
plateau + deep-layer `cosine_sim` climbing = a healthy run. Any one of those
diverging has a specific, documented lever (LR group multipliers, freezing,
curricula) rather than a generic "lower the LR".

### 12.8. Running Experiments

*   **One-off sweeps** are CLI overrides — composition + validation catch
    inconsistent combos before any GPU time is spent
    (`python -m gepard.cli.train trainer.learning_rate=3e-4 voice_cloning=disabled`).
*   **Phases are entry files, not experiments.** Pretrain is `conf/train.yaml`
    (`make train`), the LoRA fine-tune is `conf/sft.yaml` (`make finetune` →
    `gepard.cli.sft`), DPO is `conf/dpo.yaml`. Don't reach for an experiment
    preset to switch phase.
*   **A new experiment** is for a one-off *variation* layered on an entry, not a
    config copy. The repo ships two worked examples, one per style of override:

    *   `conf/experiment/ft_wider_lora.yaml` — a **fine-tune** ablation on
        `sft.yaml` using **point overrides**: wider LoRA `rank 32`, MLP
        projections added to the adapted modules, gentler `lr 1e-4` / `2 epochs`,
        distinct wandb name.

        ```yaml
        # @package _global_
        finetune:
          lora:
            rank: 32
            alpha: 64
            target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
        trainer:
          learning_rate: 1.0e-4
          num_train_epochs: 2
          wandb: {name: SFT-WIDER-LORA-1}
        ```

    *   `conf/experiment/pretrain_no_vc.yaml` — a **pretrain** ablation on
        `train.yaml` that also **swaps a whole group** (voice cloning off) via a
        `defaults` override, alongside point overrides:

        ```yaml
        # @package _global_
        defaults:
          - override /voice_cloning: disabled     # swap the whole group…
        trainer:                                   # …and tweak fields too
          learning_rate: 6.0e-4
          num_train_epochs: 3
          wandb: {name: PRETRAIN-NO-VC-1}
        ```

    Nothing loads a preset unless asked. Run one on top of its entry either way
    (`EXP` forwards the override through both `make train` and `make finetune`):

    ```bash
    make finetune EXP="experiment=ft_wider_lora"          # via the Makefile
    make train    EXP="experiment=pretrain_no_vc"
    accelerate launch --config_file accelerate/finetune_single.yaml \
        -m gepard.cli.sft experiment=ft_wider_lora        # or directly
    ```

    Copy either to `conf/experiment/<name>.yaml` for your own. The `# @package
    _global_` header is required — without it the keys nest under `experiment.*`
    instead of overriding the real groups. Everything the preset does not name
    stays exactly as the entry composed it (checkpoint, freezing, dataset, …).
*   **Guardrails**: the golden config pins in `tests/test_config_hydra.py`
    fail loudly if a *default* drifts (edit them intentionally when the
    default should change); the golden-loss tests (`make test-baseline`) catch
    numerical regressions in the loss stack; `make test-baseline-real` does
    the same against the real backbone on GPU.

---

## 13. Inference

The in-repo inference stack is the **reference implementation**: it is what the
DPO data stages run on, what the demo notebook uses, and the semantic ground
truth production serving must match. Production itself is a vLLM deployment of
the single-pass model (§1); nothing in this chapter is required at serving
time except the checkpoint files (§10.8).

### 13.1. The Classes

Everything lives in `gepard/inference/`:

| Class | File | Role |
|---|---|---|
| `GepardRunner` | `runner.py` | **The canonical runner.** `TTSRunner` + optional text-CFG; with the default `cfg_scale=1.0` it is exactly the single-pass generator. Use this one. |
| `TTSRunner` | `runner.py` | The plain single-pass base class. Still consumed directly by the DPO data stages (`gepard/data/dpo/*`), which drive the forward helpers themselves for batched group rollouts. |
| `FullAttnCache` | `runner.py` | KV-cache shim: the stock `Qwen3_5DynamicCache` constructor crashes on a model whose `layer_types` contain no `linear_attention` entries — which is precisely our full-attention-only backbone. The shim re-implements init for the all-full-attention case. |
| `UnfoldedCodecModel` | `codec_wrapper.py` | NeMo `AudioCodecModel` subclass that (a) strips the training-only SLM discriminator from the config before init (skips a ~360 MB WavLM download), and (b) adds `decode_from_codes` — direct waveform decode from the 32 unfolded per-dimension codes, bypassing mixed-radix recomposition. Needs the NeMo stack → lives in `venv_dpo`/`venv_infer`, never in the training env. |

### 13.2. Loading

`GepardRunner.from_checkpoint(path_or_repo)` — one call, self-describing
(§10.9): `gepard_config.json` → `build_model` → safetensors → tokenizer, with
`attn_implementation="eager"` as the inference default (FlashAttention-2 is a
training optimization; eager keeps the runner CPU-capable and
dependency-light) and device auto-detect. Legacy checkpoints need
`fallback=<composed config>`; everything exported by the current trainer
loads standalone.

### 13.3. Generation Anatomy

`generate(text, ref_codes=None, …)` is a hand-rolled AR loop over the
**backbone only** (the overlay is applied manually each step):

1.  **Text layout** — tokenizer + the same deterministic `TextRepeater`
    policy as training (§9.6), from the checkpoint's stamped `text_repetition`.
2.  **Prefix** — one `RefCompressor` call when `ref_codes` are given (§11.6);
    omitted entirely otherwise.
3.  **Prefill** — `[prefix? | text]` through the backbone with
    `use_cache=True` into a `FullAttnCache`; the hidden state at the SOS
    position seeds the first frame.
4.  **Frame loop** — embed the previous frame through the audio-embedding
    stack (§10.3) → one cached decode step → stop decision
    (`sigmoid(stop_head) > stop_threshold`) → sample all 32 heads
    independently in fp32 (temperature → top-k → multinomial, with optional
    repetition penalty over a sliding window of recent frames).
5.  Output: `(num_heads, T)` long tensor — `unsqueeze(0)` and feed straight
    to `UnfoldedCodecModel.decode_from_codes`.

Knobs (defaults): `temperature=1.0` (0.4 is the production/DPO operating
point), `top_k=0` (off), `stop_threshold=0.5`, `max_frames=2000` (hard
ceiling), `repetition_penalty=1.0` + `repetition_window=32`, and
`force_stop_frames` — a deterministic guardrail that truncates regardless of
the stop head (the DPO sampler's adaptive cap, §9.12, is this knob computed
per text).

### 13.4. CFG Is Optional — and Off by Default

`GepardRunner.generate` adds three arguments; **with `cfg_scale=1.0` (the
default) none of the CFG machinery runs** — no second prefill, no extra
forward per frame, bit-identical to the base single-pass path. Turning it on:

*   `cfg_scale > 1.0` — a second, text-free branch is prefilled (same speaker
    prefix, §11.6) and per-head logits are guided:
    `logit = logit_uncond + cfg_scale · (logit_cond − logit_uncond)` before
    temperature/top-k. Typical 2.0–3.0.
*   `cfg_frames = N` — onset-only guidance: after frame N the uncond branch
    is dropped entirely (the derailment CFG fixes is born in the first frames,
    §5); `None` guides every frame.
*   `cfg_uncond_mode` — how the text-free branch is built: `"empty_text"`
    (`[SOT EOT SOS]`, cleanest contrast, default) or `"audio_only"` (`[SOS]`,
    more aggressive, more OOD).

One trap: `cfg_frames=0` with `cfg_scale>1` disables guidance but still pays
the uncond prefill — the off switch is `cfg_scale=1.0`.

When to reach for CFG: short prompts on a checkpoint that has NOT been
through DPO (CFG at scale 3 is the crutch DPO later distills away, §7.4).
A post-DPO checkpoint is designed to run single-pass — CFG remains available
as a quality lever for pathological inputs, but the production configuration
is `cfg_scale=1.0`, which is also the only vLLM-compatible mode (two-pass
guidance does not fit continuous batching, §1.2).

### 13.5. The Notebook Flow

`notebooks/inference_demo.ipynb` is the end-to-end template (Colab-ready:
HF auth → clone → `nemo-toolkit[tts]` → `pip install -e .[inference]`):

```python
player = UnfoldedCodecModel.from_pretrained("nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps").to(device).eval()
model  = GepardRunner.from_checkpoint("nineninesix/gepard-1.0")

# reference voice → unfolded codes [1, T_ref, 32]
wave, _ = librosa.load("assets/ref_audio/audio_en.wav", sr=22050)
tokens, _ = player.encode(audio=wave_tensor, audio_len=wave_len)
ref_codes = unfold_tokens(tokens.cpu(), num_levels=[8, 7, 6, 6]).permute(0, 2, 1).to(device)

out = model.generate(text, ref_codes=ref_codes, temperature=0.4)   # single-pass
audio, _ = player.decode_from_codes(out.unsqueeze(0), enc_len)     # 22.05 kHz waveform
```

The `ref_codes` shape contract is the same one the training loader produces
(§9.9): `[1, T_ref, 32]` unfolded int64 codes. `unfold_tokens` +
`dequantize_codes` live in `gepard/model/codec_ops.py` and are shared by
training, the compressor, and this path — one mixed-radix implementation
everywhere.

---

## 14. Environment and Operations

This project was operated on AWS GPU instances (DLAMI-based boxes with
ephemeral NVMe); the environment tooling reflects that reality — ephemeral
fast storage, re-runnable provisioning, and explicit dataset transfer scripts.

### 14.1. Three Virtualenvs

The NeMo codec stack and the training stack cannot share an environment
(NeMo pins an old `transformers` and drags its own torch constraints), so the
repo maintains three:

| Env | Setup | Stack | Consumers |
|---|---|---|---|
| `venv` | `make setup` | torch 2.8.0 (cu128) + transformers 5.3.0 + accelerate + datasets + wandb + optional flash-attn | `make dataset / train / finetune / dpo-pairs / dpo-train`, tests |
| `venv_dpo` | `make setup_dpo` | + `nemo-toolkit[tts]` codec + Whisper + jiwer (no accelerate/wandb needed) | `make dpo-sample / dpo-score` |
| `venv_infer` | `make setup_inference` | codec + runner only — no datasets/whisper/train deps | demos, the notebook, smoke tests |

Import hygiene is enforced by tests (`tests/test_import_hygiene.py`): the
inference path must not import `datasets`/`wandb`, so a lean deploy stays lean.

### 14.2. Setup Logic (why bash, not a lockfile)

Provisioning is deliberately imperative (`scripts/setup*.sh` on
`scripts/lib/env_common.sh`, all via [`uv`](https://github.com/astral-sh/uv),
auto-installed on first run): the hard parts are runtime decisions no static
lockfile can express —

*   **CUDA-matched torch**: `cuda_wheel_tag()` reads the driver
    (`nvidia-smi`) and maps it to a wheel tag with a deliberate **ceiling of
    cu128** — it runs on any driver ≥ 12.8 including CUDA 13.x/Blackwell, and
    distributed training only ever happens in `venv` (torch 2.8.0+cu128), so
    nothing chases newer tags. Torch installs **index-only** from the PyTorch
    index (adding PyPI as extra-index is what breaks it — uv would mix a
    generic CPU torch with cu128 companions).
*   **The NeMo → transformers re-pin**: `nemo-toolkit[tts]==2.4.0` downgrades
    transformers; `uv pip install -e .[dpo|inference]` runs LAST and
    re-asserts the pyproject pin (5.3.0). Order is the contract.
*   **torchcodec ABI trial**: torchcodec ships per-torch ABI builds and NeMo
    only floors it; `fix_torchcodec()` walks versions newest→older until the
    shared library actually imports.
*   **flash-attn is optional and non-fatal**: built from source with
    `--no-build-isolation` against the installed torch (needs `nvcc` from
    `make system-deps` and explicit `wheel setuptools ninja` in the minimal
    uv venv); on failure the trainer falls back to SDPA.
*   **Self-heal**: after all installs, `verify_cuda_selfheal` re-checks
    `torch.cuda.is_available()` and reinstalls the CUDA-matched torch if a
    dependency silently replaced it.

`make system-deps` (sudo, once per machine) covers the apt layer:
`nvidia-cuda-toolkit` (nvcc), python3.12 dev headers, git-lfs, curl.
Idempotent — checks before touching anything.

The declarative *safe* dependencies (pure-Python pins incl.
`transformers==5.3.0`, hydra, per-phase extras `[train]/[dpo]/[inference]`)
live in `pyproject.toml`; the scripts end with the editable install so those
pins always win.

### 14.3. Ephemeral NVMe (the AWS Case)

The `/opt/dlami/nvme` paths throughout the configs are not a convention we
invented — that is where AWS Deep Learning AMI mounts the **instance-store
NVMe**: locally attached SSD that is fast (multi-GB/s), large, free — and
**wiped whenever the instance stops**. The operating discipline:

*   **On NVMe (rebuildable, hot)**: the HF download cache
    (`make dataset NVME=/opt/dlami/nvme` sets `HF_HOME` there), the prepared
    Arrow dataset (`data.train_dataset_path`), training checkpoints
    (`trainer.output_dir`) and the keep-index. Everything the training loop
    reads or writes at high rate.
*   **Off the box (durable)**: source datasets on the HF Hub or S3, dataset
    snapshots synced to S3 (§14.4), model checkpoints pushed to the HF Hub
    (`make upload`). Anything you cannot regenerate from these is a bug in
    your workflow — assume the NVMe contents disappear.

On a non-AWS host simply override the two path fields
(`data.train_dataset_path`, `trainer.output_dir`) or point `NVME=` anywhere
fast; nothing else assumes the DLAMI layout.

### 14.4. Dataset and Checkpoint Transfer

*   **`scripts/download_aws_dataset.sh` / `scripts/upload_aws_dataset.sh`** —
    interactive S3 sync helpers for moving prepared Arrow datasets between a
    durable bucket and the ephemeral NVMe. Both self-install AWS CLI v2 if
    missing and walk through auth (manual keys / existing profile /
    `--no-sign-request` for public buckets on download). The uploader
    auto-discovers HF datasets (directories carrying `dataset_info.json`)
    next to the script, shows sizes, and supports upload-all; transfers are
    `aws s3 sync`, so re-runs only move deltas. Typical cycle on a fresh box:
    download the prepared dataset from S3 instead of re-running `make dataset`
    over the source corpora.
*   **`scripts/upload_to_hf.py`** (`make upload REPO=… CHECKPOINT=…`) —
    pushes a checkpoint directory to the HF Hub. Validates the servable file
    set first (`config.json` + `model.safetensors` — i.e. a final/merged
    export, §10.8), backfills tokenizer files from the base model when the
    checkpoint lacks them, and creates the repo (private by default).
    Two publish modes:
    *   **Default — inference artifact.** Training resume state (optimizer,
        scheduler, RNG, FSDP shards) is skipped, and float weights are cast to
        `gepard_config.json`'s `model_dtype` (bf16) in a temporary copy —
        FSDP2 training keeps fp32 master weights, which would otherwise double
        the download for zero inference benefit. Local files are never
        modified. Override with `--dtype keep|bf16|fp16|fp32`.
    *   **`TRAINING_STATE=1` — resume snapshot.** Optimizer/scheduler/RNG
        state is included and the weights are uploaded as-is (fp32 master);
        any dtype argument is ignored with a warning, because AdamW moments
        belong to the fp32 weights they were computed against.
    Safety guards in both modes: a `model.safetensors` still carrying raw
    un-merged LoRA adapters is refused with a pointer to `make merge`
    (override: `--allow-unmerged`).

### 14.5. Makefile

The Makefile is intentionally a thin, colorful command directory — every
target is one `source venv/bin/activate && …` line away from the underlying
command, and all real behavior lives in the configs and scripts documented
above. The complete target → command map is §12.1; environment setup targets
are §14.1–14.2; `make help` prints the same inventory in the terminal. The
only Makefile-level conveniences to know: `NVME=` (HF cache redirect, §14.3),
`DPO="…"` (Hydra overrides pass-through), `SHARDS=` (parallel DPO sampling),
`CHECKPOINT=`/`REPO=` arguments for resume/merge/upload targets, and
`TRAINING_STATE=1` on `make upload` (resume snapshot: optimizer state
included, weights stay fp32 — see §14.4).
