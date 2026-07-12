# Fine-Tuning Gepard on a New Language

How to teach a trained Gepard checkpoint a language it has never seen, **without
catastrophically forgetting the languages it already speaks**. Arabic is used
throughout as the worked example, but the method is language-agnostic — only
§2 and §7 are script-specific.

This guide assumes you have read [MODEL_GUIDE.md](MODEL_GUIDE.md) §5 (short
register), §9 (data), §11 (voice cloning) and §12 (training). Section numbers
below refer to it.

Throughout, **resident languages** means the languages your base checkpoint was
already trained on, and **target language** the new one you are adding.

## Contents

- [1. The short version](#1-the-short-version)
- [2. Step 0 — Check the tokenizer before anything else](#2-step-0--check-the-tokenizer-before-anything-else)
- [3. What actually has to be learned (and what must not move)](#3-what-actually-has-to-be-learned-and-what-must-not-move)
- [4. Step 1 — Build a mixed corpus](#4-step-1--build-a-mixed-corpus)
- [5. Step 2 — The fine-tune](#5-step-2--the-fine-tune)
- [6. Step 3 — Watching for forgetting](#6-step-3--watching-for-forgetting)
- [7. Step 4 — Recalibrate the short-register thresholds](#7-step-4--recalibrate-the-short-register-thresholds)
- [8. Step 5 — Do you need a DPO round?](#8-step-5--do-you-need-a-dpo-round)
- [9. Arabic-specific notes](#9-arabic-specific-notes)
- [10. The language tag — optional, and usually skippable](#10-the-language-tag--optional-and-usually-skippable)
- [11. Checklist](#11-checklist)

---

## 1. The short version

**Do:**

- Reuse the existing SFT phase (`conf/sft.yaml`, `make finetune`) as-is — LoRA,
  frozen base.
- Train on a **mixed corpus**: target language **plus a replay share of the
  resident languages**. This is the single most important anti-forgetting lever.
- Raise the adapter capacity (rank + MLP targets) — a new language needs more
  than the short-phrase fix did.
- Re-measure the short-register thresholds (§5.3) on the target language instead
  of assuming the shipped 13/16 transfer.

**Do not:**

- Extend the tokenizer or resize the embedding table — for most languages you do
  not need to (§2), and this is the classic road to catastrophic forgetting.
- Run a full fine-tune (`finetune.lora.enabled: false` +
  `freeze_backbone: false`). That puts `embed_tokens` and the whole backbone on a
  live learning rate and rotates the pretrained text geometry away. The
  `cos_to_init/text_emb` telemetry (§12.7) exists precisely because this failure
  mode was observed.
- Plan a DPO round up front. Measure first (§8).

---

## 2. Step 0 — Check the tokenizer before anything else

Everything downstream depends on one question: **does the backbone's tokenizer
already represent the target language as real subwords, or does it fall back to
raw bytes?** Answer it before you touch a single config.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("nineninesix/qwen3_5-full-attn-only-14")
for text in ["Hello there", "صباح الخير", "كيف حالك اليوم؟"]:
    ids = tok.encode(text, add_special_tokens=False)
    print(f"{len(ids):3d} tok | {len(ids)/len(text.split()):4.1f} tok/word | "
          f"{[tok.decode([i]) for i in ids]}")
```

For Arabic on the shipped tokenizer (vocab 248,077) this prints:

```
  2 tok |  1.0 tok/word | ['Hello', ' there']
  3 tok |  1.5 tok/word | ['ص', 'باح', ' الخير']
  5 tok |  1.7 tok/word | ['كيف', ' ح', 'الك', ' اليوم', '؟']
```

Those are genuine Arabic subwords, not `<0xD9>`-style byte fragments. Two
consequences, and they define the whole recipe:

1. **No vocabulary surgery.** The Arabic token ids already exist and already
   carry Qwen3.5's pretrained embeddings. They were never *damaged* by Gepard's
   pretrain — a token that never appears in the text gets no gradient — they were
   simply never wired to acoustics. There is nothing to add, resize, or
   re-initialize. Skipping the resize also skips the largest single source of
   forgetting.
2. **Token density differs from the resident languages.** Arabic runs ~1.5–3.0
   tokens per word against ~1.0–1.3 for English. Everything in the codebase that
   is threshold-on-token-count — the repetition trigger, the keep-index — was
   calibrated at English density and needs a second look (§7).

**If your target language *does* fall back to bytes** (a very low-resource
script), this guide does not cover you: you would need vocabulary extension and
embedding initialization, which reopens the forgetting problem in its hard form.
Check first.

---

## 3. What actually has to be learned (and what must not move)

Gepard's stack splits cleanly, and the split is what makes new-language
fine-tuning tractable:

| Component | Language-specific? | What you should do |
|---|---|---|
| Codec / FSQ codes, `audio_embeddings`, `audio_embed_proj`, `codebook_heads` | **No.** These are neural-codec codes — the codec does not know or care what language the waveform is in. | Leave frozen. No new capacity is needed here. |
| `model.embed_tokens` (text table) | Already covers the target language (§2). | **Leave frozen.** It is not missing anything. |
| Backbone attention/MLP — the **text→audio routing** | **Yes.** This is the entire job: mapping target-language graphemes onto acoustic frames. | This is what LoRA adapts. |
| `ref_compressor`, `null_prefix` (speaker prefix) | Mostly no — timbre, not phonetics. | Leave frozen (`qformer_frozen`). |
| `stop_head` | Indirectly — its calibration rides on the hidden states the adapters change. | Frozen during SFT. Only DPO can train it (§8). |

So **forgetting in Gepard lives in exactly one place**: the backbone's text→audio
routing. And the SFT phase already frozen-by-construction protects everything
else — [`inject_backbone_lora`](../gepard/training/base.py) calls
`model.requires_grad_(False)` first, then injects adapters. During training the
base weights *literally do not move*. Forgetting can only enter through the
function shift of the merged adapter — which is bounded by what you train it on.

Hence: **the corpus mix is the anti-forgetting mechanism**, not any regularizer.

---

## 4. Step 1 — Build a mixed corpus

### 4.1. Tokenize the audio

Every Gepard source is a HF dataset of **pre-tokenized codec codes** (§9.1) — the
prepare step never reads audio. Run your target-language corpus through the
open-source pipeline:

> https://github.com/nineninesix-ai/nano-codec-processing-pipeline

Output must carry `nano_layer_1..8` (packed FSQ indices), `encoded_len` (frame
count), the transcript, and — ideally — a speaker column.

### 4.2. Declare both the new language and the replay data

The replay share is not a training-time knob; there is **no online source
weighting** anywhere in the loop. Sources listed in `hf_datasets` are simply
concatenated at build time (§9.3), so the mix ratio is decided by **row counts**.
Both languages go in the *same* list, in `conf/data/full_corpus.yaml`:

```yaml
train_dataset_path: "/opt/dlami/nvme/train_dataset_ar"   # a NEW output dir
max_duration_sec: 40
add_speaker_id: true
singleton_policy: null_prefix
min_clips_per_speaker: 3

hf_datasets:

  # ── target language ────────────────────────────────────────────────────────
  - reponame: your-org/arabic_nano_codec_21_dataset
    local: false
    split: train
    text_col_name: text
    speaker_id_col_name: speaker
    nano_layer_1: nano_layer_1
    nano_layer_2: nano_layer_2
    nano_layer_3: nano_layer_3
    nano_layer_4: nano_layer_4
    nano_layer_5: nano_layer_5
    nano_layer_6: nano_layer_6
    nano_layer_7: nano_layer_7
    nano_layer_8: nano_layer_8
    encoded_len: encoded_len
    speaker_prefix: "arab"       # PIN it — see below
    # language_tag: ar           # optional, see §10 — leave off for Arabic

  # ── replay: the resident languages, same sources the base was trained on ───
  - reponame: nineninesix/emolia_filtered_nano_codec_21_dataset
    local: false
    split: train
    text_col_name: text
    speaker_id_col_name: speaker
    nano_layer_1: nano_layer_1
    # ... 2..8 ...
    encoded_len: encoded_len
    speaker_prefix: "emol"
    max_len: 120000              # ← the replay-ratio knob
```

Two keys deserve attention:

- **`speaker_prefix`** — pin it. Without it the 4-char cross-source namespace is
  generated from `uuid4()` at every run, so speaker ids are not stable across
  rebuilds (§9.4). Anything keyed on speaker id (eval sets, cached indices,
  notebooks) breaks silently.
- **`max_len`** — the row cap applied *after* processing (shuffled with a fixed
  seed, then truncated). This is how you set the replay ratio: cap the resident
  source at roughly the row count of the target source.

**What ratio?** There is no validated number in this repo — treat any figure you
read as a starting point, not a result. Start at roughly **50/50** and let the
two evaluation numbers (§6) move it: target-language WER still too high → shift
toward the target language; resident-language WER regressing → shift toward
replay. Report what you land on.

**Do *not* retag or re-derive the replay data.** It must arrive at the model in
exactly the distribution the base checkpoint was trained on — that is what makes
it replay. (This is also the main reason to leave the language tag off; §10.)

### 4.3. Build with repetition ON

The SFT phase requires the corpus to be **built** with text repetition baked in —
it is a prepare-time decision (§9.6), not a training flag:

```bash
source venv/bin/activate
python -m gepard.cli.prepare text_layout=repeat_ft
```

> **Note:** use the direct command, not `make dataset text_layout=repeat_ft`.
> The `dataset` Makefile target does not forward Hydra overrides, so make would
> swallow `text_layout=repeat_ft` as a make variable and you would silently build
> a pretrain-layout (repetition-off) corpus.

### 4.4. Sanity-check the build

With `data.speaker_statistics: true` the build writes `speaker_statistics.json`
into the output dir. Read `pct_structural_null_exposure` (§9.4) — the share of
rows that take the unconditional (`null_prefix`) path from the data side alone.
New-language corpora are often speaker-poor (many speakers with 1–2 clips), and
under `singleton_policy: null_prefix` those all route to the sentinel. If this
number drifts far from what the base corpus had, the model's
conditional/unconditional balance shifts with it, and voice cloning degrades for
reasons that have nothing to do with language.

### 4.5. Keep-index (optional)

The fine-tune keep-index (§9.7) reweights the corpus toward short rows and drops
the long tail. If you use it, rebuild it against the **new** corpus, and read §7
first — its `--short-t` is a token-count threshold and inherits the same
calibration problem:

```bash
python -m gepard.data.preprocessing.build_keep_index \
    --dataset /opt/dlami/nvme/train_dataset_ar \
    --short-t 13 --target-fraction 0.5 --tail-cap 150 \
    --out-dir /opt/dlami/nvme/train_dataset_ar
```

Then point `trainer.keep_index_path` at the resulting `keep_idx_*.npy`.

---

## 5. Step 2 — The fine-tune

`conf/sft.yaml` is already the right phase: LoRA on a frozen base, voice cloning
present but compressor frozen, repetition on. Two things to change.

### 5.1. Point it at the new corpus

```yaml
# conf/data/full_corpus.yaml (or a copy)
train_dataset_path: "/opt/dlami/nvme/train_dataset_ar"
```

And in `conf/finetune/sft_lora.yaml`, `checkpoint_path` is the base you are
adapting — the shipped `nineninesix/gepard-1.0`, or your own pretrain.

### 5.2. Give the adapters enough capacity

The shipped SFT preset (r=16, `q/k/v/o` only) was sized for the short-phrase fix
— a behavioral nudge. A whole new language is a bigger ask. Note that
`last_n_layers: 16` against a 14-layer backbone already resolves to *all* layers
([`inject_lora`](../gepard/model/lora.py) clamps `first = max(0, 14-16) = 0`), so
the remaining capacity knobs are **rank** and **target modules**.

Create `conf/experiment/ft_arabic.yaml`:

```yaml
# @package _global_
finetune:
  lora:
    rank: 32
    alpha: 64
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
trainer:
  learning_rate: 1.0e-4        # wider adapters → gentler than the 2e-4 default
  num_train_epochs: 3
  wandb: {name: SFT-AR-1}
```

Run it:

```bash
make finetune EXP="experiment=ft_arabic"
```

Everything the preset does not name stays as `sft.yaml` composed it — frozen
compressor, frozen heads, frozen embeddings, repetition layout, checkpoint.

### 5.3. The thing not to do

Do not reach for a full fine-tune to "give it more room". Setting
`finetune.lora.enabled: false` with `freeze_backbone: false` puts `embed_tokens`
and every backbone weight on a live LR. That is the one configuration in this
repo that can actually destroy the resident languages — the pretrained text table
re-rotates away from Qwen3.5 semantics, and the `embed` LR group (×0.2) and the
`cos_to_init/text_emb` metric exist because of exactly this (§12.3, §12.7).

If LoRA at r=32 with MLP targets genuinely cannot fit the language, the honest
next step is more data and a longer run — not unfreezing the base.

---

## 6. Step 3 — Watching for forgetting

**The most important thing to understand here: under LoRA, the embedding-health
telemetry cannot see forgetting.** `cos_to_init/text_emb`, `drift/text_emb`,
`effective_rank/text_emb` (§12.7) all measure movement of `embed_tokens` — which
is frozen. They will be flat by construction. Flat is *not* evidence of safety;
it is evidence that you froze the thing they watch.

Forgetting under LoRA is a **behavioral** property of the merged function, so it
must be measured behaviorally:

1. **Hold out a resident-language eval set** before you start. Transcribe
   generations with an ASR model and record WER *and* speaker similarity on the
   base checkpoint — that is your baseline. Re-measure after the fine-tune. This
   is the only real forgetting detector you have.
2. **Hold out a target-language eval set** the same way. The pair of numbers is
   what moves the mix ratio (§4.2).
3. `loss/level_*` (per-head CE, §12.7) — all 32 curves should fall together. One
   flat channel among 31 is a data/unfold bug in the new corpus, not a language
   problem.
4. `vc/*` — if target-language speakers are out of distribution for the frozen
   compressor, `query_cos_sim_*` and `norm/prefix_cond` will say so. See §9 below.

The DPO stage-2 scorer (`gepard/data/dpo/score.py`) is a ready-made ASR + WER +
duration harness and can be reused for both eval sets — mind the Whisper model
choice (§8).

---

## 7. Step 4 — Recalibrate the short-register thresholds

The short-register fix (§5) is threshold-driven, and the thresholds are in
**text tokens**:

- `text_layout.apply_below: 13` — repeat only prompts shorter than this;
- `text_layout.target_text_tokens: 16` — repeat up to about this many tokens;
- keep-index `--short-t 13` — what counts as a "short" row.

All three come from the failure-rate sweep in §5.3, run at **English token
density**. Arabic packs ~1.7 tokens per word against English's ~1.2, so an Arabic
utterance crosses 13 tokens at roughly 7–8 words where English needs ~10. Concretely:
acoustically short Arabic prompts can land *above* `apply_below` and receive no
repetition at all, while still sitting in the fragile register the repetition was
invented for.

Do not port the numbers. **Re-run the sweep** on the fine-tuned checkpoint with
target-language prompts: generate across a range of prompt token budgets, count
runaway/derailed generations, and find where the failure rate plateaus. That plateau
is your `apply_below`; the target budget follows from it. The probe from §5.1
(belief entropy flat and elevated across the generation = derailed; dropping within
the first ~50 frames = clean) is the cheapest classifier for this.

Because repetition is baked in at prepare time (§9.6), a new threshold means
**rebuilding the corpus** and re-running the fine-tune. Budget for one calibration
cycle.

---

## 8. Step 5 — Do you need a DPO round?

**Probably not, and you should not plan for one up front.** Two facts:

- The default SFT base, `nineninesix/gepard-1.0`, is **already the output of two
  DPO rounds** (§12.5). Its stop head has been trained against runaway and
  premature stops.
- SFT does not damage it directly: `stop_head` is frozen throughout the LoRA
  fine-tune. But the adapters change the hidden states feeding it, so its
  *calibration* can drift on target-language inputs. May, not will.

So measure, then escalate. The ladder:

1. **Measure.** After the fine-tune, run the §7 sweep on target-language prompts.
   Failure rate on short prompts below ~5% (the plateau band of §5.3) → you are
   done. No DPO.
2. **If short prompts derail — try inference-time CFG first.** It costs no
   training at all: `GepardRunner.generate(..., cfg_scale=2.5, cfg_frames=20)`
   ([runner.py](../gepard/inference/runner.py)). Onset-only guidance, so the
   second pass runs for the first 20 frames and single-pass decoding resumes after
   — this is the exact mechanism DPO later distills into the weights (§7.4).
3. **DPO only if CFG is insufficient, or you cannot afford a two-pass prefill in
   serving.** But then it is the *only* option: **DPO is the one place in this
   repo where `stop_head` is unfrozen** (§12.5, its own LR of 1e-4). SFT
   structurally cannot reach it.

### If you do run DPO, four things need target-language recalibration

The DPO configs are English-calibrated, and two of the defaults fail *silently*
on another language:

| Config | Default | Why it breaks |
|---|---|---|
| `dpo/reward.whisper_model` | `distil-whisper/distil-large-v3` | **English-only model.** Point it at a multilingual Whisper (e.g. `openai/whisper-large-v3`). Otherwise the WER term is noise — and since empty ASR caps WER at 1.0, the model reward-hacks into silence via `w_empty`. |
| `dpo/reward.whisper_language` | `en` | Set to the target language (`ar`). |
| `dpo/reward.sec_base` / `sec_per_word` | `0.7` / `0.4` | The expected-duration model is linear in **word count**, fitted on English. Arabic is morphologically denser (clitics written joined), so words-per-second differs and the mandatory two-sided length penalty (§7.3) starts lying in both directions. Refit on your own corpus: regress `encoded_len / codec.frame_rate_hz` on word count. |
| `dpo/sampling.texts_file` + `ref_audios` | English short prompts, mixed refs | `assets/dpo_seed/short_focused_v2.jsonl` is 2.5k English prompts. You need target-language seed prompts in the same short register, and reference voices in the target language. |

DPO training is itself LoRA-based (`q,v` targets) plus the stop head, so it is
just as non-destructive to the base weights as SFT.

---

## 9. Arabic-specific notes

Not about Gepard, but they will dominate your quality if ignored:

- **Undiacritized orthography.** Written Arabic normally omits short vowels, so
  the grapheme→phoneme mapping is genuinely ambiguous and the model must infer
  vowels from context. This is the central difficulty of Arabic TTS. Either supply
  **diacritized transcripts** (best), or accept that you need substantially more
  data for the model to learn the disambiguation implicitly. Whichever you choose,
  do it consistently — a corpus half-diacritized is worse than either extreme, and
  inference must match training.
- **Text normalization.** Numerals, dates, Latin-script insertions and
  abbreviations need to be verbalized in the transcript, and the *same*
  normalization must run at inference. Gepard has no text frontend — what you
  tokenize is what it speaks.
- **Dialect.** MSA and the spoken dialects are effectively different registers.
  Mixing them untagged in one corpus teaches an average of both. If you need
  control over which one comes out, that is the one case where the language tag
  earns its keep — see §10.
- **Voice cloning across languages** is an open question in this codebase (§11.5,
  listed under future work: "cross-lingual reference robustness"). The compressor
  is frozen at SFT, which protects the existing speaker space, but if Arabic
  timbres are out of distribution for it, cloning quality can drop. Watch `vc/*`
  and evaluate cloning separately from intelligibility.

---

## 10. The language tag — optional, and usually skippable

The prepare pipeline **supports** language tagging, and it is worth knowing about
even though this guide's Arabic recipe does not use it.

**What it does.** Set `language_tag: ar` on an `hf_datasets` item and every
transcript from that source is prefixed as `"ar: {text}"` *before tokenization*
([processor.py:331](../gepard/data/preprocessing/processor.py#L331)). The tag
becomes ordinary text tokens inside the `[SOT … EOT]` frame — no tokenizer change,
no special ids. It is a pure text-level routing convention.

**Why Arabic does not need it.** A tag is a disambiguator: it exists for cases
where *the tokens alone cannot tell the model which language this is*. English,
German and Spanish share the Latin alphabet, so `sie` is ambiguous and a tag
carries real information. Arabic shares no tokens with Latin script at all — the
script **is** the language signal, and the tag adds nothing the model does not
already see.

**And it is not free.** Two costs, both real:

1. **Inference must reproduce it, and nothing does that for you.** The runner has
   no language handling whatsoever — if you trained with `language_tag: ar`, the
   caller must pass `runner.generate("ar: مرحبا")` by hand. Forget it once and the
   model receives an out-of-distribution prompt. §9.13 lists this as an
   explicitly unenforced invariant.
2. **It perturbs the repetition threshold.** The tag is prepended *before*
   tokenization, and `sample_R(len(text_tokens))` sees the tagged length. So
   `"ar: "` (~2–3 tokens) sits inside every repeated copy and shifts the
   `apply_below: 13` decision (§7) — a small effect, but one more thing to hold
   in your head.

There is also a subtler reason to leave it off: your **replay** data must stay
untagged regardless, because it has to arrive exactly as the base checkpoint saw
it. Tagging only the new language gives you a two-convention corpus. Tagging both
puts the resident languages out of distribution and makes them re-learn what
already worked. Untagged everywhere is the only uniform option.

**When the tag genuinely earns its keep:**

- **Same-script languages** — adding Persian or Urdu alongside Arabic (shared
  script, different phonetics), or several Latin-script languages.
- **Dialect / register control** — MSA vs Egyptian Arabic, where you want to
  *choose* at inference which one comes out. The tag is the only lever in the
  codebase for this.
- **Any case where you want an explicit switch** the model can be conditioned on
  rather than inferring from orthography.

**Decide before you build.** The tag is baked into `text_ids` at prepare time, so
adding or removing it later means rebuilding the corpus and re-running the
fine-tune. If same-script languages or dialect control are anywhere on your
roadmap, put the tag in from the first build — retrofitting is a full cycle.

---

## 11. Checklist

- [ ] Tokenizer produces real subwords for the target language (§2). If it does
      not, stop — this guide does not apply.
- [ ] Target corpus tokenized through the nano-codec pipeline; `nano_layer_1..8`,
      `encoded_len`, transcript, speaker column present.
- [ ] Replay source(s) from the resident languages added to the **same**
      `hf_datasets` list, ratio set via `max_len`, starting around 50/50.
- [ ] `speaker_prefix` pinned on every source.
- [ ] Language tag decision made **before** the build (§10) — off by default.
- [ ] Corpus built with `python -m gepard.cli.prepare text_layout=repeat_ft`
      (not via `make dataset`).
- [ ] `pct_structural_null_exposure` in `speaker_statistics.json` checked against
      the base corpus.
- [ ] Held-out eval sets for **both** the target and the resident languages,
      baselined on the base checkpoint *before* training.
- [ ] Fine-tune via `make finetune EXP="experiment=ft_arabic"` — LoRA only, base
      frozen, no full fine-tune.
- [ ] Short-register thresholds re-swept on the target language (§7); corpus
      rebuilt if they moved.
- [ ] Runaway rate measured. CFG at inference tried before any DPO round (§8).
- [ ] If DPO: multilingual Whisper, refitted duration model, target-language seed
      prompts and reference voices.
