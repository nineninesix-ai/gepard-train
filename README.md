<div align="center">
  <img src="assets/logo.png" alt="Gepard Logo" width="100%"/>
  <br><br>

  [![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/NzP3rjB4SB?style=flat)](https://discord.gg/NzP3rjB4SB) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Tech Report](https://img.shields.io/badge/📄-Tech%20Report-red.svg)](https://huggingface.co/nineninesix/gepard-1.0/resolve/main/gepard_techreport.pdf) [![API](https://img.shields.io/badge/⚡-Try%20the%20API-brightgreen.svg)](https://www.nineninesix.ai/) [![Space](https://img.shields.io/badge/🤗-Demo%20Space-yellow.svg)](https://huggingface.co/spaces/nineninesix/gepard)

# Gepard — Train
The reference **training** stack: the data pipeline and the three training stages — pretrain, LoRA fine-tune, and DPO — that produce the model served by [gepard-inference](https://github.com/nineninesix-ai/gepard-inference.git).
</div>

## About

**GEPARD** — **Ge**nerative, **P**rosody-aware, **A**utoregressive text-to-speech for **R**ealtime **D**ialogues — is a decoder-only TTS model built to be served by a *stock* LLM engine (vLLM) without custom CUDA kernels in the decode loop. A standard full-attention Qwen3.5 backbone predicts discrete audio codes; an FSQ-based NVIDIA NanoCodec turns them into a 22.05 kHz waveform. Everything non-standard — zero-shot voice cloning, text-repetition, classifier-free guidance — is kept out of the autoregressive loop or distilled into the weights, so generation stays **single-pass**.

This repository is the **training** side of Gepard. It takes a tokenized speech corpus and runs the full lifecycle end to end:

- **Dataset build** — encode raw audio to NanoCodec tokens and assemble the training corpus.
- **Pretrain** — the base model on the full corpus (4-GPU FSDP), with zero-shot voice cloning trained via a Q-Former compressor + SupCon.
- **LoRA fine-tune (SFT)** — short-phrase adaptation with frozen backbone + adapters.
- **DPO** — offline CFG distillation: self-generated preference pairs bake the quality of two-pass classifier-free guidance into single-pass weights.
- **Merge & publish** — fold adapters, stamp self-describing checkpoints, and upload Hugging Face-ready models with an auto-generated model card.

Everything is driven by one unified [Hydra](https://hydra.cc) config tree (`conf/`) with a single entry file per phase. The design, architecture, data pipeline, and every training stage are documented in **[docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md)**. Adding a language the model has never seen has its own recipe: **[docs/NEW_LANGUAGE_GUIDE.md](docs/NEW_LANGUAGE_GUIDE.md)**. Full experimental detail is in the technical report (**[gepard_techreport.pdf](gepard_techreport.pdf)**).

The model these stages produce is:

- **vLLM-native, single-pass** — the whole audio frame (32 orthogonal FSQ channels) is sampled in one step, no depth-transformer.
- **Real-time on vLLM** — a single stream runs at ≈ 0.040 RTF with ≈ 0.032 s time-to-first-audio on an RTX 5090, scaling to ≈ 204× aggregate throughput on one server-class GPU.
- **CFG distilled via DPO** — two-pass classifier-free-guidance quality baked into the weights, free at serving time (CFG stays available as an optional quality lever).
- **Zero-shot voice cloning** — clone a voice from a short reference clip; the speaker profile is extracted once at prefill and never enters the decode loop.

## Installation

Requires a CUDA GPU and Python 3.12. The setup script builds a local `venv/` with a CUDA-matched PyTorch, the NeMo codec, and the Gepard training package.

```bash
# 1. (once per machine, optional) system packages — nvcc, python3.12 headers, git-lfs
make system-deps

# 2. create venv/ and install the training stack
make setup

# 3. authenticate Hugging Face (dataset + checkpoint pull/push)
make login
```

Two optional environments layer on top: `make setup_dpo` (NeMo codec + Whisper + WER, for the DPO data pipeline) and `make setup_inference` (to run the demo notebook / smoke-test a trained checkpoint).

## Quick Start

Point at a source corpus and run the stages. The one fully public, pre-tokenized example dataset — already encoded with the default codec — is [`nineninesix/emolia_filtered_nano_codec_21_dataset`](https://huggingface.co/datasets/nineninesix/emolia_filtered_nano_codec_21_dataset) (a filtered slice of **Emilia**); uncomment its block in [`conf/data/full_corpus.yaml`](conf/data/full_corpus.yaml) to build on open data end to end.

```bash
make dataset       # build the tokenized training corpus (prepare stage)
make train         # pretrain the base model (4-GPU FSDP)
make finetune      # LoRA short-phrase fine-tune (single GPU)
```

DPO (optional, four stages — see MODEL_GUIDE §7 / §12.5):

```bash
make setup_dpo
make dpo-sample-sharded SHARDS=4     # rollouts (run `make dpo-progress` in a 2nd terminal)
make dpo-score && make dpo-pairs     # reward + preference pairs
make dpo-train                       # LoRA + stop-head DPO
```

Publish a trained checkpoint:

```bash
make merge  CHECKPOINT=checkpoints/checkpoint-N     # fold LoRA adapters (for fine-tune (sft) only)
make upload REPO=you/model CHECKPOINT=<merged-dir>  # bf16 + auto-generated model card
```

Every stage composes its config from `conf/` and validates it before any GPU time — see MODEL_GUIDE §8 (config system) and §12 (training). One-off variations layer on an entry via `experiment=` presets (`conf/experiment/`, §12.8).

**Demo notebook.** [`notebooks/inference_demo.ipynb`](notebooks/inference_demo.ipynb) is meant for quick **in-project** checks — loading a just-trained checkpoint and listening to it without leaving the repo. For pure inference (no training dependencies), the [gepard-inference](https://github.com/nineninesix-ai/gepard-inference.git) repo ships an equivalent Colab.

## Commands

```bash
make help          # show all commands
make system-deps   # install apt deps (nvcc, python3.12, git-lfs) — sudo, once
make setup         # build venv/ and install the training stack
make login         # authenticate Hugging Face
make dataset       # build the tokenized training corpus
make train         # pretrain (4-GPU FSDP)         │ make resume CHECKPOINT=…
make finetune      # LoRA fine-tune (single GPU)    │ make finetune-resume CHECKPOINT=…
make dpo-dataset   # DPO stages 1–3 (sample→score→pairs)
make dpo-train     # DPO stage 4 (LoRA + stop head)
make merge         # fold LoRA adapters into a servable checkpoint
make upload        # push a checkpoint + model card to the Hub
make test          # config + golden-baseline tests
```

## Related Repositories

- **[gepard-inference](https://github.com/nineninesix-ai/gepard-inference.git)** — the transformers-based inference stack (server + client, plus a ready-to-run Colab for pure inference).
- **[gepard-vllm](https://github.com/nineninesix-ai/gepard-vllm)** — the vLLM serving "wheels": the production, continuous-batching inference path.
- **[dataset processing pipeline](https://github.com/nineninesix-ai/nano-codec-processing-pipeline.git)** - Prepare your own audio dataset using nanocodec tokenization.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{gepard_2026,
  author = {Abdurazakov Ulanbek, Pavlov Denis, and Bakashov Nursultan},
  title = {Gepard: Real-Time Decoder-Only TTS Native to vLLM},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/nineninesix/gepard-1.0}},
  note = {Open-source, vLLM-native autoregressive TTS}
}
```

## References

Gepard builds on and is inspired by a great deal of open work. The main pieces:

```bibtex
@misc{qwen3,
  title={Qwen3 Technical Report},
  author={Qwen Team},
  year={2025},
  eprint={2505.09388},
  archivePrefix={arXiv}
}

@inproceedings{kwon2023vllm,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E and Zhang, Hao and Stoica, Ion},
  booktitle={Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)},
  pages={611--626},
  year={2023},
  eprint={2309.06180},
  archivePrefix={arXiv}
}


@article{nvidia2025nanocodec,
  title={NanoCodec: Towards High-Quality Ultra Fast Speech LLM Inference},
  author={Casanova, Edresson and Neekhara, Paarth and Langman, Ryan and Hussain, Shehzeen and Ghosh, Subhankar and Yang, Xuesong and Juki{\'c}, Ante and Li, Jason and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2508.05835},
  year={2025}
}

@article{he2024emilia,
  title={Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation},
  author={He, Haorui and Shang, Zengqiang and Wang, Chaoren and Li, Xuyuan and Gu, Yicheng and Hua, Hua and Liu, Liwei and Yang, Chen and Li, Jiaqi and Shi, Peiyang and Jin, Zhizheng and others},
  journal={arXiv preprint arXiv:2407.05361},
  year={2024}
}

@article{mentzer2023fsq,
  title={Finite Scalar Quantization: VQ-VAE Made Simple},
  author={Mentzer, Fabian and Agustsson, Eirikur and Tschannen, Michael and Malireddy, Srikanth and Alshina, Elena},
  journal={arXiv preprint arXiv:2309.15505},
  year={2023}
}

@article{nvidia2024magpie,
  title={Improving Robustness of LLM-based Speech Synthesis by Learning Monotonic Alignment},
  author={Neekhara, Paarth and Hussain, Shehzeen and Ghosh, Subhankar and Li, Jason and Valle, Rafael and Badlani, Rohan and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2406.17957},
  year={2024}
}

@article{ho2022cfg,
  title={Classifier-Free Diffusion Guidance},
  author={Ho, Jonathan and Salimans, Tim},
  journal={arXiv preprint arXiv:2207.12598},
  year={2022}
}

@article{rafailov2023dpo,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}

@article{meng2024simpo,
  title={SimPO: Simple Preference Optimization with a Reference-Free Reward},
  author={Meng, Yu and Xia, Mengzhou and Chen, Danqi},
  journal={arXiv preprint arXiv:2405.14734},
  year={2024}
}



@article{voicestar2025,
  title={VoiceStar: Robust Zero-Shot Autoregressive TTS with Duration Control and Extrapolation},
  author={Peng, Puyuan and Li, Shang-Wen and Mohamed, Abdelrahman and Harwath, David},
  journal={arXiv preprint arXiv:2505.19462},
  year={2025}
}



```

## License

Apache 2.0 — see the [LICENSE](LICENSE) file for details.

Gepard loads the NVIDIA NeMo **NanoCodec** (`nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`) at runtime. That model is not covered by Apache 2.0 — it is governed by the [NVIDIA Open Model License Agreement](https://developer.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf). See the [NOTICE](NOTICE) file for third-party attribution.
