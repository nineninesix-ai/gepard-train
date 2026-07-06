"""DPO data stages: sample (generate candidates), score (decode + WER/SIM),
pairs (build preference pairs + ref logprobs).

Intentionally no eager re-exports: the stages run in two different venvs
(sample/score in venv_dpo with NeMo, pairs/training in the main venv) and each
module pulls its own heavy dependencies lazily. Run as modules:

    python -m gepard.data.dpo.sample --config <dpo.yaml>
    python -m gepard.data.dpo.score  --config <dpo.yaml>
    python -m gepard.data.dpo.pairs  --config <dpo.yaml>
"""
