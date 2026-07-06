"""Auxiliary losses: SupCon (supcon), DPO (dpo).

Intentionally no eager re-exports — the data layer imports `losses.supcon`
directly; keeping this __init__ empty avoids import-order cycles.
"""
