"""
Live dashboard + startup banner for Gepard.

The dashboard renders a single rich.live region containing a per-shard table.
Workers publish progress milestones via ``log_event(logger, msg, **fields)``;
those records carry an ``event`` payload that a small handler attached to the
gepard root logger turns into table updates. Regular log lines (info/warn/
error without an event payload) flow through the same root logger and rich
prints them above the live region — no manual scroll panel needed.

The banner is a one-shot rich render of the resolved configuration followed
by a ``time.sleep`` so the operator can scan settings before the run starts.
The pause length is the constant ``BANNER_PAUSE_SEC`` in ``core.py``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .core import BANNER_PAUSE_SEC, ROOT_LOGGER

_STAGE_STYLE = {
    "queued":    ("dim",     "queued"),
    "started":   ("yellow",  "started"),
    "filtering": ("yellow",  "filter"),
    "filtered":  ("cyan",    "filtered"),
    "mapping":   ("yellow",  "map"),
    "mapped":    ("cyan",    "mapped"),
    "done":      ("green",   "done"),
    "error":     ("red",     "ERROR"),
}

# Coarse stage→percent mapping. HF datasets does not expose per-row progress
# from inside `.filter` / `.map`, and we'd rather not instrument the row
# callables in the processor for this. So the bar advances at milestone
# boundaries: ~half the wall time is filter + map combined, the rest is the
# initial worker spin-up and the final column cleanup.
_STAGE_PCT = {
    "queued": 0,
    "started": 3,
    "filtering": 10,
    "filtered": 50,
    "mapping": 55,
    "mapped": 95,
    "done": 100,
    "error": 0,
}


def _bar(pct: int, width: int = 10) -> str:
    pct = max(0, min(100, pct))
    filled = int(round(width * pct / 100))
    return "█" * filled + "░" * (width - filled)


@dataclass
class _ShardState:
    idx: int
    stage: str = "queued"
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: str = ""
    pct: int = 0

    def elapsed(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at


class _DashboardEventHandler(logging.Handler):
    """Forwards ``record.event`` payloads into the dashboard, ignores the rest.

    Attached to the gepard root logger by ``LiveDashboard.__enter__``. Every
    record with an ``event`` attribute (set by ``log_event``) is consumed
    here; records without one are no-ops for this handler — they continue
    flowing to the file/console handlers normally.
    """

    def __init__(self, dashboard: "LiveDashboard"):
        super().__init__(level=logging.DEBUG)
        self.dashboard = dashboard

    def emit(self, record: logging.LogRecord) -> None:
        ev = getattr(record, "event", None)
        if not ev:
            return
        try:
            self.dashboard.handle_event(ev)
        except Exception:
            # A UI handler must never crash the run.
            pass


class LiveDashboard:
    """Context-manager wrapper around ``rich.live.Live`` for the per-shard table.

    Use as::

        log_state = setup_main_logging(scope="dataset")
        with LiveDashboard(log_state) as dash:
            dash.start_item("emilia", 1, n_total_items, n_shards=20)
            ...run workers...

    The dashboard is stateless about the orchestrator — it only reacts to
    events. The orchestrator drives it via ``start_item``/``finish_item`` to
    reset the table when moving to the next source dataset.
    """

    def __init__(self, log_state: dict):
        self.log_state = log_state
        self.console = log_state["console"]
        self.scope = log_state["scope"]
        self._live: Optional[Live] = None
        self._handler: Optional[_DashboardEventHandler] = None
        self._lock = threading.Lock()

        self._item_name: str = ""
        self._item_idx: int = 0
        self._item_total: int = 0
        self._n_shards: int = 0
        self._shards: dict[int, _ShardState] = {}
        self._global_started: Optional[float] = None

    # ----- public orchestrator API -----

    def __enter__(self) -> "LiveDashboard":
        self._handler = _DashboardEventHandler(self)
        # Register on the root logger so main-process events reach the dashboard,
        # AND on the QueueListener's handler tuple so worker events do too —
        # the listener's tuple was frozen when it started (before this dashboard
        # entered), so root.addHandler alone is not enough.
        logging.getLogger(ROOT_LOGGER).addHandler(self._handler)
        listener = self.log_state.get("listener")
        if listener is not None:
            listener.handlers = (*listener.handlers, self._handler)
        self._global_started = time.monotonic()
        self._live = Live(
            self,
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._handler is not None:
            listener = self.log_state.get("listener")
            if listener is not None:
                listener.handlers = tuple(h for h in listener.handlers if h is not self._handler)
            logging.getLogger(ROOT_LOGGER).removeHandler(self._handler)
            self._handler = None
        return False

    def start_item(self, name: str, idx: int, total: int, n_shards: int) -> None:
        with self._lock:
            self._item_name = name
            self._item_idx = idx
            self._item_total = total
            self._n_shards = n_shards
            self._shards = {i: _ShardState(idx=i) for i in range(n_shards)}

    def finish_item(self) -> None:
        # Keep the last table state visible — no-op for now.
        pass

    # ----- event ingestion (called from listener thread) -----

    def handle_event(self, ev: dict) -> None:
        kind = ev.get("type", "shard")
        if kind == "item_start":
            self.start_item(
                name=ev.get("name", ""),
                idx=int(ev.get("idx", 1)),
                total=int(ev.get("total", 1)),
                n_shards=int(ev.get("n_shards", 0)),
            )
            return
        if kind == "item_done":
            self.finish_item()
            return
        if kind != "shard":
            return
        shard = ev.get("shard")
        if shard is None:
            return
        with self._lock:
            if shard not in self._shards:
                return
            s = self._shards[shard]
            stage = ev.get("stage")
            if stage:
                s.stage = stage
                # Stage-derived pct is monotonic — once done we don't undo it.
                stage_pct = _STAGE_PCT.get(stage, s.pct)
                if stage != "error":
                    s.pct = max(s.pct, stage_pct)
            if "rows_in" in ev:
                s.rows_in = ev["rows_in"]
            if "rows_out" in ev:
                s.rows_out = ev["rows_out"]
            if stage == "started" and s.started_at is None:
                s.started_at = time.monotonic()
            if stage in ("done", "error"):
                s.finished_at = time.monotonic()
            if stage == "error":
                s.error = str(ev.get("error", ""))

    # ----- rich render -----

    def __rich__(self):
        with self._lock:
            n_shards = self._n_shards
            shards = list(self._shards.values())
            item_name = self._item_name
            item_idx = self._item_idx
            item_total = self._item_total

        if n_shards == 0:
            return Panel(
                Text("waiting for first dataset…", style="dim"),
                title=f"{self.scope} run",
                border_style="cyan",
            )

        done = sum(1 for s in shards if s.stage == "done")
        errored = sum(1 for s in shards if s.stage == "error")
        in_flight = n_shards - done - errored
        n_str = f"{done}/{n_shards} done"
        if errored:
            n_str += f" · {errored} ERROR"
        if in_flight:
            n_str += f" · {in_flight} in flight"

        title = f"[{item_idx}/{item_total}] {item_name} — {n_str}"

        table = Table(expand=True, padding=(0, 1))
        table.add_column("#", justify="right", style="dim", no_wrap=True, width=3)
        table.add_column("Stage", no_wrap=True, width=10)
        table.add_column("Progress", no_wrap=True, width=18)
        table.add_column("In", justify="right", no_wrap=True)
        table.add_column("Out", justify="right", no_wrap=True)
        table.add_column("Elapsed", justify="right", no_wrap=True, width=8)
        table.add_column("Note", overflow="ellipsis", no_wrap=True, min_width=0)

        for s in sorted(shards, key=lambda x: x.idx):
            style, label = _STAGE_STYLE.get(s.stage, ("white", s.stage))
            elapsed = s.elapsed()
            elapsed_str = f"{elapsed:5.1f}s" if elapsed is not None else "—"
            rows_in = f"{s.rows_in:,}" if s.rows_in is not None else "—"
            rows_out = f"{s.rows_out:,}" if s.rows_out is not None else "—"
            note = s.error if s.stage == "error" else ""
            bar_style = "red" if s.stage == "error" else (
                "green" if s.pct >= 100 else "cyan"
            )
            progress_cell = Text.assemble(
                (_bar(s.pct), bar_style), " ", (f"{s.pct:3d}%", "dim"),
            )
            table.add_row(
                str(s.idx),
                Text(label, style=style),
                progress_cell,
                rows_in,
                rows_out,
                elapsed_str,
                Text(note, style="red") if note else Text(""),
            )

        return Panel(table, title=title, border_style="cyan")


# ----- banner -----

def print_dataset_banner(cfg, args, log_dir, console=None) -> None:
    """Render the resolved dataset config and pause for the operator.

    Called once from ``gepard.cli.prepare`` before any worker spawns. The pause
    is hardcoded (``BANNER_PAUSE_SEC``) — if/when we add a logging block to
    the YAML config this can read from there instead.
    """
    if console is None:
        from rich.console import Console
        console = Console()

    output = args.output or cfg.data.train_dataset_path

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right", no_wrap=True)
    summary.add_column(style="bold cyan", overflow="fold")
    summary.add_row("Run dir", str(log_dir))
    summary.add_row("Tokenizer", str(cfg.tokens.tokenizer_name))
    summary.add_row("Sources", str(len(cfg.data.hf_datasets)))
    summary.add_row("Max duration", f"{cfg.data.max_duration_sec}s")
    summary.add_row("Shards/source", str(args.n_shards))
    summary.add_row("Output", str(output))
    summary.add_row("Codec layers", str(cfg.codec.num_layers))
    summary.add_row("FSQ levels", str(list(cfg.codec.fsq_levels)))
    summary.add_row("Unfold codes", str(cfg.codec.do_unfold))
    summary.add_row("Frame rate", f"{cfg.codec.frame_rate_hz} Hz")
    summary.add_row("Speaker IDs", _yes_no(cfg.data.add_speaker_id))
    summary.add_row("Row IDs", _yes_no(cfg.data.add_row_id))
    summary.add_row("Singleton policy", str(cfg.data.singleton_policy))
    summary.add_row("Min clips/speaker", str(cfg.data.min_clips_per_speaker))
    summary.add_row("Speaker stats", _yes_no(cfg.data.speaker_statistics))

    console.print(Panel(summary, title="Gepard — Dataset Preparation",
                        border_style="cyan"))

    sources = Table(title="Sources", title_style="bold", expand=True, padding=(0, 1))
    sources.add_column("#", style="dim", width=3, justify="right")
    sources.add_column("Name", overflow="fold")
    sources.add_column("Repo", overflow="fold")
    sources.add_column("Split", no_wrap=True)
    sources.add_column("Lang", no_wrap=True)
    sources.add_column("Cap", justify="right", no_wrap=True)
    for i, item in enumerate(cfg.data.hf_datasets, 1):
        item = dict(item)
        cap = item.get("max_len")
        sources.add_row(
            str(i),
            str(item.get("name") or "—"),
            str(item["reponame"]),
            str(item.get("split") or "—"),
            str(item.get("language_tag") or "—"),
            f"{int(cap):,}" if cap else "—",
        )
    console.print(sources)

    console.print(f"[dim]Starting in {BANNER_PAUSE_SEC}s…[/dim]")
    time.sleep(BANNER_PAUSE_SEC)


def print_train_banner(cfg, log_dir, console=None) -> None:
    """Render the resolved training config as a rich panel, then pause.

    The training analogue of ``print_dataset_banner`` — called once from
    ``gepard.cli.train`` on rank 0 before the model loads. Model parameter
    counts are not known yet (the model is built later in ``trainer.setup``);
    those surface as ordinary log lines during setup. The full field-by-field
    config is written to the log file separately (the caller dumps YAML at
    DEBUG), so this panel stays a scannable summary rather than a wall of keys.
    """
    if console is None:
        from rich.console import Console
        console = Console()

    tr, vc, tl = cfg.trainer, cfg.voice_cloning, cfg.text_layout
    phase = "LoRA fine-tune" if cfg.finetune.lora.enabled else "Pretrain / SFT"
    eff_batch = tr.per_device_train_batch_size * tr.gradient_accumulation_steps

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right", no_wrap=True)
    summary.add_column(style="bold cyan", overflow="fold")
    summary.add_row("Run dir", str(log_dir))
    summary.add_row("Phase", phase)
    summary.add_row("Backbone", str(cfg.model.backbone_id))
    summary.add_row("Precision", str(cfg.model.dtype))
    summary.add_row("Dataset", str(cfg.data.train_dataset_path))
    summary.add_row("Output dir", str(tr.output_dir))
    summary.add_row("Resume from", str(cfg.run.resume_from or "—"))
    summary.add_row("Epochs", str(tr.num_train_epochs))
    summary.add_row(
        "Batch (device x accum)",
        f"{tr.per_device_train_batch_size} x {tr.gradient_accumulation_steps} "
        f"= {eff_batch}/device-step",
    )
    summary.add_row("Learning rate", f"{tr.learning_rate:g}")
    summary.add_row("Scheduler", f"{tr.lr_scheduler_type}, warmup {tr.warmup_steps}")
    summary.add_row("Max grad norm", str(tr.max_grad_norm))
    summary.add_row("Save every", f"{tr.save_steps} steps (keep {tr.save_total_limit})")
    summary.add_row("Voice cloning", _yes_no(vc.enabled))
    summary.add_row("SupCon", _yes_no(vc.training.supcon.enabled))
    summary.add_row(
        "Text repetition",
        f"yes (target {tl.target_text_tokens}, below {tl.apply_below})"
        if tl.enabled else "no",
    )
    summary.add_row(
        "wandb",
        f"{tr.wandb.project}/{tr.wandb.name}" if "wandb" in tr.report_to else "off",
    )

    console.print(Panel(summary, title=f"Gepard — Training ({phase})",
                        border_style="cyan"))
    console.print(f"[dim]Starting in {BANNER_PAUSE_SEC}s…[/dim]")
    time.sleep(BANNER_PAUSE_SEC)


def print_dpo_sample_banner(cfg, log_dir, shard_i, shard_n, console=None) -> None:
    """Render the resolved DPO rollout-sampling config as a rich panel, then pause.

    The Phase-III analogue of ``print_train_banner`` — called once from
    ``gepard.data.dpo.sample`` on shard 0 (the representative writer) before the
    codec and policy load. `text_repetition` is deliberately absent: at sampling
    time it is inherited from the checkpoint's ``gepard_config.json`` (the runner
    reads it there), not from ``conf/dpo.yaml``, so surfacing a config value here
    would be misleading. The full field-by-field config goes to the log file at
    DEBUG, so this panel stays a scannable summary.
    """
    if console is None:
        from rich.console import Console
        console = Console()

    s = cfg.sampling
    cfg_desc = (
        f"scale {s.cfg_scale} @ {'all frames' if s.cfg_frames <= 0 else f'first {s.cfg_frames}'}"
        f" / {s.cfg_uncond_mode}"
        if s.cfg_scale != 1.0 else "off"
    )
    source = (f"pool: {s.speaker_pool}" if s.speaker_pool
              else f"{len(s.ref_audios)} ref audio(s)")

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right", no_wrap=True)
    summary.add_column(style="bold cyan", overflow="fold")
    summary.add_row("Run dir", str(log_dir))
    summary.add_row("Run name", str(cfg.run_name))
    summary.add_row("Policy checkpoint", str(s.checkpoint))
    summary.add_row("Texts file", str(s.texts_file))
    summary.add_row("Speakers", source)
    summary.add_row("Speakers / text", str(s.speakers_per_text))
    summary.add_row("Samples / group", str(s.num_samples))
    summary.add_row("Sampling", f"temp {s.temperature}, top_k {s.top_k}, "
                                f"stop@{s.stop_threshold}")
    summary.add_row("Text-CFG", cfg_desc)
    summary.add_row("Null-prefix prob", f"{s.null_prefix_prob:g}")
    summary.add_row("Shards", str(shard_n))
    summary.add_row("Tokens out", str(cfg.tokens_dir))
    summary.add_row("Frame rate", f"{cfg.codec.frame_rate_hz} Hz")

    console.print(Panel(summary, title="Gepard — DPO Rollout Sampling (stage 1/4)",
                        border_style="cyan"))
    console.print(f"[dim]Starting in {BANNER_PAUSE_SEC}s…[/dim]")
    time.sleep(BANNER_PAUSE_SEC)


def _yes_no(val) -> str:
    return "yes" if val else "no"
