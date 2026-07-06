"""
Logging core for Gepard.

Single source of truth for the logger hierarchy, file routing, and the
multiprocessing-safe log channel used by the dataset-prep workers.

Hierarchy
---------
All loggers live under ``gepard.*``. The ``scope`` argument to
``setup_main_logging`` selects the second segment used for file naming and
filter routing — currently ``"dataset"`` (used by ``gepard.cli.prepare``) and,
later, ``"train"`` (for ``gepard.cli.train``).

Per-run layout (created on every entry to ``setup_main_logging``)::

    logs/{scope}_{YYYYMMDD_HHMMSS}/
        {scope}_main.log        # everything under gepard.*
        {scope}_processor.log   # everything except gepard.{scope}.shard.*
        {scope}_shards.log      # only gepard.{scope}.shard.* (worker output)

File routing is implemented with two filters (``PrefixFilter`` /
``ExcludePrefixFilter``) so the hierarchy stays in one place.

Multiprocessing
---------------
Workers (``ProcessPoolExecutor``) call ``init_worker_logging(queue)`` first
thing in their entry point. That clears any inherited handlers and replaces
them with a single ``QueueHandler``. The main process runs a ``QueueListener``
that fans every worker record back into the same handler chain (file +
console + dashboard), so worker logs end up in the same files as orchestration
logs and never interleave on stdout.

Console output
--------------
The console handler is a ``rich.logging.RichHandler`` bound to a shared
``Console`` instance. ``LiveDashboard`` re-uses that exact console for its
``Live`` region, which lets log lines auto-scroll above the table while it
renders.

Dashboard events
----------------
Workers (and the orchestrator) annotate progress milestones with
``logger.info("...", extra={"event": {...}})``. These are routed to the live
dashboard alongside being recorded as normal log lines. ``log_event`` is the
canonical helper.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT_LOGGER = "gepard"
BANNER_PAUSE_SEC = 3
LOGS_DIR = Path("logs")

_DATEFMT = "%H:%M:%S"
_FILE_FORMAT = "%(asctime)s %(levelname)-5s %(short_name)-22s %(message)s"
_CONSOLE_FORMAT = "%(short_name)-22s %(message)s"


class ShortNameFilter(logging.Filter):
    """Strip the ``gepard.`` prefix into ``record.short_name`` for compact output."""

    def filter(self, record: logging.LogRecord) -> bool:
        n = record.name
        prefix = ROOT_LOGGER + "."
        record.short_name = n[len(prefix):] if n.startswith(prefix) else n
        return True


class PrefixFilter(logging.Filter):
    """Accept only records whose name starts with one of ``prefixes``."""

    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name == p or record.name.startswith(p + ".") for p in self.prefixes)


class ExcludePrefixFilter(logging.Filter):
    """Reject records whose name starts with one of ``prefixes``."""

    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(record.name == p or record.name.startswith(p + ".") for p in self.prefixes)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the gepard root.

    Pass either the leaf name (``"dataset.processor"``) or the fully qualified
    name (``"gepard.dataset.processor"``).
    """
    if not name.startswith(ROOT_LOGGER + ".") and name != ROOT_LOGGER:
        name = f"{ROOT_LOGGER}.{name}"
    return logging.getLogger(name)


def make_run_id(scope: str) -> str:
    return f"{scope}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Emit a log record carrying a structured event payload.

    The payload is attached as ``record.event`` so handlers can route on it
    (the live dashboard does). The textual ``message`` is what shows in files
    and the console scroll, so keep it human-readable.
    """
    logger.info(message, extra={"event": fields})


def _silence_hf_progress() -> None:
    """Disable HuggingFace datasets' tqdm bars and lower its log verbosity.

    Called both in the main process (after dashboard wires up) and in every
    worker (in ``init_worker_logging``). Otherwise ``dataset.filter``,
    ``dataset.map`` and ``load_dataset(num_proc=...)`` flood stdout with
    progress bars from N processes simultaneously.
    """
    try:
        import datasets
        datasets.disable_progress_bars()
        datasets.logging.set_verbosity_error()
    except Exception:
        pass


def enable_hf_progress() -> None:
    """Re-enable HuggingFace datasets' tqdm bars.

    Use this around long-running calls that the live dashboard doesn't cover
    (e.g. ``Dataset.save_to_disk`` after the dashboard has exited) — HF will
    print its native shard-write progress to stderr. Safe to call repeatedly.
    """
    try:
        import datasets
        datasets.enable_progress_bars()
    except Exception:
        pass


def _build_file_handler(path: Path, level: int) -> logging.FileHandler:
    h = logging.FileHandler(path, encoding="utf-8")
    h.setLevel(level)
    h.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATEFMT))
    h.addFilter(ShortNameFilter())
    return h


def setup_main_logging(
    scope: str = "dataset",
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> dict:
    """Wire file + console handlers, start a QueueListener for worker logs.

    Returns a state dict containing ``run_id``, ``log_dir``, ``queue`` (the
    ``mp.Queue`` to pass to workers), ``listener`` (the QueueListener; stop
    it via ``shutdown_logging``), and ``console`` (the shared rich Console).
    """
    run_id = make_run_id(scope)
    if log_dir is None:
        log_dir = LOGS_DIR / run_id
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.propagate = False

    main_h = _build_file_handler(log_dir / f"{scope}_main.log", file_level)
    root.addHandler(main_h)

    proc_h = _build_file_handler(log_dir / f"{scope}_processor.log", file_level)
    proc_h.addFilter(ExcludePrefixFilter((f"{ROOT_LOGGER}.{scope}.shard",)))
    root.addHandler(proc_h)

    shard_h = _build_file_handler(log_dir / f"{scope}_shards.log", file_level)
    shard_h.addFilter(PrefixFilter((f"{ROOT_LOGGER}.{scope}.shard",)))
    root.addHandler(shard_h)

    from rich.console import Console
    from rich.logging import RichHandler

    console = Console(stderr=True, soft_wrap=False)
    rich_h = RichHandler(
        console=console,
        level=console_level,
        show_path=False,
        show_time=True,
        omit_repeated_times=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format=_DATEFMT,
    )
    rich_h.addFilter(ShortNameFilter())
    rich_h.setFormatter(logging.Formatter("%(short_name)-22s %(message)s"))
    root.addHandler(rich_h)

    # Manager().Queue() is a proxy queue: unlike mp.Queue(), it can be passed
    # as a positional/keyword argument to ProcessPoolExecutor.submit (mp.Queue
    # is only sharable via fork-inheritance, which ProcessPoolExecutor never
    # uses since it pickles every argument). The manager reference is held
    # in the state dict so its background process stays alive for the run.
    manager = mp.Manager()
    log_queue = manager.Queue(-1)
    listener = logging.handlers.QueueListener(
        log_queue, *root.handlers, respect_handler_level=True
    )
    listener.start()

    _silence_hf_progress()

    return {
        "run_id": run_id,
        "log_dir": log_dir,
        "queue": log_queue,
        "listener": listener,
        "manager": manager,
        "console": console,
        "scope": scope,
    }


def shutdown_logging(state: dict) -> None:
    """Stop the QueueListener and tear down the manager. Safe to call once."""
    listener = state.get("listener")
    if listener is not None:
        listener.stop()
        state["listener"] = None
    state["queue"] = None
    manager = state.get("manager")
    if manager is not None:
        try:
            manager.shutdown()
        except Exception:
            pass
        state["manager"] = None


def setup_train_logging(
    scope: str = "train",
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    is_main: bool = True,
    filename: Optional[str] = None,
) -> dict:
    """Wire file + console logging for a training / pipeline entry point.

    The training analogue of ``setup_main_logging``, deliberately lighter.
    Training parallelism is separate processes (torchrun ranks, or the DPO
    sampler's shard processes) — not a ``ProcessPoolExecutor`` worker pool — so
    there is **no** ``mp.Queue`` / ``QueueListener`` here (that machinery exists
    only to funnel dataset-prep workers back to the parent). The active process
    gets a plain ``FileHandler`` plus a rich console handler; when ``is_main``
    is False the process gets a single ``NullHandler`` so its ``gepard.*``
    records are dropped instead of duplicating the file or interleaving.

    File path: ``log_dir/filename`` when ``filename`` is given (used by the DPO
    sampler so each shard writes its own ``shard{i}.log`` into one shared run
    dir), else ``log_dir/{scope}_main.log``. ``log_dir`` itself defaults to a
    fresh timestamped ``logs/{scope}_{ts}/`` when not passed.

    ``is_main`` is the rank-0 gate for torchrun (one writer per multi-rank run);
    the DPO sampler instead passes ``is_main=True`` on every shard with a
    distinct ``filename``, because each shard is its own independent writer.

    Only the ``gepard.*`` hierarchy is touched (``propagate = False``): the
    ``wandb`` and ``transformers`` loggers are left completely alone, so this
    never intercepts, reformats, or files wandb output.

    Returns a state dict (``run_id``, ``log_dir``, ``console``, ``scope``);
    pass it to ``shutdown_train_logging`` in a ``finally`` block.
    """
    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.propagate = False

    if not is_main:
        root.addHandler(logging.NullHandler())
        return {"run_id": None, "log_dir": None, "console": None, "scope": scope}

    run_id = make_run_id(scope)
    if log_dir is None:
        log_dir = LOGS_DIR / run_id
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root.addHandler(_build_file_handler(log_dir / (filename or f"{scope}_main.log"), file_level))

    from rich.console import Console
    from rich.logging import RichHandler

    console = Console(stderr=True, soft_wrap=False)
    rich_h = RichHandler(
        console=console,
        level=console_level,
        show_path=False,
        show_time=True,
        omit_repeated_times=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format=_DATEFMT,
    )
    rich_h.addFilter(ShortNameFilter())
    rich_h.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root.addHandler(rich_h)

    return {
        "run_id": run_id,
        "log_dir": log_dir,
        "console": console,
        "scope": scope,
    }


def shutdown_train_logging(state: dict) -> None:
    """Flush and close the training handlers on the gepard root. Safe once."""
    root = logging.getLogger(ROOT_LOGGER)
    for h in list(root.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        root.removeHandler(h)


def init_worker_logging(queue, level: int = logging.INFO) -> None:
    """First call in every worker process.

    Replaces any inherited handlers on the gepard root with a single
    ``QueueHandler`` pointing at the parent's queue, so all worker logs
    surface in the parent without stdout interleave. Also silences the
    HuggingFace datasets progress bars inside the worker.
    """
    root = logging.getLogger(ROOT_LOGGER)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.propagate = False

    _silence_hf_progress()
