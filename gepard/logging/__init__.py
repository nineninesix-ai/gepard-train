"""Unified logging for the Gepard pipeline (dataset preparation, training).

Public API:

    from gepard.logging import (
        get_logger, log_event,
        setup_main_logging, shutdown_logging, init_worker_logging,
        LiveDashboard, print_dataset_banner,
        BANNER_PAUSE_SEC,
    )

See ``gepard.logging.core`` for the hierarchy + file routing rules and
``gepard.logging.dashboard`` for the rich.live UI.
"""

from .core import (
    BANNER_PAUSE_SEC,
    LOGS_DIR,
    ROOT_LOGGER,
    enable_hf_progress,
    get_logger,
    init_worker_logging,
    log_event,
    make_run_id,
    setup_main_logging,
    setup_train_logging,
    shutdown_logging,
    shutdown_train_logging,
)
from .dashboard import (
    LiveDashboard,
    print_dataset_banner,
    print_dpo_sample_banner,
    print_train_banner,
)
from .model_card import (
    build_dpo_training_metadata,
    build_training_metadata,
    load_training_metadata,
    render_model_card,
    write_dpo_training_metadata,
    write_model_card,
    write_training_metadata,
)

__all__ = [
    "build_dpo_training_metadata",
    "build_training_metadata",
    "load_training_metadata",
    "render_model_card",
    "write_dpo_training_metadata",
    "write_model_card",
    "write_training_metadata",
    "BANNER_PAUSE_SEC",
    "LOGS_DIR",
    "ROOT_LOGGER",
    "LiveDashboard",
    "enable_hf_progress",
    "get_logger",
    "init_worker_logging",
    "log_event",
    "make_run_id",
    "print_dataset_banner",
    "print_dpo_sample_banner",
    "print_train_banner",
    "setup_main_logging",
    "setup_train_logging",
    "shutdown_logging",
    "shutdown_train_logging",
]
