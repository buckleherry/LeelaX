from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def create_tb_writer(log_dir: str | Path | None = None) -> SummaryWriter:
    if log_dir is None:
        # default to runs/leelax-YYYYmmdd-HHMMSS
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = Path("runs") / f"leelax-{ts}"
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def log_metrics(
    writer: SummaryWriter,
    metrics: Dict[str, Any],
    step: int,
    prefix: str = "train",
) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f"{prefix}/{k}", v, step)

