from pathlib import Path
import torch

from leelax.net.model import LeelaXNet
from leelax.train.optim import create_optimizer, create_scheduler
from leelax.train.logger import create_tb_writer, log_metrics


def test_create_optimizer_and_scheduler():
    model = LeelaXNet()
    opt = create_optimizer(model, lr=1e-3)
    sched = create_scheduler(opt, scheduler_type="cosine", T_max=10)
    assert opt is not None
    assert sched is not None

    # correct order: optimizer.step() before scheduler.step()
    opt.step()
    sched.step()


def test_tensorboard_logger(tmp_path: Path):
    writer = create_tb_writer(tmp_path / "runs")
    log_metrics(writer, {"loss": 1.23, "value_loss": 0.5}, step=1)
    writer.close()
    assert any((tmp_path / "runs").iterdir())

