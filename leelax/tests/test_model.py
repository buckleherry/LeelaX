import torch

from leelax.net.model import LeelaXNet


def test_model_forward_shapes():
    model = LeelaXNet(in_channels=24, channels=64, num_blocks=4)
    x = torch.randn(2, 24, 8, 8)
    policy_logits, value = model(x)
    assert policy_logits.shape == (2, 4864)
    assert value.shape == (2, 1)


def test_model_no_nans():
    model = LeelaXNet()
    x = torch.randn(1, 24, 8, 8)
    policy_logits, value = model(x)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()

