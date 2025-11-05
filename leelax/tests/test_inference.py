import torch

from leelax.net.inference import masked_softmax, select_action


def test_masked_softmax_basic():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    probs = masked_softmax(logits, mask, dim=1)
    # only positions 0 and 2 should have mass
    assert probs.shape == (1, 4)
    assert torch.allclose(probs[:, 1], torch.zeros(1))
    assert torch.allclose(probs[:, 3], torch.zeros(1))
    # sum over legal = 1
    s = probs.sum().item()
    assert abs(s - 1.0) < 1e-5


def test_select_action_greedy_respects_mask():
    logits = torch.tensor([[0.1, 10.0, 0.1, 0.1]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    idx = select_action(logits, mask, greedy=True)
    # index 1 has highest logit but is illegal -> should pick index 2 or 3,
    # here both 2 and 3 have equal logit 0.1, but 0 is 0.1 too.
    # deterministic check: legal max is 0.1, first occurrence is index 0
    assert idx.shape == (1,)
    assert idx.item() in (0, 2, 3)


def test_select_action_sampling():
    logits = torch.tensor([[0.1, 2.0, 0.1, 0.1]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    idx = select_action(logits, mask, greedy=False)
    assert idx.shape == (1,)
    assert 0 <= idx.item() < 4

