from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = F.relu(out)
        return out


class LeelaXNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        channels: int = 64,
        num_blocks: int = 4,
        policy_channels: int = 32,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        # policy head → 76 channels → 8×8×76 = 4864
        self.policy_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_logits_conv = nn.Conv2d(policy_channels, 76, 1, bias=True)

        # value head
        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        out = self.stem(x)
        out = self.stem_bn(out)
        out = F.relu(out)

        for blk in self.blocks:
            out = blk(out)

        # policy
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = self.policy_logits_conv(p)  # [B, 76, 8, 8]
        B = p.size(0)
        policy_logits = p.view(B, -1)   # [B, 4864]

        # value
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.mean(dim=(2, 3))
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return policy_logits, v

