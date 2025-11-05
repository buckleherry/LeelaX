from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple 2-layer ResNet block for 8x8 chess inputs."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
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
    """Small AlphaZero-style policy/value network.

    Input:
        x: [B, 24, 8, 8]  (24 env planes)

    Output:
        policy_logits: [B, 4672]  (8*8*73)
        value:         [B, 1]     (tanh)
    """

    def __init__(
        self,
        in_channels: int = 24,
        channels: int = 64,
        num_blocks: int = 4,
        policy_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks

        # stem
        self.stem = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        # residual tower
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # policy head
        self.policy_conv = nn.Conv2d(
            channels, policy_channels, kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        # final conv to 73 move types per square
        self.policy_logits_conv = nn.Conv2d(
            policy_channels, 73, kernel_size=1, bias=True
        )

        # value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 24, 8, 8]
        out = self.stem(x)
        out = self.stem_bn(out)
        out = F.relu(out)

        for block in self.blocks:
            out = block(out)

        # --- policy head ---
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = self.policy_logits_conv(p)  # [B, 73, 8, 8]
        # flatten to [B, 4672]
        B = p.size(0)
        policy_logits = p.view(B, -1)

        # --- value head ---
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        # global spatial average: [B, 32, 1, 1]
        v = v.mean(dim=(2, 3))
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # [B, 1]

        return policy_logits, v
