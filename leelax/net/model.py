from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Residual Block
# --------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)

        y = self.conv2(y)
        y = self.bn2(y)

        return F.relu(x + y, inplace=True)


# --------------------------
# Main Net (configurable)
# --------------------------

class LeelaXNet(nn.Module):
    """
    AlphaZero-style architecture with:
    - configurable trunk channels
    - configurable number of residual blocks
    - policy head staying with 76 output channels (per your repo)
    - value head identical to earlier design
    """

    def __init__(self, in_channels: int = 24, channels: int = 64, num_blocks: int = 4):
        super().__init__()

        # Stem
        self.stem    = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # --------------------------
        # Policy head (your version)
        # --------------------------
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_logits_conv = nn.Conv2d(32, 76, 1, bias=True)

        # --------------------------
        # Value head
        # --------------------------
        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(32)
        self.value_fc1  = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2  = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # trunk
        x = F.relu(self.stem_bn(self.stem(x)), inplace=True)
        for blk in self.blocks:
            x = blk(x)

        # policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        p = self.policy_logits_conv(p)  # [B, 76, 8, 8]
        p = p.view(p.size(0), 76 * 8 * 8)  # flatten to match your policy_index logic

        # value head
        v = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = torch.tanh(self.value_fc2(v))

        return p, v


# --------------------------
# 128×6 Variant
# --------------------------

class LeelaXNet128x6(LeelaXNet):
    def __init__(self, in_channels: int = 24):
        super().__init__(in_channels=in_channels, channels=128, num_blocks=6)


# --------------------------
# Factory
# --------------------------

def build_model(model_size: str, in_channels: int = 24) -> nn.Module:
    """
    Factory for selecting model variants:
    - small  → 64×4
    - base   → 96×6
    - 128x6  → 128×6 (recommended)
    """
    ms = model_size.lower()

    if ms == "small":
        return LeelaXNet(in_channels=in_channels, channels=64, num_blocks=4)
    elif ms == "base":
        return LeelaXNet(in_channels=in_channels, channels=96, num_blocks=6)
    elif ms in ("128x6", "large", "xl"):
        return LeelaXNet128x6(in_channels=in_channels)

    # fallback
    return LeelaXNet(in_channels=in_channels, channels=64, num_blocks=4)

