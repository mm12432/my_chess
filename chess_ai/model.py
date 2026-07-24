"""Neural-network model definitions for AlphaZero-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE
from my_chess.chess_ai.encoding import BOARD_COLS, BOARD_ROWS, NUM_CHANNELS

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    torch = None
    nn = None
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    import torch as torch_module


@dataclass(frozen=True)
class NetworkConfig:
    """Configures the AlphaZero-style residual CNN."""

    input_channels: int = NUM_CHANNELS
    board_rows: int = BOARD_ROWS
    board_cols: int = BOARD_COLS
    hidden_channels: int = 64
    residual_blocks: int = 4
    policy_size: int = ACTION_SPACE_SIZE


def require_torch() -> None:
    """Raises a clear error if torch has not been installed yet."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install `torch` before using chess_ai.model."
        )


if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        """A standard small residual block for board-feature refinement."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: "torch_module.Tensor") -> "torch_module.Tensor":
            """Runs one residual update."""
            residual = x
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x = self.relu(x + residual)
            return x

    class AlphaZeroNet(nn.Module):
        """Small residual CNN producing policy logits and scalar value."""

        def __init__(self, config: NetworkConfig | None = None) -> None:
            super().__init__()
            self.config = config or NetworkConfig()
            hidden = self.config.hidden_channels
            rows = self.config.board_rows
            cols = self.config.board_cols

            self.stem = nn.Sequential(
                nn.Conv2d(self.config.input_channels, hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            )
            self.backbone = nn.Sequential(
                *[ResidualBlock(hidden) for _ in range(self.config.residual_blocks)]
            )

            self.policy_head = nn.Sequential(
                nn.Conv2d(hidden, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(2 * rows * cols, self.config.policy_size),
            )

            self.value_head = nn.Sequential(
                nn.Conv2d(hidden, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(rows * cols, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
                nn.Tanh(),
            )

        def forward(
            self, x: "torch_module.Tensor"
        ) -> tuple["torch_module.Tensor", "torch_module.Tensor"]:
            """Runs policy/value inference on a batch of encoded boards."""
            x = self.stem(x)
            x = self.backbone(x)
            policy_logits = self.policy_head(x)
            value = self.value_head(x)
            return policy_logits, value.squeeze(-1)


def build_model(config: NetworkConfig | None = None):
    """Constructs the default model when torch is available."""
    require_torch()
    return AlphaZeroNet(config)
