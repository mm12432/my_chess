"""Small end-to-end training smoke test entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_project_imports() -> None:
    """Ensures `my_chess` imports work when executing this file directly."""
    project_root = Path(__file__).resolve().parents[1]
    project_parent = project_root.parent
    if str(project_parent) not in sys.path:
        sys.path.insert(0, str(project_parent))


setup_project_imports()

import torch

from my_chess.chess_ai.dataset import ReplayBuffer
from my_chess.chess_ai.evaluator import TorchPolicyValueEvaluator
from my_chess.chess_ai.mcts import MCTS, MCTSConfig
from my_chess.chess_ai.model import build_model
from my_chess.chess_ai.self_play import SelfPlayGenerator
from my_chess.chess_ai.train import AlphaZeroTrainer


def main() -> None:
    """Generates a small batch from self-play and trains one step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model()
    trainer = AlphaZeroTrainer(model, device=device)
    evaluator = TorchPolicyValueEvaluator(model, device=device)
    generator = SelfPlayGenerator(MCTS(evaluator, MCTSConfig(num_simulations=16)))
    buffer = ReplayBuffer(capacity=256)

    samples = generator.play_game(max_moves=8)
    buffer.extend(samples)
    metrics = trainer.train_step(buffer.samples())
    print(f"device={device}")
    print(metrics)


if __name__ == "__main__":
    main()
