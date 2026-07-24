"""Training utilities and CLI entrypoint for policy/value learning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as functional

from my_chess.chess_ai.dataset import ReplayBuffer, TrainingSample
from my_chess.chess_ai.evaluator import HybridPolicyValueEvaluator
from my_chess.chess_ai.mcts import MCTS, MCTSConfig
from my_chess.chess_ai.model import AlphaZeroNet, NetworkConfig, build_model
from my_chess.chess_ai.self_play import SelfPlayGenerator

DEFAULT_CHECKPOINT_DIR = Path("artifacts") / "checkpoints"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "latest.pt"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for one-step supervised updates."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainingRunConfig:
    """Top-level training-loop configuration."""

    iterations: int = 10
    games_per_iteration: int = 4
    max_moves_per_game: int = 128
    num_simulations: int = 32
    replay_buffer_capacity: int = 4096
    batch_size: int = 64
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH


class AlphaZeroTrainer:
    """Small trainer for policy/value supervised updates."""

    def __init__(
        self,
        model: AlphaZeroNet,
        config: TrainingConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config or TrainingConfig()
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, samples: Iterable[TrainingSample]) -> dict[str, float]:
        """Runs one optimization step over a batch of samples."""
        samples = list(samples)
        if not samples:
            raise ValueError("train_step requires at least one sample")

        states = torch.tensor(
            [sample.state for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        policy_targets = torch.tensor(
            [sample.policy_target for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        value_targets = torch.tensor(
            [sample.value_target for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )

        self.model.train()
        policy_logits, values = self.model(states)

        policy_log_probs = functional.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_targets * policy_log_probs).sum(dim=1).mean()
        value_loss = functional.mse_loss(values, value_targets)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }


def save_checkpoint(
    model: AlphaZeroNet,
    path: str | Path,
    trainer: AlphaZeroTrainer | None = None,
    extra: dict | None = None,
) -> Path:
    """Saves model (and optionally optimizer) state to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "network_config": model.config.__dict__,
    }
    if trainer is not None:
        payload["optimizer_state_dict"] = trainer.optimizer.state_dict()
        payload["training_config"] = trainer.config.__dict__
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_training_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[AlphaZeroNet, dict]:
    """Loads model plus raw checkpoint payload for optimizer/metadata reuse."""
    checkpoint = torch.load(Path(path), map_location=device)
    network_config = NetworkConfig(**checkpoint.get("network_config", {}))
    model = build_model(network_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    trainer: AlphaZeroTrainer | None = None,
) -> AlphaZeroNet:
    """Loads a model checkpoint and optionally restores optimizer state."""
    model, checkpoint = load_training_checkpoint(path, device=device)
    if trainer is not None and "optimizer_state_dict" in checkpoint:
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model


def incremental_train_on_samples(
    samples: Iterable[TrainingSample],
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    steps: int = 8,
    batch_size: int = 64,
    verbose: bool = False,
) -> dict[str, float]:
    """Loads the latest checkpoint, trains on supplied samples, and saves it back."""
    samples = list(samples)
    if not samples:
        raise ValueError("incremental_train_on_samples requires at least one sample")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    if verbose:
        print(
            f"[online-train] start samples={len(samples)} steps={max(1, steps)} "
            f"batch={batch_size} device={device}"
        )

    optimizer_state = None
    extra = {}
    if checkpoint_path.exists():
        if verbose:
            print(f"[online-train] loading checkpoint={checkpoint_path}")
        model, checkpoint = load_training_checkpoint(checkpoint_path, device=device)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        extra = checkpoint.get("extra", {})
    else:
        if verbose:
            print("[online-train] no checkpoint found, building fresh model")
        model = build_model()

    trainer = AlphaZeroTrainer(model, config=training_config, device=device)
    if optimizer_state is not None:
        if verbose:
            print("[online-train] restoring optimizer state")
        trainer.optimizer.load_state_dict(optimizer_state)

    replay_buffer = ReplayBuffer(capacity=max(len(samples), batch_size * 4))
    replay_buffer.extend(samples)
    if verbose:
        print(f"[online-train] replay_buffer={len(replay_buffer)}")

    latest_metrics: dict[str, float] = {}
    total_steps = max(1, steps)
    for step_index in range(1, total_steps + 1):
        batch = replay_buffer.sample_batch(batch_size)
        latest_metrics = trainer.train_step(batch)
        if verbose:
            print(
                "[online-train] "
                f"step={step_index}/{total_steps} "
                f"loss={latest_metrics['loss']:.4f} "
                f"policy={latest_metrics['policy_loss']:.4f} "
                f"value={latest_metrics['value_loss']:.4f}"
            )

    total_games = int(extra.get("human_games", 0)) + 1
    extra.update(
        {
            "human_games": total_games,
            "last_incremental_steps": max(1, steps),
            "last_incremental_batch_size": min(batch_size, len(replay_buffer)),
            "device": device,
        }
    )
    save_checkpoint(
        trainer.model,
        checkpoint_path,
        trainer=trainer,
        extra=extra,
    )
    if verbose:
        print(
            f"[online-train] saved checkpoint={checkpoint_path} human_games={total_games}"
        )
    return latest_metrics


def run_training(
    run_config: TrainingRunConfig,
    training_config: TrainingConfig | None = None,
    device: str | None = None,
    resume: bool = True,
) -> AlphaZeroNet:
    """Runs iterative self-play training and saves checkpoints."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    start_iteration = 1
    checkpoint_extra: dict = {}

    if resume and run_config.checkpoint_path.exists():
        model, checkpoint = load_training_checkpoint(run_config.checkpoint_path, device=device)
        checkpoint_extra = checkpoint.get("extra", {})
        start_iteration = int(checkpoint_extra.get("iteration", 0)) + 1
        print(
            f"[resume] checkpoint={run_config.checkpoint_path} "
            f"last_iteration={checkpoint_extra.get('iteration', 0)} "
            f"buffer={checkpoint_extra.get('buffer_size', 'n/a')}"
        )
    else:
        model = build_model()

    trainer = AlphaZeroTrainer(model, config=training_config, device=device)
    if resume and run_config.checkpoint_path.exists():
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            trainer.optimizer.load_state_dict(optimizer_state)

    evaluator = HybridPolicyValueEvaluator(trainer.model, device=device)
    generator = SelfPlayGenerator(
        MCTS(evaluator, MCTSConfig(num_simulations=run_config.num_simulations))
    )
    replay_buffer = ReplayBuffer(capacity=run_config.replay_buffer_capacity)

    end_iteration = start_iteration + run_config.iterations - 1
    for iteration in range(start_iteration, end_iteration + 1):
        generated = 0
        for _ in range(run_config.games_per_iteration):
            samples = generator.play_game(max_moves=run_config.max_moves_per_game)
            replay_buffer.extend(samples)
            generated += len(samples)

        batch = replay_buffer.samples()[-run_config.batch_size :]
        if not batch:
            raise RuntimeError("No samples were generated for training")

        metrics = trainer.train_step(batch)
        checkpoint_path = save_checkpoint(
            trainer.model,
            run_config.checkpoint_path,
            trainer=trainer,
            extra={
                "iteration": iteration,
                "generated_samples": generated,
                "buffer_size": len(replay_buffer),
                "device": device,
            },
        )
        print(
            (
                f"[iter {iteration}/{end_iteration}] "
                f"generated={generated} "
                f"buffer={len(replay_buffer)} "
                f"loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} "
                f"checkpoint={checkpoint_path}"
            )
        )

    return trainer.model


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI arguments for the training entrypoint."""
    parser = argparse.ArgumentParser(description="Train the Xiangqi AlphaZero model")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=4)
    parser.add_argument("--max-moves-per-game", type=int, default=128)
    parser.add_argument("--num-simulations", type=int, default=32)
    parser.add_argument("--replay-buffer-capacity", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint path to save/load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from a fresh model instead of resuming from checkpoint.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for iterative self-play training."""
    parser = build_arg_parser()
    args = parser.parse_args()

    run_config = TrainingRunConfig(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        max_moves_per_game=args.max_moves_per_game,
        num_simulations=args.num_simulations,
        replay_buffer_capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training_device={device}")
    run_training(
        run_config,
        training_config=training_config,
        device=device,
        resume=not args.fresh,
    )


if __name__ == "__main__":
    main()
