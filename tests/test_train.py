"""Tests for training steps and checkpoint utilities."""

import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE
from my_chess.chess_ai.dataset import TrainingSample
from my_chess.chess_ai.encoding import empty_tensor
from my_chess.chess_ai.model import TORCH_AVAILABLE, build_model
from my_chess.chess_ai.train import AlphaZeroTrainer, load_checkpoint, save_checkpoint


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for training tests")
class TestTrainer(unittest.TestCase):
    """Tests one-step training on synthetic samples."""

    def test_train_step_returns_loss_metrics(self):
        model = build_model()
        trainer = AlphaZeroTrainer(model)
        policy_target = [0.0] * ACTION_SPACE_SIZE
        policy_target[0] = 1.0
        sample = TrainingSample(
            state=empty_tensor(),
            policy_target=policy_target,
            value_target=1.0,
        )

        metrics = trainer.train_step([sample])

        self.assertIn("loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertGreaterEqual(metrics["loss"], 0.0)

    def test_checkpoint_round_trip(self):
        model = build_model()
        trainer = AlphaZeroTrainer(model)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "latest.pt")
            save_checkpoint(trainer.model, checkpoint_path, trainer=trainer)
            loaded_model = load_checkpoint(checkpoint_path)

        self.assertEqual(
            set(loaded_model.state_dict().keys()),
            set(model.state_dict().keys()),
        )


if __name__ == "__main__":
    unittest.main()
