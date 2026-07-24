"""Self-play sample generation built on top of MCTS."""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE
from my_chess.chess_ai.dataset import TrainingSample
from my_chess.chess_ai.encoding import encode_board
from my_chess.chess_ai.mcts import (
    MCTS,
    MCTSConfig,
    PolicyValueFn,
    uniform_policy_value_fn,
)
from my_chess.chess_core.chessboard import Chessboard


@dataclass
class SelfPlayRecord:
    """Intermediate record kept until the game result is known."""

    state: List[List[List[int]]]
    policy_target: List[float]
    player_is_red: bool


class SelfPlayGenerator:
    """Generates self-play games and turns them into training samples."""

    def __init__(
        self,
        mcts: MCTS | None = None,
        temperature: float = 1.0,
        temperature_moves: int = 12,
    ) -> None:
        self.mcts = mcts or MCTS(uniform_policy_value_fn, MCTSConfig())
        self.temperature = temperature
        self.temperature_moves = temperature_moves

    def play_game(self, max_moves: int = 512) -> List[TrainingSample]:
        """Plays one self-play game and returns finished training samples."""
        board = Chessboard("self_play")
        board.init_board()
        records: List[SelfPlayRecord] = []

        for move_index in range(max_moves):
            if board.is_end():
                break

            visit_distribution = self.mcts.run(board)
            if not visit_distribution:
                break

            records.append(
                SelfPlayRecord(
                    state=encode_board(board),
                    policy_target=self._dense_policy_target(visit_distribution),
                    player_is_red=board.is_red_turn,
                )
            )
            temp = self.temperature if move_index < self.temperature_moves else 1e-3
            best_action_id = self._sample_action(visit_distribution, temp)
            selected_move = self.mcts.action_id_to_legal_move(board, best_action_id)
            board.apply_move(selected_move)

        return self._finalize_records(records, board)

    @staticmethod
    def _sample_action(visit_distribution: Dict[int, float], temperature: float) -> int:
        """Samples a root action from the visit distribution."""
        items = list(visit_distribution.items())
        if len(items) == 1 or temperature <= 1e-3:
            return max(items, key=lambda item: item[1])[0]

        action_ids = [action_id for action_id, _ in items]
        visits = [max(prob, 1e-12) for _, prob in items]
        inv_temp = 1.0 / temperature
        weights = [math.pow(visit, inv_temp) for visit in visits]
        total = sum(weights)
        thresholds = []
        cumulative = 0.0
        for weight in weights:
            cumulative += weight / total
            thresholds.append(cumulative)
        probe = random.random()
        for action_id, threshold in zip(action_ids, thresholds):
            if probe <= threshold:
                return action_id
        return action_ids[-1]

    @staticmethod
    def _dense_policy_target(visit_distribution: Dict[int, float]) -> List[float]:
        """Converts sparse root visits into a dense action-space vector."""
        target = [0.0] * ACTION_SPACE_SIZE
        for action_id, probability in visit_distribution.items():
            target[action_id] = probability
        return target

    @staticmethod
    def _finalize_records(
        records: List[SelfPlayRecord], final_board: Chessboard
    ) -> List[TrainingSample]:
        """Backfills terminal values into collected records."""
        winner = final_board.get_winner()
        samples: List[TrainingSample] = []
        for record in records:
            value_target = SelfPlayGenerator._winner_to_value(
                winner, record.player_is_red
            )
            samples.append(
                TrainingSample(
                    state=record.state,
                    policy_target=record.policy_target,
                    value_target=value_target,
                )
            )
        return samples

    @staticmethod
    def _winner_to_value(winner: str | None, player_is_red: bool) -> float:
        """Maps winner label to value target from the acting player's view."""
        if winner is None:
            return 0.0
        if winner == "Red":
            return 1.0 if player_is_red else -1.0
        if winner == "Black":
            return -1.0 if player_is_red else 1.0
        raise ValueError(f"Unexpected winner label: {winner}")


def play_games_parallel(
    policy_value_fn: PolicyValueFn,
    num_games: int,
    num_simulations: int,
    max_moves: int,
    workers: int = 1,
    temperature: float = 1.0,
    temperature_moves: int = 12,
) -> List[TrainingSample]:
    """Plays multiple self-play games, optionally in parallel threads."""
    if num_games <= 0:
        return []

    workers = max(1, min(workers, num_games))

    def _play_one() -> List[TrainingSample]:
        generator = SelfPlayGenerator(
            MCTS(
                policy_value_fn,
                MCTSConfig(num_simulations=num_simulations),
            ),
            temperature=temperature,
            temperature_moves=temperature_moves,
        )
        return generator.play_game(max_moves=max_moves)

    if workers == 1:
        samples: List[TrainingSample] = []
        for _ in range(num_games):
            samples.extend(_play_one())
        return samples

    samples = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_play_one) for _ in range(num_games)]
        for future in as_completed(futures):
            samples.extend(future.result())
    return samples
