"""Self-play sample generation built on top of MCTS.

Pure AlphaZero labelling: value targets come only from the real game result.
Games that hit the move cap have no known result, so their value targets are
masked out (value_weight=0) instead of being falsely labelled as draws.
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE, canonical_action_id
from my_chess.chess_ai.dataset import TrainingSample
from my_chess.chess_ai.encoding import encode_board
from my_chess.chess_ai.mcts import (
    MCTS,
    MCTSConfig,
    PolicyValueFn,
    uniform_policy_value_fn,
)
from my_chess.chess_core.chessboard import Chessboard

RESULT_TRUNCATED = "truncated"


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
        temperature_moves: int = 30,
    ) -> None:
        self.mcts = mcts or MCTS(uniform_policy_value_fn, MCTSConfig())
        self.temperature = temperature
        self.temperature_moves = temperature_moves
        # "Red" / "Black" / "truncated" for the last finished game.
        self.last_result: str = RESULT_TRUNCATED

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
                    policy_target=dense_policy_target(
                        visit_distribution, board.is_red_turn
                    ),
                    player_is_red=board.is_red_turn,
                )
            )
            temp = self.temperature if move_index < self.temperature_moves else 0.0
            best_action_id = self._sample_action(visit_distribution, temp)
            selected_move = self.mcts.action_id_to_legal_move(board, best_action_id)
            board.apply_move(selected_move)

        winner = board.get_winner()
        self.last_result = winner if winner is not None else RESULT_TRUNCATED
        return self._finalize_records(records, winner)

    @staticmethod
    def _sample_action(visit_distribution: Dict[int, float], temperature: float) -> int:
        """Samples a root action from the visit distribution."""
        items = list(visit_distribution.items())
        if len(items) == 1 or temperature <= 1e-3:
            return max(items, key=lambda item: item[1])[0]

        inv_temp = 1.0 / temperature
        weights = [math.pow(max(prob, 1e-12), inv_temp) for _, prob in items]
        return random.choices([a for a, _ in items], weights=weights, k=1)[0]

    @staticmethod
    def _finalize_records(
        records: List[SelfPlayRecord], winner: str | None
    ) -> List[TrainingSample]:
        """Backfills terminal values; masks value loss for truncated games."""
        value_weight = 0.0 if winner is None else 1.0
        return [
            TrainingSample(
                state=record.state,
                policy_target=record.policy_target,
                value_target=winner_to_value(winner, record.player_is_red),
                value_weight=value_weight,
            )
            for record in records
        ]


def dense_policy_target(
    visit_distribution: Dict[int, float], is_red_turn: bool
) -> List[float]:
    """Converts sparse absolute-id visits into a dense canonical target vector."""
    target = [0.0] * ACTION_SPACE_SIZE
    for action_id, probability in visit_distribution.items():
        target[canonical_action_id(action_id, is_red_turn)] = probability
    return target


def winner_to_value(winner: str | None, player_is_red: bool) -> float:
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
    temperature_moves: int = 30,
) -> tuple[List[TrainingSample], List[str]]:
    """Plays multiple self-play games; returns samples and per-game results."""
    if num_games <= 0:
        return [], []

    workers = max(1, min(workers, num_games))

    def _play_one() -> tuple[List[TrainingSample], str]:
        generator = SelfPlayGenerator(
            MCTS(policy_value_fn, MCTSConfig(num_simulations=num_simulations)),
            temperature=temperature,
            temperature_moves=temperature_moves,
        )
        samples = generator.play_game(max_moves=max_moves)
        return samples, generator.last_result

    samples: List[TrainingSample] = []
    results: List[str] = []
    if workers == 1:
        for _ in range(num_games):
            game_samples, result = _play_one()
            samples.extend(game_samples)
            results.append(result)
        return samples, results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_play_one) for _ in range(num_games)]
        for future in as_completed(futures):
            game_samples, result = future.result()
            samples.extend(game_samples)
            results.append(result)
    return samples, results
