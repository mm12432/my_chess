"""Self-play sample generation built on top of MCTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from my_chess.chess_ai.action_space import ACTION_SPACE_SIZE
from my_chess.chess_ai.dataset import TrainingSample
from my_chess.chess_ai.encoding import encode_board
from my_chess.chess_ai.mcts import MCTS, MCTSConfig, uniform_policy_value_fn
from my_chess.chess_core.chessboard import Chessboard


@dataclass
class SelfPlayRecord:
    """Intermediate record kept until the game result is known."""

    state: List[List[List[int]]]
    policy_target: List[float]
    player_is_red: bool


class SelfPlayGenerator:
    """Generates self-play games and turns them into training samples."""

    def __init__(self, mcts: MCTS | None = None) -> None:
        self.mcts = mcts or MCTS(uniform_policy_value_fn, MCTSConfig())

    def play_game(self, max_moves: int = 512) -> List[TrainingSample]:
        """Plays one self-play game and returns finished training samples."""
        board = Chessboard("self_play")
        board.init_board()
        records: List[SelfPlayRecord] = []

        for _ in range(max_moves):
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
            best_action_id = max(visit_distribution.items(), key=lambda item: item[1])[0]
            selected_move = self.mcts.action_id_to_legal_move(board, best_action_id)
            board.apply_move(selected_move)

        return self._finalize_records(records, board)

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
