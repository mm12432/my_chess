"""Self-play sample generation built on top of MCTS.

Pure AlphaZero labelling: games play until a rule-defined outcome
("Red", "Black", or "Draw" under the 120-ply natural move limit) without
artificial mid-game truncation.
"""

from __future__ import annotations

import itertools
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from my_chess.chess_ai.action_space import (
    ACTION_SPACE_SIZE,
    HORIZONTAL_FLIP_INDICES,
    canonical_action_id,
)
from my_chess.chess_ai.dataset import TrainingSample
from my_chess.chess_ai.encoding import encode_board
from my_chess.chess_ai.mcts import (
    MCTS,
    MCTSConfig,
    PolicyValueFn,
    uniform_policy_value_fn,
)
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.rust_bridge import (
    BOARD_COLS,
    BOARD_ROWS,
    NUM_CHANNELS,
    RUST_AVAILABLE,
    RustBoardHandle,
    RustMctsHandle,
)

RESULT_TRUNCATED = "truncated"


@dataclass
class SelfPlayRecord:
    """Intermediate record kept until the game result is known."""

    state: List[List[List[int]]] | np.ndarray
    policy_target: List[float] | np.ndarray
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
        # "Red" / "Black" / "Draw" / "truncated" for the last finished game.
        self.last_result: str = RESULT_TRUNCATED

    def play_game(self, max_moves: int | None = None) -> List[TrainingSample]:
        """Plays one self-play game until rule termination (or optional max_moves)."""
        if RUST_AVAILABLE and hasattr(
            self.mcts.policy_value_fn, "forward_states_numpy"
        ):
            samples, results = play_games_batched_rust(
                self.mcts.policy_value_fn,
                num_games=1,
                num_simulations=self.mcts.config.num_simulations,
                max_moves=max_moves,
                temperature=self.temperature,
                temperature_moves=self.temperature_moves,
                c_puct=self.mcts.config.c_puct,
                dirichlet_alpha=self.mcts.config.dirichlet_alpha,
                dirichlet_epsilon=self.mcts.config.dirichlet_epsilon,
                virtual_loss_batch=self.mcts.config.virtual_loss_batch,
                augment_horizontal=False,
            )
            self.last_result = results[0] if results else RESULT_TRUNCATED
            return samples

        board = Chessboard("self_play")
        board.init_board()
        records: List[SelfPlayRecord] = []

        step_iter = (
            range(max_moves)
            if (max_moves is not None and max_moves > 0)
            else itertools.count()
        )
        for move_index in step_iter:
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
        records: List[SelfPlayRecord],
        winner: str | None,
        augment_horizontal: bool = False,
    ) -> List[TrainingSample]:
        """Backfills terminal values; real outcomes ('Red'/'Black'/'Draw') get weight 1."""
        value_weight = 0.0 if winner is None else 1.0
        samples: List[TrainingSample] = []
        for record in records:
            v_target = winner_to_value(winner, record.player_is_red)
            samples.append(
                TrainingSample(
                    state=record.state,
                    policy_target=record.policy_target,
                    value_target=v_target,
                    value_weight=value_weight,
                )
            )
            if augment_horizontal and isinstance(record.state, np.ndarray):
                flipped_state = np.ascontiguousarray(record.state[:, :, ::-1])
                flipped_policy = np.asarray(record.policy_target, dtype=np.float32)[
                    HORIZONTAL_FLIP_INDICES
                ]
                samples.append(
                    TrainingSample(
                        state=flipped_state,
                        policy_target=flipped_policy,
                        value_target=v_target,
                        value_weight=value_weight,
                    )
                )
        return samples


def dense_policy_target(
    visit_distribution: Dict[int, float], is_red_turn: bool
) -> List[float]:
    """Converts sparse absolute-id visits into a dense canonical target vector."""
    target = [0.0] * ACTION_SPACE_SIZE
    for action_id, probability in visit_distribution.items():
        target[canonical_action_id(action_id, is_red_turn)] = probability
    return target


def winner_to_value(winner: str | None, player_is_red: bool) -> float:
    """Maps winner label ('Red', 'Black', 'Draw', or None) to value target."""
    if winner is None or winner == "Draw":
        return 0.0
    if winner == "Red":
        return 1.0 if player_is_red else -1.0
    if winner == "Black":
        return -1.0 if player_is_red else 1.0
    raise ValueError(f"Unexpected winner label: {winner}")


def play_games_batched_rust(
    policy_value_fn: PolicyValueFn,
    num_games: int,
    num_simulations: int,
    max_moves: int | None = None,
    temperature: float = 1.0,
    temperature_moves: int = 30,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    virtual_loss_batch: int = 8,
    augment_horizontal: bool = True,
) -> tuple[List[TrainingSample], List[str]]:
    """Plays `num_games` concurrently in Rust until rule termination, batching on GPU."""
    if num_games <= 0:
        return [], []

    vl_batch = max(1, int(virtual_loss_batch))
    max_batch_leaves = num_games * vl_batch
    states_buf = np.empty(
        (max_batch_leaves, NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32
    )

    boards = [RustBoardHandle.new_initial() for _ in range(num_games)]
    game_records: List[List[SelfPlayRecord]] = [[] for _ in range(num_games)]
    game_results: List[str | None] = [None] * num_games
    active_indices = list(range(num_games))

    step_iter = (
        range(max_moves)
        if (max_moves is not None and max_moves > 0)
        else itertools.count()
    )

    try:
        for move_index in step_iter:
            if not active_indices:
                break

            if move_index > 0 and move_index % 20 == 0:
                print(
                    f"  [rust-selfplay] ply {move_index}: {len(active_indices)}/{num_games} games active...",
                    flush=True,
                )

            still_active: List[int] = []
            for g_idx in active_indices:
                winner_label = boards[g_idx].get_winner_label()
                if winner_label is not None:
                    game_results[g_idx] = winner_label
                else:
                    still_active.append(g_idx)
            active_indices = still_active
            if not active_indices:
                break

            mcts_list = [
                RustMctsHandle(
                    boards[g_idx],
                    num_simulations=num_simulations,
                    c_puct=c_puct,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon,
                )
                for g_idx in active_indices
            ]

            try:
                while True:
                    leaf_counts: List[int] = []
                    total_leaves = 0
                    for mcts in mcts_list:
                        if mcts.is_finished:
                            leaf_counts.append(0)
                            continue
                        n_leaves = mcts.select_leaves_into(
                            vl_batch,
                            states_buf[total_leaves : total_leaves + vl_batch],
                        )
                        leaf_counts.append(n_leaves)
                        total_leaves += n_leaves

                    if total_leaves == 0:
                        break

                    logits_np, values_np = policy_value_fn.forward_states_numpy(
                        states_buf[:total_leaves]
                    )
                    offset = 0
                    for mcts, count in zip(mcts_list, leaf_counts):
                        if count > 0:
                            mcts.apply_evaluated_leaves(
                                count,
                                logits_np[offset : offset + count],
                                values_np[offset : offset + count],
                            )
                            offset += count

                temp = temperature if move_index < temperature_moves else 0.0
                next_active: List[int] = []
                for g_idx, mcts in zip(active_indices, mcts_list):
                    dist, _ = mcts.get_root_result()
                    if not dist:
                        winner_label = boards[g_idx].get_winner_label()
                        game_results[g_idx] = (
                            winner_label
                            if winner_label is not None
                            else RESULT_TRUNCATED
                        )
                        continue

                    board_h = boards[g_idx]
                    game_records[g_idx].append(
                        SelfPlayRecord(
                            state=board_h.encode_canonical_numpy(),
                            policy_target=mcts.get_canonical_policy_target_numpy(),
                            player_is_red=board_h.is_red_turn,
                        )
                    )
                    chosen_action = SelfPlayGenerator._sample_action(dist, temp)
                    board_h.push_action(chosen_action)
                    next_active.append(g_idx)

                active_indices = next_active
            finally:
                for mcts in mcts_list:
                    mcts.close()

        for g_idx in active_indices:
            if game_results[g_idx] is None:
                winner_label = boards[g_idx].get_winner_label()
                game_results[g_idx] = (
                    winner_label if winner_label is not None else RESULT_TRUNCATED
                )
    finally:
        for b in boards:
            b.close()

    all_samples: List[TrainingSample] = []
    final_results: List[str] = []
    for g_idx in range(num_games):
        res = game_results[g_idx] or RESULT_TRUNCATED
        final_results.append(res)
        winner = res if res in ("Red", "Black", "Draw") else None
        all_samples.extend(
            SelfPlayGenerator._finalize_records(
                game_records[g_idx],
                winner,
                augment_horizontal=augment_horizontal,
            )
        )
    return all_samples, final_results


def play_games_parallel(
    policy_value_fn: PolicyValueFn,
    num_games: int,
    num_simulations: int,
    max_moves: int | None = None,
    workers: int = 1,
    temperature: float = 1.0,
    temperature_moves: int = 30,
) -> tuple[List[TrainingSample], List[str]]:
    """Plays multiple self-play games; returns samples and per-game results."""
    if num_games <= 0:
        return [], []

    if RUST_AVAILABLE and hasattr(policy_value_fn, "forward_states_numpy"):
        return play_games_batched_rust(
            policy_value_fn,
            num_games=num_games,
            num_simulations=num_simulations,
            max_moves=max_moves,
            temperature=temperature,
            temperature_moves=temperature_moves,
            augment_horizontal=True,
        )

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
