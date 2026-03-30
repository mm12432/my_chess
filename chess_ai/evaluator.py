"""Policy/value evaluation helpers for search and training."""

from __future__ import annotations

import math
from typing import Dict

import torch

from my_chess.chess_ai.action_space import move_to_action_id
from my_chess.chess_ai.encoding import encode_board
from my_chess.chess_ai.model import AlphaZeroNet
from my_chess.chess_core import chessman
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move

PIECE_VALUES = {
    chessman.King: 1000.0,
    chessman.Rook: 9.0,
    chessman.Cannon: 4.5,
    chessman.Knight: 4.0,
    chessman.Elephant: 2.0,
    chessman.Mandarin: 2.0,
    chessman.Pawn: 1.5,
}


def piece_value(piece: chessman.Chessman | None) -> float:
    """Returns a coarse tactical value for a piece."""
    if piece is None:
        return 0.0
    for piece_type, value in PIECE_VALUES.items():
        if isinstance(piece, piece_type):
            return value
    return 0.0


def heuristic_value(board: Chessboard) -> float:
    """Returns a normalized board value from the current side-to-move view."""
    red_material = 0.0
    black_material = 0.0
    for piece in board.chessmans_hash.values():
        value = piece_value(piece)
        if piece.is_red:
            red_material += value
        else:
            black_material += value

    material_delta = (
        red_material - black_material if board.is_red_turn else black_material - red_material
    )
    return math.tanh(material_delta / 12.0)


def heuristic_move_score(board: Chessboard, move: Move) -> float:
    """Assigns a simple tactical score to one legal move."""
    score = 1.0
    target_piece = board.get_chessman(move.to_point.x, move.to_point.y)
    if target_piece is not None:
        score += piece_value(target_piece) * 3.0

    child_board = board.clone()
    child_board.apply_move(move)
    winner = child_board.get_winner()
    mover_wins = (winner == "Red" and board.is_red_turn) or (
        winner == "Black" and not board.is_red_turn
    )
    if mover_wins:
        score += 10_000.0

    return score


def heuristic_policy_value_fn(board: Chessboard) -> tuple[Dict[int, float], float]:
    """Pure heuristic policy/value estimate used for bootstrapping."""
    legal_moves = board.get_legal_moves()
    if not legal_moves:
        return {}, 0.0

    raw_scores = [heuristic_move_score(board, move) for move in legal_moves]
    total = sum(raw_scores)
    priors = {
        move_to_action_id(move): score / total
        for move, score in zip(legal_moves, raw_scores)
    }
    return priors, heuristic_value(board)


class TorchPolicyValueEvaluator:
    """Wraps a torch model into the MCTS policy/value callable interface."""

    def __init__(self, model: AlphaZeroNet, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, board: Chessboard) -> tuple[Dict[int, float], float]:
        """Returns legal-action priors and scalar value for the current board."""
        self.model.eval()
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return {}, 0.0

        state = torch.tensor(
            [encode_board(board)],
            dtype=torch.float32,
            device=self.device,
        )
        policy_logits, value = self.model(state)
        logits = policy_logits[0]

        action_ids = [move_to_action_id(move) for move in legal_moves]
        legal_logits = logits[action_ids]
        legal_probs = torch.softmax(legal_logits, dim=0)

        priors = {
            action_id: float(probability.item())
            for action_id, probability in zip(action_ids, legal_probs)
        }
        return priors, float(value.item())


class HybridPolicyValueEvaluator:
    """Blends model output with heuristic guidance for practical play/training."""

    def __init__(
        self,
        model: AlphaZeroNet,
        device: str = "cpu",
        model_prior_weight: float = 0.35,
        model_value_weight: float = 0.35,
    ) -> None:
        self.model_evaluator = TorchPolicyValueEvaluator(model, device=device)
        self.model_prior_weight = model_prior_weight
        self.model_value_weight = model_value_weight

    def __call__(self, board: Chessboard) -> tuple[Dict[int, float], float]:
        """Returns blended priors/value for the current board."""
        heuristic_priors, heuristic_eval = heuristic_policy_value_fn(board)
        model_priors, model_eval = self.model_evaluator(board)
        if not heuristic_priors:
            return {}, 0.0

        blended: Dict[int, float] = {}
        for action_id, heuristic_prior in heuristic_priors.items():
            model_prior = model_priors.get(action_id, 0.0)
            blended[action_id] = (
                self.model_prior_weight * model_prior
                + (1.0 - self.model_prior_weight) * heuristic_prior
            )

        total = sum(blended.values())
        if total > 0:
            blended = {
                action_id: prior / total for action_id, prior in blended.items()
            }

        value = (
            self.model_value_weight * model_eval
            + (1.0 - self.model_value_weight) * heuristic_eval
        )
        return blended, value
