"""Policy/value evaluation for search: pure neural network, no handcrafted scoring.

AlphaZero principle: the only ground truth is the game outcome (win/loss).
No material counts or positional tables — those would forbid sacrifices that
the network could otherwise discover through self-play.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from my_chess.chess_ai.action_space import canonical_action_id, move_to_action_id
from my_chess.chess_ai.encoding import encode_boards_batch
from my_chess.chess_ai.model import AlphaZeroNet
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move


class TorchPolicyValueEvaluator:
    """Wraps a torch model into the MCTS policy/value callable interface.

    The network sees the board from the side-to-move perspective, so policy
    logits are indexed with canonical action ids and mapped back to absolute
    action ids for MCTS.
    """

    def __init__(self, model: AlphaZeroNet, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        board: Chessboard,
        legal_moves: list[Move] | None = None,
    ) -> tuple[Dict[int, float], float]:
        """Returns legal-action priors and scalar value for the current board."""
        if legal_moves is None:
            legal_moves = board.get_legal_moves()
        if not legal_moves:
            return {}, 0.0
        return self.evaluate_batch([board], [legal_moves])[0]

    @torch.inference_mode()
    def forward_states_numpy(
        self, states_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs a batched forward pass on pre-encoded canonical (N, 15, 10, 9) float32 states.

        Returns contiguous float32 numpy arrays: (policy_logits[N, 8100], values[N]).
        """
        states = torch.from_numpy(states_np).to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        policy_logits, values = self.model(states)
        logits_np = np.ascontiguousarray(
            policy_logits.detach().float().cpu().numpy(), dtype=np.float32
        )
        values_np = np.ascontiguousarray(
            values.detach().float().cpu().reshape(-1).numpy(), dtype=np.float32
        )
        return logits_np, values_np

    @torch.inference_mode()
    def evaluate_batch(
        self,
        boards: Sequence[Chessboard],
        legal_moves_list: Sequence[list[Move] | None] | None = None,
    ) -> List[tuple[Dict[int, float], float]]:
        """Evaluates multiple boards in a single batched forward pass."""
        if not boards:
            return []
        if legal_moves_list is None:
            resolved = [b.get_legal_moves() for b in boards]
        else:
            resolved = [
                b.get_legal_moves() if moves is None else moves
                for b, moves in zip(boards, legal_moves_list)
            ]

        states = torch.from_numpy(encode_boards_batch(boards)).to(self.device)
        policy_logits, values = self.model(states)
        logits_np = policy_logits.float().cpu().numpy()
        values_list = values.float().cpu().reshape(-1).tolist()

        results: List[tuple[Dict[int, float], float]] = []
        for idx, (board, legal_moves) in enumerate(zip(boards, resolved)):
            if not legal_moves:
                results.append(({}, 0.0))
                continue
            is_red = board.is_red_turn
            action_ids = [move_to_action_id(move) for move in legal_moves]
            net_ids = [canonical_action_id(a, is_red) for a in action_ids]
            legal_logits = logits_np[idx, net_ids]
            legal_logits = legal_logits - legal_logits.max()
            probs = np.exp(legal_logits)
            probs /= probs.sum()
            priors = dict(zip(action_ids, probs.tolist()))
            results.append((priors, float(values_list[idx])))
        return results
