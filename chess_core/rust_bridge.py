"""Zero-copy ctypes bridge to the native Rust rules and MCTS engine."""

from __future__ import annotations

import ctypes
import random
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from my_chess.chess_core.chessboard import Chessboard

BOARD_COLS = 9
BOARD_ROWS = 10
NUM_SQUARES = 90
ACTION_SPACE_SIZE = 8100
NUM_CHANNELS = 15
STATE_TENSOR_SIZE = NUM_CHANNELS * BOARD_ROWS * BOARD_COLS  # 1350

# Matches Rust piece codes:
# 1: Rook, 2: Knight, 3: Cannon, 4: Elephant, 5: Mandarin, 6: King, 7: Pawn
FEN_TO_RUST_CODE: Dict[str, int] = {
    "R": 1,
    "N": 2,
    "C": 3,
    "B": 4,
    "A": 5,
    "K": 6,
    "P": 7,
    "r": -1,
    "n": -2,
    "c": -3,
    "b": -4,
    "a": -5,
    "k": -6,
    "p": -7,
}


def _find_dll_path() -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "bin" / "my_chess_rust_core.dll",
        here.parent / "rust_core" / "target" / "release" / "my_chess_rust_core.dll",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_library() -> ctypes.CDLL | None:
    dll_path = _find_dll_path()
    if dll_path is None:
        return None
    try:
        lib = ctypes.CDLL(str(dll_path))
    except OSError:
        return None

    c_i8_p = ctypes.POINTER(ctypes.c_int8)
    c_u16_p = ctypes.POINTER(ctypes.c_uint16)
    c_f32_p = ctypes.POINTER(ctypes.c_float)

    lib.rust_gen_legal_actions.argtypes = [c_i8_p, ctypes.c_bool, c_u16_p]
    lib.rust_gen_legal_actions.restype = ctypes.c_size_t

    lib.rust_encode_board_canonical.argtypes = [c_i8_p, ctypes.c_bool, c_f32_p]
    lib.rust_encode_board_canonical.restype = None

    lib.rust_board_new_initial.argtypes = []
    lib.rust_board_new_initial.restype = ctypes.c_void_p

    lib.rust_board_from_state.argtypes = [
        c_i8_p,
        ctypes.c_bool,
        ctypes.c_size_t,
        ctypes.c_uint16,
    ]
    lib.rust_board_from_state.restype = ctypes.c_void_p

    lib.rust_board_clone.argtypes = [ctypes.c_void_p]
    lib.rust_board_clone.restype = ctypes.c_void_p

    lib.rust_board_free.argtypes = [ctypes.c_void_p]
    lib.rust_board_free.restype = None

    lib.rust_board_push_action.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
    lib.rust_board_push_action.restype = None

    lib.rust_board_is_red_turn.argtypes = [ctypes.c_void_p]
    lib.rust_board_is_red_turn.restype = ctypes.c_bool

    lib.rust_board_get_winner.argtypes = [ctypes.c_void_p]
    lib.rust_board_get_winner.restype = ctypes.c_int32

    lib.rust_board_encode_canonical.argtypes = [ctypes.c_void_p, c_f32_p]
    lib.rust_board_encode_canonical.restype = None

    lib.rust_mcts_new.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint64,
    ]
    lib.rust_mcts_new.restype = ctypes.c_void_p

    lib.rust_mcts_free.argtypes = [ctypes.c_void_p]
    lib.rust_mcts_free.restype = None

    lib.rust_mcts_is_finished.argtypes = [ctypes.c_void_p]
    lib.rust_mcts_is_finished.restype = ctypes.c_bool

    lib.rust_mcts_select_leaves.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        c_f32_p,
    ]
    lib.rust_mcts_select_leaves.restype = ctypes.c_size_t

    lib.rust_mcts_apply_evaluated_leaves.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        c_f32_p,
        c_f32_p,
    ]
    lib.rust_mcts_apply_evaluated_leaves.restype = None

    lib.rust_mcts_get_root_result.argtypes = [
        ctypes.c_void_p,
        c_u16_p,
        c_f32_p,
        c_f32_p,
    ]
    lib.rust_mcts_get_root_result.restype = ctypes.c_size_t

    lib.rust_mcts_write_canonical_policy_target.argtypes = [
        ctypes.c_void_p,
        c_f32_p,
    ]
    lib.rust_mcts_write_canonical_policy_target.restype = None

    return lib


RUST_LIB = _load_library()
RUST_AVAILABLE = RUST_LIB is not None


def board_to_squares_array(board: "Chessboard") -> np.ndarray:
    """Converts a Python Chessboard into a flat int8[90] array for Rust."""
    arr = np.zeros(NUM_SQUARES, dtype=np.int8)
    for piece in board.chessmans_hash.values():
        code = FEN_TO_RUST_CODE.get(piece.fen_char)
        if code is not None:
            arr[piece.row_num * BOARD_COLS + piece.col_num] = code
    return arr


def rust_encode_board_numpy(board: "Chessboard") -> np.ndarray:
    """Encodes a Python Chessboard into a canonical (15, 10, 9) float32 array via Rust."""
    if RUST_LIB is None:
        raise RuntimeError("Rust core library is not loaded")
    squares = board_to_squares_array(board)
    out = np.empty((NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    RUST_LIB.rust_encode_board_canonical(
        squares.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        bool(board.is_red_turn),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def rust_gen_legal_action_ids(board: "Chessboard") -> List[int]:
    """Returns all legal action IDs for `board` computed by the Rust engine."""
    if RUST_LIB is None:
        raise RuntimeError("Rust core library is not loaded")
    squares = board_to_squares_array(board)
    out_actions = np.empty(128, dtype=np.uint16)
    count = RUST_LIB.rust_gen_legal_actions(
        squares.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        bool(board.is_red_turn),
        out_actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out_actions[:count].astype(int).tolist()


class RustBoardHandle:
    """RAII wrapper around a native Rust `FastBoard` pointer."""

    __slots__ = ("_ptr",)

    def __init__(self, ptr: int) -> None:
        if not ptr:
            raise RuntimeError("Null FastBoard pointer")
        self._ptr = ptr

    @classmethod
    def new_initial(cls) -> "RustBoardHandle":
        if RUST_LIB is None:
            raise RuntimeError("Rust core library is not loaded")
        return cls(RUST_LIB.rust_board_new_initial())

    @classmethod
    def from_python_board(cls, board: "Chessboard") -> "RustBoardHandle":
        if RUST_LIB is None:
            raise RuntimeError("Rust core library is not loaded")
        squares = board_to_squares_array(board)
        repeats = int(board.hash_history.get(board.current_hash, 1))
        halfmove_clock = int(getattr(board, "halfmove_clock", 0))
        ptr = RUST_LIB.rust_board_from_state(
            squares.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            bool(board.is_red_turn),
            repeats,
            halfmove_clock,
        )
        return cls(ptr)

    def close(self) -> None:
        if getattr(self, "_ptr", 0) and RUST_LIB is not None:
            RUST_LIB.rust_board_free(self._ptr)
            self._ptr = 0

    def __del__(self) -> None:
        self.close()

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def is_red_turn(self) -> bool:
        return bool(RUST_LIB.rust_board_is_red_turn(self._ptr))

    def push_action(self, action_id: int) -> None:
        RUST_LIB.rust_board_push_action(self._ptr, int(action_id))

    def get_winner_code(self) -> int:
        """Returns 0 (ongoing), 1 (Red wins), -1 (Black wins), or 2 (Draw)."""
        return int(RUST_LIB.rust_board_get_winner(self._ptr))

    def get_winner_label(self) -> str | None:
        code = self.get_winner_code()
        if code == 1:
            return "Red"
        if code == -1:
            return "Black"
        if code == 2:
            return "Draw"
        return None

    def encode_canonical_numpy(self) -> np.ndarray:
        out = np.empty((NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
        RUST_LIB.rust_board_encode_canonical(
            self._ptr,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out


class RustMctsHandle:
    """RAII wrapper around a native Rust `RustMcts` search tree."""

    __slots__ = ("_ptr",)

    def __init__(
        self,
        board_handle: RustBoardHandle,
        num_simulations: int,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        seed: int | None = None,
    ) -> None:
        if RUST_LIB is None:
            raise RuntimeError("Rust core library is not loaded")
        if seed is None:
            seed = random.getrandbits(64) or 1
        self._ptr = RUST_LIB.rust_mcts_new(
            board_handle.ptr,
            int(num_simulations),
            float(c_puct),
            float(dirichlet_alpha),
            float(dirichlet_epsilon),
            int(seed) & 0xFFFFFFFFFFFFFFFF,
        )

    def close(self) -> None:
        if getattr(self, "_ptr", 0) and RUST_LIB is not None:
            RUST_LIB.rust_mcts_free(self._ptr)
            self._ptr = 0

    def __del__(self) -> None:
        self.close()

    @property
    def is_finished(self) -> bool:
        return bool(RUST_LIB.rust_mcts_is_finished(self._ptr))

    def select_leaves_into(self, max_leaves: int, out_states_view: np.ndarray) -> int:
        """Selects up to `max_leaves` leaf states and writes them into `out_states_view`."""
        return int(
            RUST_LIB.rust_mcts_select_leaves(
                self._ptr,
                int(max_leaves),
                out_states_view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        )

    def apply_evaluated_leaves(
        self, num_leaves: int, logits_view: np.ndarray, values_view: np.ndarray
    ) -> None:
        RUST_LIB.rust_mcts_apply_evaluated_leaves(
            self._ptr,
            int(num_leaves),
            logits_view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values_view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def get_root_result(self) -> Tuple[Dict[int, float], float]:
        actions = np.empty(128, dtype=np.uint16)
        probs = np.empty(128, dtype=np.float32)
        root_val = ctypes.c_float(0.0)
        count = int(
            RUST_LIB.rust_mcts_get_root_result(
                self._ptr,
                actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(root_val),
            )
        )
        dist = {int(actions[i]): float(probs[i]) for i in range(count)}
        return dist, float(root_val.value)

    def get_canonical_policy_target_numpy(self) -> np.ndarray:
        target = np.empty(ACTION_SPACE_SIZE, dtype=np.float32)
        RUST_LIB.rust_mcts_write_canonical_policy_target(
            self._ptr,
            target.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return target
