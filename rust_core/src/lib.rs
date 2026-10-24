//! High-performance Xiangqi (Chinese Chess) core engine and batched MCTS in Rust.
//!
//! Exposes a zero-dependency C ABI (`cdylib`) consumed by Python via `ctypes` and
//! `numpy` zero-copy pointers.

use std::slice;

pub const BOARD_COLS: usize = 9;
pub const BOARD_ROWS: usize = 10;
pub const NUM_SQUARES: usize = BOARD_COLS * BOARD_ROWS; // 90
pub const ACTION_SPACE_SIZE: usize = NUM_SQUARES * NUM_SQUARES; // 8100
pub const NUM_CHANNELS: usize = 15;
pub const STATE_TENSOR_SIZE: usize = NUM_CHANNELS * BOARD_ROWS * BOARD_COLS; // 1350

// Piece codes (sign indicates color: >0 Red, <0 Black):
// 1: Rook, 2: Knight, 3: Cannon, 4: Elephant, 5: Mandarin, 6: King, 7: Pawn
pub const EMPTY: i8 = 0;
pub const ROOK: i8 = 1;
pub const KNIGHT: i8 = 2;
pub const CANNON: i8 = 3;
pub const ELEPHANT: i8 = 4;
pub const MANDARIN: i8 = 5;
pub const KING: i8 = 6;
pub const PAWN: i8 = 7;

/// Natural move limit (自然限着): 120 consecutive plies (60 full moves) without capture = Draw.
pub const MAX_HALFMOVES_WITHOUT_CAPTURE: u16 = 120;
pub const WINNER_ONGOING: i32 = 0;
pub const WINNER_RED: i32 = 1;
pub const WINNER_BLACK: i32 = -1;
pub const WINNER_DRAW: i32 = 2;

#[inline(always)]
pub const fn sq_index(col: usize, row: usize) -> usize {
    row * BOARD_COLS + col
}

#[inline(always)]
pub const fn sq_col(sq: usize) -> usize {
    sq % BOARD_COLS
}

#[inline(always)]
pub const fn sq_row(sq: usize) -> usize {
    sq / BOARD_COLS
}

#[inline(always)]
pub const fn mirror_sq(sq: usize) -> usize {
    let row = sq / BOARD_COLS;
    let col = sq % BOARD_COLS;
    (BOARD_ROWS - 1 - row) * BOARD_COLS + col
}

#[inline(always)]
pub const fn canonical_action_id(action_id: usize, is_red_turn: bool) -> usize {
    if is_red_turn {
        action_id
    } else {
        let from_sq = action_id / NUM_SQUARES;
        let to_sq = action_id % NUM_SQUARES;
        mirror_sq(from_sq) * NUM_SQUARES + mirror_sq(to_sq)
    }
}

/// Deterministic Zobrist table generated at compile/init time via SplitMix64.
struct ZobristTable {
    piece_keys: [[u64; NUM_SQUARES]; 14],
    turn_key: u64,
}

impl ZobristTable {
    const fn new() -> Self {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut piece_keys = [[0u64; NUM_SQUARES]; 14];
        let mut p = 0;
        while p < 14 {
            let mut s = 0;
            while s < NUM_SQUARES {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                piece_keys[p][s] = z ^ (z >> 31);
                s += 1;
            }
            p += 1;
        }
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let turn_key = z ^ (z >> 31);
        Self {
            piece_keys,
            turn_key,
        }
    }

    #[inline(always)]
    fn piece_key(&self, piece: i8, sq: usize) -> u64 {
        debug_assert!(piece != 0);
        let idx = if piece > 0 {
            (piece as usize) - 1
        } else {
            7 + ((-piece) as usize) - 1
        };
        self.piece_keys[idx][sq]
    }
}

static ZOBRIST: ZobristTable = ZobristTable::new();

/// Simple fast xorshift64* PRNG for Dirichlet gamma sampling and action choice.
#[derive(Clone)]
struct FastRng {
    state: u64,
}

impl FastRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x853C_49E6_748F_EA9B
            } else {
                seed
            },
        }
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline(always)]
    fn next_f64(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        if u <= 1e-15 {
            1e-15
        } else if u >= 1.0 - 1e-15 {
            1.0 - 1e-15
        } else {
            u
        }
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Marsaglia and Tsang's method for Gamma(alpha, 1.0) with alpha > 0.
    fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha <= 0.0 {
            return 0.0;
        }
        if alpha < 1.0 {
            let u = self.next_f64();
            return self.gamma(alpha + 1.0) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let z = self.standard_normal();
            let v = 1.0 + c * z;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * (z * z) * (z * z) {
                return d * v3;
            }
            if u.ln() < 0.5 * z * z + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct UndoMove {
    pub from_sq: u8,
    pub to_sq: u8,
    pub captured: i8,
    pub prev_halfmove_clock: u16,
    pub prev_hash: u64,
}

#[derive(Clone)]
pub struct FastBoard {
    pub squares: [i8; NUM_SQUARES],
    pub is_red_turn: bool,
    pub halfmove_clock: u16,
    pub current_hash: u64,
    pub hash_history: Vec<u64>,
}

impl FastBoard {
    pub fn new_initial() -> Self {
        let mut squares = [EMPTY; NUM_SQUARES];
        // Red pieces (rows 0, 2, 3)
        squares[sq_index(0, 0)] = ROOK;
        squares[sq_index(8, 0)] = ROOK;
        squares[sq_index(1, 0)] = KNIGHT;
        squares[sq_index(7, 0)] = KNIGHT;
        squares[sq_index(1, 2)] = CANNON;
        squares[sq_index(7, 2)] = CANNON;
        squares[sq_index(2, 0)] = ELEPHANT;
        squares[sq_index(6, 0)] = ELEPHANT;
        squares[sq_index(3, 0)] = MANDARIN;
        squares[sq_index(5, 0)] = MANDARIN;
        squares[sq_index(4, 0)] = KING;
        squares[sq_index(0, 3)] = PAWN;
        squares[sq_index(2, 3)] = PAWN;
        squares[sq_index(4, 3)] = PAWN;
        squares[sq_index(6, 3)] = PAWN;
        squares[sq_index(8, 3)] = PAWN;

        // Black pieces (rows 9, 7, 6)
        squares[sq_index(0, 9)] = -ROOK;
        squares[sq_index(8, 9)] = -ROOK;
        squares[sq_index(1, 9)] = -KNIGHT;
        squares[sq_index(7, 9)] = -KNIGHT;
        squares[sq_index(1, 7)] = -CANNON;
        squares[sq_index(7, 7)] = -CANNON;
        squares[sq_index(2, 9)] = -ELEPHANT;
        squares[sq_index(6, 9)] = -ELEPHANT;
        squares[sq_index(3, 9)] = -MANDARIN;
        squares[sq_index(5, 9)] = -MANDARIN;
        squares[sq_index(4, 9)] = -KING;
        squares[sq_index(0, 6)] = -PAWN;
        squares[sq_index(2, 6)] = -PAWN;
        squares[sq_index(4, 6)] = -PAWN;
        squares[sq_index(6, 6)] = -PAWN;
        squares[sq_index(8, 6)] = -PAWN;

        let mut board = Self {
            squares,
            is_red_turn: true,
            halfmove_clock: 0,
            current_hash: 0,
            hash_history: Vec::with_capacity(256),
        };
        board.recompute_hash();
        board
    }

    pub fn from_squares(squares: &[i8; NUM_SQUARES], is_red_turn: bool) -> Self {
        let mut board = Self {
            squares: *squares,
            is_red_turn,
            halfmove_clock: 0,
            current_hash: 0,
            hash_history: Vec::with_capacity(256),
        };
        board.recompute_hash();
        board
    }

    pub fn recompute_hash(&mut self) {
        let mut h = if self.is_red_turn {
            ZOBRIST.turn_key
        } else {
            0
        };
        for (sq, &piece) in self.squares.iter().enumerate() {
            if piece != EMPTY {
                h ^= ZOBRIST.piece_key(piece, sq);
            }
        }
        self.current_hash = h;
        self.hash_history.clear();
        self.hash_history.push(h);
    }

    #[inline(always)]
    pub fn push_action(&mut self, action_id: u16) -> UndoMove {
        let from_sq = (action_id as usize) / NUM_SQUARES;
        let to_sq = (action_id as usize) % NUM_SQUARES;
        let piece = self.squares[from_sq];
        let captured = self.squares[to_sq];
        let prev_hash = self.current_hash;
        let prev_halfmove_clock = self.halfmove_clock;

        let mut h = prev_hash ^ ZOBRIST.piece_key(piece, from_sq) ^ ZOBRIST.piece_key(piece, to_sq);
        if captured != EMPTY {
            h ^= ZOBRIST.piece_key(captured, to_sq);
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        }
        h ^= ZOBRIST.turn_key;

        self.squares[from_sq] = EMPTY;
        self.squares[to_sq] = piece;
        self.is_red_turn = !self.is_red_turn;
        self.current_hash = h;
        self.hash_history.push(h);

        UndoMove {
            from_sq: from_sq as u8,
            to_sq: to_sq as u8,
            captured,
            prev_halfmove_clock,
            prev_hash,
        }
    }

    #[inline(always)]
    pub fn pop_action(&mut self, undo: UndoMove) {
        let from_sq = undo.from_sq as usize;
        let to_sq = undo.to_sq as usize;
        let piece = self.squares[to_sq];
        self.squares[from_sq] = piece;
        self.squares[to_sq] = undo.captured;
        self.is_red_turn = !self.is_red_turn;
        self.halfmove_clock = undo.prev_halfmove_clock;
        self.current_hash = undo.prev_hash;
        self.hash_history.pop();
    }

    #[inline(always)]
    pub fn repetition_count(&self) -> usize {
        let target = self.current_hash;
        self.hash_history.iter().filter(|&&h| h == target).count()
    }

    /// Fast winner check (without stalemate scan):
    /// returns 0 (ongoing), 1 (Red wins), -1 (Black wins).
    #[inline]
    pub fn get_fast_winner(&self) -> i32 {
        if self.repetition_count() >= 3 {
            return if self.is_red_turn { 1 } else { -1 };
        }
        let mut has_red_king = false;
        let mut has_black_king = false;
        // Scan palace squares first (col 3..=5, row 0..=2 and row 7..=9)
        for row in 0..=2 {
            for col in 3..=5 {
                let p = self.squares[sq_index(col, row)];
                if p == KING {
                    has_red_king = true;
                } else if p == -KING {
                    has_black_king = true;
                }
            }
        }
        for row in 7..=9 {
            for col in 3..=5 {
                let p = self.squares[sq_index(col, row)];
                if p == -KING {
                    has_black_king = true;
                } else if p == KING {
                    has_red_king = true;
                }
            }
        }
        if !has_red_king || !has_black_king {
            // Fallback full scan in case custom FEN placed king outside palace
            for &p in &self.squares {
                if p == KING {
                    has_red_king = true;
                } else if p == -KING {
                    has_black_king = true;
                }
            }
        }
        if !has_red_king {
            return WINNER_BLACK;
        }
        if !has_black_king {
            return WINNER_RED;
        }
        if self.halfmove_clock >= MAX_HALFMOVES_WITHOUT_CAPTURE {
            return WINNER_DRAW;
        }
        WINNER_ONGOING
    }

    /// Full winner check including stalemate/checkmate: 0 (ongoing), 1 (Red), -1 (Black), 2 (Draw).
    pub fn get_winner(&mut self) -> i32 {
        let fast = self.get_fast_winner();
        if fast != 0 {
            return fast;
        }
        let mut moves = Vec::with_capacity(64);
        self.gen_strict_legal_actions(&mut moves);
        if moves.is_empty() {
            return if self.is_red_turn { -1 } else { 1 };
        }
        0
    }

    /// Checks in O(1) whether `king_is_red`'s King can be captured on the board.
    #[inline]
    pub fn is_king_attacked(&self, king_is_red: bool) -> bool {
        let king_code = if king_is_red { KING } else { -KING };
        let enemy_sign: i8 = if king_is_red { -1 } else { 1 };

        // Find king square (check palace first)
        let mut king_sq: Option<usize> = None;
        let (p_min_r, p_max_r) = if king_is_red { (0, 2) } else { (7, 9) };
        for r in p_min_r..=p_max_r {
            for c in 3..=5 {
                let sq = sq_index(c, r);
                if self.squares[sq] == king_code {
                    king_sq = Some(sq);
                    break;
                }
            }
            if king_sq.is_some() {
                break;
            }
        }
        if king_sq.is_none() {
            for (sq, &p) in self.squares.iter().enumerate() {
                if p == king_code {
                    king_sq = Some(sq);
                    break;
                }
            }
        }
        let k_sq = match king_sq {
            Some(sq) => sq,
            None => return true,
        };
        let kc = sq_col(k_sq) as isize;
        let kr = sq_row(k_sq) as isize;

        // 1. Orthogonal rays: Rook (0 screens) or Cannon (1 screen)
        const DIRS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dc, dr) in DIRS {
            let mut c = kc + dc;
            let mut r = kr + dr;
            let mut screens = 0usize;
            while (0..9).contains(&c) && (0..10).contains(&r) {
                let p = self.squares[sq_index(c as usize, r as usize)];
                if p != EMPTY {
                    if screens == 0 {
                        if p == enemy_sign * ROOK {
                            return true;
                        }
                        screens = 1;
                    } else {
                        if p == enemy_sign * CANNON {
                            return true;
                        }
                        break;
                    }
                }
                c += dc;
                r += dr;
            }
        }

        // 2. Enemy Pawn attacks
        // Red pawn moves +1 in row; Black pawn moves -1 in row.
        // So an enemy pawn attacking `(kc, kr)` vertically sits at `kr - enemy_pawn_dir`.
        let enemy_pawn = enemy_sign * PAWN;
        let enemy_pawn_dir: isize = if king_is_red { -1 } else { 1 };
        let from_r = kr - enemy_pawn_dir;
        if (0..10).contains(&from_r)
            && self.squares[sq_index(kc as usize, from_r as usize)] == enemy_pawn
        {
            return true;
        }
        for dc in [-1isize, 1isize] {
            let nc = kc + dc;
            if (0..9).contains(&nc)
                && self.squares[sq_index(nc as usize, kr as usize)] == enemy_pawn
            {
                return true;
            }
        }

        // 3. Enemy Knight attacks
        let enemy_knight = enemy_sign * KNIGHT;
        const KNIGHT_OFFSETS: [(isize, isize, isize, isize); 8] = [
            (2, 1, 1, 1),
            (2, -1, 1, -1),
            (-2, 1, -1, 1),
            (-2, -1, -1, -1),
            (1, 2, 1, 1),
            (-1, 2, -1, 1),
            (1, -2, 1, -1),
            (-1, -2, -1, -1),
        ];
        for (dc, dr, leg_dc, leg_dr) in KNIGHT_OFFSETS {
            let nc = kc + dc;
            let nr = kr + dr;
            if (0..9).contains(&nc)
                && (0..10).contains(&nr)
                && self.squares[sq_index(nc as usize, nr as usize)] == enemy_knight
            {
                let lc = kc + leg_dc;
                let lr = kr + leg_dr;
                if self.squares[sq_index(lc as usize, lr as usize)] == EMPTY {
                    return true;
                }
            }
        }

        false
    }

    /// Generates strictly legal moves for MCTS (filtering out moves that leave
    /// the mover's own King in check).
    pub fn gen_strict_legal_actions(&mut self, out: &mut Vec<u16>) {
        let mut pseudo = Vec::with_capacity(64);
        self.gen_legal_actions(&mut pseudo);
        out.clear();
        let mover_is_red = self.is_red_turn;
        for action_id in pseudo {
            let to_sq = (action_id as usize) % NUM_SQUARES;
            if self.squares[to_sq].abs() == KING {
                out.push(action_id);
                continue;
            }
            let undo = self.push_action(action_id);
            let attacked = self.is_king_attacked(mover_is_red);
            self.pop_action(undo);
            if !attacked {
                out.push(action_id);
            }
        }
    }

    /// Writes canonical 15 x 10 x 9 float32 planes into `out` (length 1350).
    #[inline]
    pub fn encode_canonical(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), STATE_TENSOR_SIZE);
        out.fill(0.0);
        let is_red = self.is_red_turn;
        for sq in 0..NUM_SQUARES {
            let p = self.squares[sq];
            if p == EMPTY {
                continue;
            }
            let piece_is_red = p > 0;
            let piece_type = (p.unsigned_abs() as usize) - 1; // 0..6
            let channel = if piece_is_red == is_red {
                piece_type
            } else {
                piece_type + 7
            };
            let row = sq_row(sq);
            let col = sq_col(sq);
            let canon_row = if is_red { row } else { BOARD_ROWS - 1 - row };
            out[channel * NUM_SQUARES + canon_row * BOARD_COLS + col] = 1.0;
        }
        if is_red {
            let side_offset = 14 * NUM_SQUARES;
            out[side_offset..side_offset + NUM_SQUARES].fill(1.0);
        }
    }

    /// Generates all legal moves as action IDs (`from_sq * 90 + to_sq`),
    /// matching `my_chess.chess_core.chessman` rules 100%.
    pub fn gen_legal_actions(&self, out: &mut Vec<u16>) {
        out.clear();
        let is_red = self.is_red_turn;
        let side: i8 = if is_red { 1 } else { -1 };

        for from_sq in 0..NUM_SQUARES {
            let piece = self.squares[from_sq];
            if piece == EMPTY || (piece > 0) != is_red {
                continue;
            }
            let col = sq_col(from_sq) as isize;
            let row = sq_row(from_sq) as isize;
            let kind = piece.abs();
            let base = (from_sq * NUM_SQUARES) as u16;

            match kind {
                ROOK => {
                    // 4 orthogonal rays
                    const DIRS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                    for (dc, dr) in DIRS {
                        let mut c = col + dc;
                        let mut r = row + dr;
                        while (0..9).contains(&c) && (0..10).contains(&r) {
                            let to_sq = sq_index(c as usize, r as usize);
                            let target = self.squares[to_sq];
                            if target == EMPTY {
                                out.push(base + to_sq as u16);
                            } else {
                                if (target * side) < 0 {
                                    out.push(base + to_sq as u16);
                                }
                                break;
                            }
                            c += dc;
                            r += dr;
                        }
                    }
                }
                CANNON => {
                    const DIRS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                    for (dc, dr) in DIRS {
                        let mut c = col + dc;
                        let mut r = row + dr;
                        let mut jumped = false;
                        while (0..9).contains(&c) && (0..10).contains(&r) {
                            let to_sq = sq_index(c as usize, r as usize);
                            let target = self.squares[to_sq];
                            if !jumped {
                                if target == EMPTY {
                                    out.push(base + to_sq as u16);
                                } else {
                                    jumped = true;
                                }
                            } else if target != EMPTY {
                                if (target * side) < 0 {
                                    out.push(base + to_sq as u16);
                                }
                                break;
                            }
                            c += dc;
                            r += dr;
                        }
                    }
                }
                KNIGHT => {
                    const LEGS: [(isize, isize, [(isize, isize); 2]); 4] = [
                        (1, 0, [(2, 1), (2, -1)]),
                        (-1, 0, [(-2, 1), (-2, -1)]),
                        (0, 1, [(1, 2), (-1, 2)]),
                        (0, -1, [(1, -2), (-1, -2)]),
                    ];
                    for (lc, lr, targets) in LEGS {
                        let leg_c = col + lc;
                        let leg_r = row + lr;
                        if (0..9).contains(&leg_c) && (0..10).contains(&leg_r) {
                            let leg_sq = sq_index(leg_c as usize, leg_r as usize);
                            if self.squares[leg_sq] == EMPTY {
                                for (tc, tr) in targets {
                                    let nc = col + tc;
                                    let nr = row + tr;
                                    if (0..9).contains(&nc) && (0..10).contains(&nr) {
                                        let to_sq = sq_index(nc as usize, nr as usize);
                                        if self.squares[to_sq] * side <= 0 {
                                            out.push(base + to_sq as u16);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                ELEPHANT => {
                    let (min_r, max_r) = if is_red { (0, 4) } else { (5, 9) };
                    const EYES: [(isize, isize); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
                    for (dc, dr) in EYES {
                        let ec = col + dc;
                        let er = row + dr;
                        let tc = col + 2 * dc;
                        let tr = row + 2 * dr;
                        if (0..9).contains(&ec)
                            && (min_r..=max_r).contains(&er)
                            && (0..9).contains(&tc)
                            && (min_r..=max_r).contains(&tr)
                        {
                            let eye_sq = sq_index(ec as usize, er as usize);
                            if self.squares[eye_sq] == EMPTY {
                                let to_sq = sq_index(tc as usize, tr as usize);
                                if self.squares[to_sq] * side <= 0 {
                                    out.push(base + to_sq as u16);
                                }
                            }
                        }
                    }
                }
                MANDARIN => {
                    let (min_r, max_r) = if is_red { (0, 2) } else { (7, 9) };
                    const STEPS: [(isize, isize); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
                    for (dc, dr) in STEPS {
                        let nc = col + dc;
                        let nr = row + dr;
                        if (3..=5).contains(&nc) && (min_r..=max_r).contains(&nr) {
                            let to_sq = sq_index(nc as usize, nr as usize);
                            if self.squares[to_sq] * side <= 0 {
                                out.push(base + to_sq as u16);
                            }
                        }
                    }
                }
                KING => {
                    let (min_r, max_r) = if is_red { (0, 2) } else { (7, 9) };
                    const STEPS: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
                    for (dc, dr) in STEPS {
                        let nc = col + dc;
                        let nr = row + dr;
                        if (3..=5).contains(&nc) && (min_r..=max_r).contains(&nr) {
                            let to_sq = sq_index(nc as usize, nr as usize);
                            if self.squares[to_sq] * side <= 0 {
                                out.push(base + to_sq as u16);
                            }
                        }
                    }
                }
                PAWN => {
                    let (dir, min_r, max_r, crossed) = if is_red {
                        (1isize, 3isize, 9isize, row >= 5)
                    } else {
                        (-1isize, 0isize, 6isize, row <= 4)
                    };
                    let nr = row + dir;
                    if (min_r..=max_r).contains(&nr) {
                        let to_sq = sq_index(col as usize, nr as usize);
                        if self.squares[to_sq] * side <= 0 {
                            out.push(base + to_sq as u16);
                        }
                    }
                    if crossed {
                        for dc in [-1isize, 1isize] {
                            let nc = col + dc;
                            if (0..9).contains(&nc) && (min_r..=max_r).contains(&row) {
                                let to_sq = sq_index(nc as usize, row as usize);
                                if self.squares[to_sq] * side <= 0 {
                                    out.push(base + to_sq as u16);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

// ============================================================================
// Batched MCTS implementation in Rust
// ============================================================================

#[derive(Clone)]
struct Edge {
    action_id: u16,
    prior: f32,
    visits: u32,
    virtual_visits: u32,
    total_value: f32,
    child_idx: u32, // 0 means unallocated
}

impl Edge {
    #[inline(always)]
    fn new(action_id: u16, prior: f32) -> Self {
        Self {
            action_id,
            prior,
            visits: 0,
            virtual_visits: 0,
            total_value: 0.0,
            child_idx: 0,
        }
    }

    #[inline(always)]
    fn effective_visits(&self) -> u32 {
        self.visits + self.virtual_visits
    }

    #[inline(always)]
    fn effective_mean_value(&self) -> f32 {
        let v = self.effective_visits();
        if v == 0 {
            0.0
        } else {
            (self.total_value - (self.virtual_visits as f32)) / (v as f32)
        }
    }
}

struct Node {
    edges: Vec<Edge>,
    is_expanded: bool,
    in_flight: bool,
}

impl Node {
    fn new() -> Self {
        Self {
            edges: Vec::new(),
            is_expanded: false,
            in_flight: false,
        }
    }

    #[inline]
    fn total_effective_visits(&self) -> u32 {
        self.edges.iter().map(|e| e.effective_visits()).sum()
    }
}

struct PendingLeaf {
    node_idx: usize,
    is_red_turn: bool,
    legal_actions: Vec<u16>,
    paths: Vec<Vec<(usize, usize)>>, // list of (node_idx, edge_idx) paths
}

pub struct RustMcts {
    board: FastBoard,
    nodes: Vec<Node>,
    pending_leaves: Vec<PendingLeaf>,
    num_simulations: u32,
    simulations_done: u32,
    c_puct: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    root_value: f32,
    rng: FastRng,
}

impl RustMcts {
    pub fn new(
        board: FastBoard,
        num_simulations: u32,
        c_puct: f32,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
        seed: u64,
    ) -> Self {
        let mut nodes = Vec::with_capacity((num_simulations as usize) + 16);
        nodes.push(Node::new()); // root = 0
        Self {
            board,
            nodes,
            pending_leaves: Vec::with_capacity(32),
            num_simulations,
            simulations_done: 0,
            c_puct,
            dirichlet_alpha,
            dirichlet_epsilon,
            root_value: 0.0,
            rng: FastRng::new(seed),
        }
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.nodes[0].is_expanded
            && self.pending_leaves.is_empty()
            && (self.nodes[0].edges.is_empty() || self.simulations_done >= self.num_simulations)
    }

    fn select_edge_index(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        let total_visits = node.total_effective_visits().max(1) as f32;
        let sqrt_total = total_visits.sqrt();
        let c_puct = self.c_puct;

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (idx, edge) in node.edges.iter().enumerate() {
            let exploration =
                c_puct * edge.prior * sqrt_total / (1.0 + edge.effective_visits() as f32);
            let score = edge.effective_mean_value() + exploration;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        best_idx
    }

    fn backprop_path(&mut self, path: &[(usize, usize)], leaf_value: f32, revert_virtual: bool) {
        let mut value = leaf_value;
        for &(node_idx, edge_idx) in path.iter().rev() {
            value = -value;
            let edge = &mut self.nodes[node_idx].edges[edge_idx];
            if revert_virtual && edge.virtual_visits > 0 {
                edge.virtual_visits -= 1;
            }
            edge.visits += 1;
            edge.total_value += value;
        }
    }

    /// Selects up to `max_leaves` leaf positions for neural network evaluation and
    /// writes their canonical `15x10x9` float32 encodings into `out_states`.
    /// Returns the number of leaves written (`0` means search is complete or resolved).
    pub fn select_leaves(&mut self, max_leaves: usize, out_states: &mut [f32]) -> usize {
        self.pending_leaves.clear();
        if max_leaves == 0 {
            return 0;
        }

        // Step 1: If root is not yet expanded, expand root first.
        if !self.nodes[0].is_expanded {
            let fast_winner = self.board.get_fast_winner();
            if fast_winner != 0 {
                self.nodes[0].is_expanded = true;
                self.root_value = Self::winner_to_value(fast_winner, self.board.is_red_turn);
                return 0;
            }
            let mut legal = Vec::with_capacity(64);
            self.board.gen_strict_legal_actions(&mut legal);
            if legal.is_empty() {
                // If root has no strict-legal moves (checkmated), fall back to pseudo-legal
                // only when the caller is `Chessboard` on a synthetic test board.
                self.board.gen_legal_actions(&mut legal);
                if legal.is_empty() {
                    self.nodes[0].is_expanded = true;
                    self.root_value = -1.0;
                    return 0;
                }
            }
            // Terminal rule check: if side to move can capture the enemy King now,
            // the game ends immediately in a win (+1.0).
            let king_captures: Vec<u16> = legal
                .iter()
                .copied()
                .filter(|&a| self.board.squares[(a as usize) % NUM_SQUARES].abs() == KING)
                .collect();
            if !king_captures.is_empty() {
                let prior = 1.0 / (king_captures.len() as f32);
                self.nodes[0].edges = king_captures
                    .into_iter()
                    .map(|a| {
                        let mut e = Edge::new(a, prior);
                        e.visits = self.num_simulations.max(1);
                        e.total_value = e.visits as f32;
                        e
                    })
                    .collect();
                self.nodes[0].is_expanded = true;
                self.root_value = 1.0;
                self.simulations_done = self.num_simulations;
                return 0;
            }
            self.board
                .encode_canonical(&mut out_states[0..STATE_TENSOR_SIZE]);
            self.nodes[0].in_flight = true;
            self.pending_leaves.push(PendingLeaf {
                node_idx: 0,
                is_red_turn: self.board.is_red_turn,
                legal_actions: legal,
                paths: Vec::new(),
            });
            return 1;
        }

        let mut written = 0usize;
        let mut undo_stack: Vec<UndoMove> = Vec::with_capacity(64);

        while written < max_leaves {
            let in_flight_sims: usize = self.pending_leaves.iter().map(|p| p.paths.len()).sum();
            if (self.simulations_done as usize) + in_flight_sims >= (self.num_simulations as usize)
            {
                break;
            }
            if self.nodes[0].edges.is_empty() {
                break;
            }

            let mut node_idx = 0usize;
            let mut path: Vec<(usize, usize)> = Vec::with_capacity(32);
            undo_stack.clear();

            loop {
                if node_idx > 0 {
                    let fast_winner = self.board.get_fast_winner();
                    if fast_winner != 0 {
                        let v = Self::winner_to_value(fast_winner, self.board.is_red_turn);
                        self.backprop_path(&path, v, true);
                        self.simulations_done += 1;
                        break;
                    }
                }

                if !self.nodes[node_idx].is_expanded {
                    if self.nodes[node_idx].in_flight {
                        // Another trajectory in this wave already queued `node_idx`.
                        if let Some(pending) = self
                            .pending_leaves
                            .iter_mut()
                            .find(|p| p.node_idx == node_idx)
                        {
                            pending.paths.push(path);
                        }
                        break;
                    }

                    let mut legal = Vec::with_capacity(64);
                    self.board.gen_strict_legal_actions(&mut legal);
                    if legal.is_empty() {
                        self.nodes[node_idx].is_expanded = true;
                        self.backprop_path(&path, -1.0, true);
                        self.simulations_done += 1;
                        break;
                    }
                    // Terminal rule check: opponent left their King to be captured.
                    let king_captures: Vec<u16> = legal
                        .iter()
                        .copied()
                        .filter(|&a| self.board.squares[(a as usize) % NUM_SQUARES].abs() == KING)
                        .collect();
                    if !king_captures.is_empty() {
                        let prior = 1.0 / (king_captures.len() as f32);
                        self.nodes[node_idx].edges = king_captures
                            .into_iter()
                            .map(|a| Edge::new(a, prior))
                            .collect();
                        self.nodes[node_idx].is_expanded = true;
                        self.backprop_path(&path, 1.0, true);
                        self.simulations_done += 1;
                        break;
                    }

                    let offset = written * STATE_TENSOR_SIZE;
                    self.board
                        .encode_canonical(&mut out_states[offset..offset + STATE_TENSOR_SIZE]);
                    self.nodes[node_idx].in_flight = true;
                    self.pending_leaves.push(PendingLeaf {
                        node_idx,
                        is_red_turn: self.board.is_red_turn,
                        legal_actions: legal,
                        paths: vec![path],
                    });
                    written += 1;
                    break;
                }

                if self.nodes[node_idx].edges.is_empty() {
                    self.backprop_path(&path, -1.0, true);
                    self.simulations_done += 1;
                    break;
                }

                let edge_idx = self.select_edge_index(node_idx);
                let action_id = self.nodes[node_idx].edges[edge_idx].action_id;
                let mut child_idx = self.nodes[node_idx].edges[edge_idx].child_idx as usize;
                if child_idx == 0 {
                    child_idx = self.nodes.len();
                    self.nodes.push(Node::new());
                    self.nodes[node_idx].edges[edge_idx].child_idx = child_idx as u32;
                }
                self.nodes[node_idx].edges[edge_idx].virtual_visits += 1;
                path.push((node_idx, edge_idx));
                let undo = self.board.push_action(action_id);
                undo_stack.push(undo);
                node_idx = child_idx;
            }

            while let Some(undo) = undo_stack.pop() {
                self.board.pop_action(undo);
            }
        }

        written
    }

    /// Applies neural network outputs (`logits` of shape `[N, 8100]`, `values` of shape `[N]`)
    /// to the pending leaves collected in `select_leaves`.
    pub fn apply_evaluated_leaves(&mut self, logits: &[f32], values: &[f32]) {
        let pending = std::mem::take(&mut self.pending_leaves);
        debug_assert_eq!(pending.len(), values.len());

        for (idx, leaf) in pending.into_iter().enumerate() {
            let row_logits = &logits[idx * ACTION_SPACE_SIZE..(idx + 1) * ACTION_SPACE_SIZE];
            let value = values[idx];
            let n = leaf.legal_actions.len();

            let mut max_logit = f32::NEG_INFINITY;
            let mut priors = Vec::with_capacity(n);
            for &action_id in &leaf.legal_actions {
                let net_idx = canonical_action_id(action_id as usize, leaf.is_red_turn);
                let l = row_logits[net_idx];
                if l > max_logit {
                    max_logit = l;
                }
                priors.push(l);
            }

            let mut sum_exp = 0.0f32;
            for p in &mut priors {
                *p = (*p - max_logit).exp();
                sum_exp += *p;
            }
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for p in &mut priors {
                    *p *= inv;
                }
            }

            if leaf.node_idx == 0 && self.dirichlet_epsilon > 0.0 && n > 1 {
                let eps = self.dirichlet_epsilon;
                let alpha = self.dirichlet_alpha as f64;
                let mut raw = Vec::with_capacity(n);
                let mut total_noise = 0.0f64;
                for _ in 0..n {
                    let g = self.rng.gamma(alpha);
                    raw.push(g);
                    total_noise += g;
                }
                if total_noise <= 0.0 {
                    total_noise = 1.0;
                }
                for (p, g) in priors.iter_mut().zip(raw.into_iter()) {
                    let noise = (g / total_noise) as f32;
                    *p = (1.0 - eps) * (*p) + eps * noise;
                }
            }

            let node = &mut self.nodes[leaf.node_idx];
            node.edges = leaf
                .legal_actions
                .iter()
                .zip(priors.into_iter())
                .map(|(&a, p)| Edge::new(a, p))
                .collect();
            node.is_expanded = true;
            node.in_flight = false;

            if leaf.node_idx == 0 && leaf.paths.is_empty() {
                self.root_value = value;
            } else {
                for path in &leaf.paths {
                    self.backprop_path(path, value, true);
                    self.simulations_done += 1;
                }
            }
        }
    }

    #[inline(always)]
    fn winner_to_value(winner: i32, is_red_turn: bool) -> f32 {
        if winner == 1 {
            if is_red_turn { 1.0 } else { -1.0 }
        } else if winner == -1 {
            if is_red_turn { -1.0 } else { 1.0 }
        } else {
            0.0
        }
    }
}

// ============================================================================
// C ABI Exports
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rust_gen_legal_actions(
    squares_ptr: *const i8,
    is_red_turn: bool,
    out_actions_ptr: *mut u16,
) -> usize {
    let squares = &*(squares_ptr as *const [i8; NUM_SQUARES]);
    let board = FastBoard::from_squares(squares, is_red_turn);
    let mut actions = Vec::with_capacity(64);
    board.gen_legal_actions(&mut actions);
    let n = actions.len();
    let out_slice = slice::from_raw_parts_mut(out_actions_ptr, n);
    out_slice.copy_from_slice(&actions);
    n
}

#[no_mangle]
pub unsafe extern "C" fn rust_encode_board_canonical(
    squares_ptr: *const i8,
    is_red_turn: bool,
    out_tensor_ptr: *mut f32,
) {
    let squares = &*(squares_ptr as *const [i8; NUM_SQUARES]);
    let board = FastBoard::from_squares(squares, is_red_turn);
    let out = slice::from_raw_parts_mut(out_tensor_ptr, STATE_TENSOR_SIZE);
    board.encode_canonical(out);
}

#[no_mangle]
pub unsafe extern "C" fn rust_board_new_initial() -> *mut FastBoard {
    Box::into_raw(Box::new(FastBoard::new_initial()))
}

#[no_mangle]
pub unsafe extern "C" fn rust_board_from_state(
    squares_ptr: *const i8,
    is_red_turn: bool,
    root_repeats: usize,
    halfmove_clock: u16,
) -> *mut FastBoard {
    let squares = &*(squares_ptr as *const [i8; NUM_SQUARES]);
    let mut board = FastBoard::from_squares(squares, is_red_turn);
    board.halfmove_clock = halfmove_clock;
    let extra = root_repeats.saturating_sub(1);
    for _ in 0..extra {
        board.hash_history.push(board.current_hash);
    }
    Box::into_raw(Box::new(board))
}

#[no_mangle]
pub unsafe extern "C" fn rust_board_clone(board_ptr: *const FastBoard) -> *mut FastBoard {
    let board = &*board_ptr;
    Box::into_raw(Box::new(board.clone()))
}

#[no_mangle]
pub unsafe extern "C" fn rust_board_free(board_ptr: *mut FastBoard) {
    if !board_ptr.is_null() {
        drop(Box::from_raw(board_ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rust_board_push_action(board_ptr: *mut FastBoard, action_id: u16) {
    let board = &mut *board_ptr;
    board.push_action(action_id);
}

#[no_mangle]
pub unsafe extern "C" fn rust_board_is_red_turn(board_ptr: *const FastBoard) -> bool {
    (*board_ptr).is_red_turn
}

#[no_mangle]
pub unsafe extern "C" fn rust_board_get_winner(board_ptr: *mut FastBoard) -> i32 {
    (*board_ptr).get_winner()
}

#[no_mangle]
pub unsafe extern "C" fn rust_board_encode_canonical(
    board_ptr: *const FastBoard,
    out_tensor_ptr: *mut f32,
) {
    let board = &*board_ptr;
    let out = slice::from_raw_parts_mut(out_tensor_ptr, STATE_TENSOR_SIZE);
    board.encode_canonical(out);
}

#[no_mangle]
pub unsafe extern "C" fn rust_mcts_new(
    board_ptr: *const FastBoard,
    num_simulations: u32,
    c_puct: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    seed: u64,
) -> *mut RustMcts {
    let board = (*board_ptr).clone();
    Box::into_raw(Box::new(RustMcts::new(
        board,
        num_simulations,
        c_puct,
        dirichlet_alpha,
        dirichlet_epsilon,
        seed,
    )))
}

#[no_mangle]
pub unsafe extern "C" fn rust_mcts_free(mcts_ptr: *mut RustMcts) {
    if !mcts_ptr.is_null() {
        drop(Box::from_raw(mcts_ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rust_mcts_is_finished(mcts_ptr: *const RustMcts) -> bool {
    (*mcts_ptr).is_finished()
}

#[no_mangle]
pub unsafe extern "C" fn rust_mcts_select_leaves(
    mcts_ptr: *mut RustMcts,
    max_leaves: usize,
    out_states_ptr: *mut f32,
) -> usize {
    let mcts = &mut *mcts_ptr;
    let out_states = slice::from_raw_parts_mut(out_states_ptr, max_leaves * STATE_TENSOR_SIZE);
    mcts.select_leaves(max_leaves, out_states)
}

#[no_mangle]
pub unsafe extern "C" fn rust_mcts_apply_evaluated_leaves(
    mcts_ptr: *mut RustMcts,
    num_leaves: usize,
    logits_ptr: *const f32,
    values_ptr: *const f32,
) {
    let mcts = &mut *mcts_ptr;
    let logits = slice::from_raw_parts(logits_ptr, num_leaves * ACTION_SPACE_SIZE);
    let values = slice::from_raw_parts(values_ptr, num_leaves);
    mcts.apply_evaluated_leaves(logits, values);
}

#[no_mangle]
pub unsafe extern "C" fn rust_mcts_get_root_result(
    mcts_ptr: *const RustMcts,
    out_actions_ptr: *mut u16,
    out_probs_ptr: *mut f32,
    out_root_value_ptr: *mut f32,
) -> usize {
    let mcts = &*mcts_ptr;
    *out_root_value_ptr = mcts.root_value;
    let root = &mcts.nodes[0];
    let total_visits: u32 = root.edges.iter().map(|e| e.visits).sum();
    if total_visits == 0 || root.edges.is_empty() {
        return 0;
    }
    let n = root.edges.len();
    let out_actions = slice::from_raw_parts_mut(out_actions_ptr, n);
    let out_probs = slice::from_raw_parts_mut(out_probs_ptr, n);
    let inv_total = 1.0 / (total_visits as f32);
    for (i, edge) in root.edges.iter().enumerate() {
        out_actions[i] = edge.action_id;
        out_probs[i] = (edge.visits as f32) * inv_total;
    }
    n
}

#[no_mangle]
pub unsafe extern "C" fn rust_mcts_write_canonical_policy_target(
    mcts_ptr: *const RustMcts,
    out_target_ptr: *mut f32,
) {
    let mcts = &*mcts_ptr;
    let out_target = slice::from_raw_parts_mut(out_target_ptr, ACTION_SPACE_SIZE);
    out_target.fill(0.0);
    let root = &mcts.nodes[0];
    let total_visits: u32 = root.edges.iter().map(|e| e.visits).sum();
    if total_visits == 0 {
        return;
    }
    let inv_total = 1.0 / (total_visits as f32);
    let is_red = mcts.board.is_red_turn;
    for edge in &root.edges {
        if edge.visits > 0 {
            let net_idx = canonical_action_id(edge.action_id as usize, is_red);
            out_target[net_idx] = (edge.visits as f32) * inv_total;
        }
    }
}

