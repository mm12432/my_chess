# my_chess (AlphaZero Remake)
A Python implementation of Chinese Chess (Xiangqi) with Pygame, currently being upgraded to include an AlphaZero-style AI.

[English](README.md) | [中文](README_CN.md)

## What's New in v1.3
- **Rust High-Performance MCTS Engine**: Introduced native Rust core search extension (`rust_core` compiled as `my_chess_rust_core.dll`) with fast C FFI bindings; features zero-copy make/unmake move backtracking, strict checkmate detection, and executes 800 MCTS simulations in under 0.3s.
- **MCTS Search Quality Upgrade (FPU Reduction)**: Integrated First-Play Urgency (FPU) penalty reduction for unvisited nodes in both Rust and Python MCTS, focusing the search tree on high-prior tactical lines and boosting opening move concentration by over 300%.
- **Automated Online Self-Training (`online-train`)**: Built-in automated online match engine powered by `ego-lite` browser environment to challenge `play.xiangqi.com` computer bots, extract teacher demonstrations, and incrementally fine-tune `artifacts/checkpoints/latest.pt` directly on CUDA.
- **10 Bot Difficulty Levels (`--level [1-10]`)**: Full support for selecting opponent bot difficulty from Level 1 (Meow beginner bot) to Level 10 (Hu grandmaster bot) for progressive curricula learning.
- **Zero-Calibration Resolution Auto-Adaptation**: Dynamic DOM geometry evaluation via `getBoundingClientRect()` at runtime enables pixel-perfect centering on any screen (1080p, 2K, 4K, ultrawide) and any DPI scaling ratio (100%~200%) without manual calibration.
- **Bidirectional WXF Notation Converter**: Added robust converter between internal ICCS moves and standard World Xiangqi Federation (WXF) algebraic notation (e.g. `P9+1`, `c2=5`, front/back disambiguation `+n+6`).
- **Post-Game GUI Incremental Learning**: Supports human vs AI play with immediate learning (`python run.py ai-black learn`), 3x policy weighting on human winning moves, and symmetry mirror augmentation.
- **Expanded Test Suite**: Unit tests increased to 58, verifying pure rule outcomes, WXF notation parsing, and Rust engine safety.

## What's New in v1.2
- **UI Overhaul**: Expanded window layout (800×800 → 1000×800) with a new sidebar panel featuring turn indicator, game timer, standard Chinese notation move log, and control buttons (Restart, Save FEN).
- **Rule Engine**: Implemented Zobrist Hashing for position tracking, perpetual check/chase detection (offender loses on 3-fold repetition), and stalemate detection (no legal moves = loss).
- **Chinese Notation**: Real-time standard Chinese Chess notation (e.g., 炮二平五) displayed in the sidebar.
- **FEN Support**: Save and load board positions using standard FEN strings.
- **Code Quality**: Achieved Pylint 10.00/10. Full PEP 8 compliance with project rename (`MyChess` → `my_chess`). Complete docstring coverage.
- **Testing**: 24 unit tests covering all 7 piece types' movement rules, stalemate, repetition penalty, and notation correctness.
- **Coordinate System**: Added UCCI/ICCS coordinate conversion for future AI integration.

## What's New in v1.1
- **Modernization**: Migrated legacy Python 2 code to Python 3.8+ standards (Type Hints, `dataclasses`, new `super()` syntax).
- **Engineering**: Added `pyproject.toml`, standard `.gitignore`, and pinned dependencies in `requirements.txt`.
- **Infrastructure**: Introduced `run.py` as a unified entry point to fix import path issues.
- **Code Style**: Enforced `black` and `isort` formatting rules.
- **Fixes**: Resolved Python 3 crash in CLI mode input handling.

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mm12432/my_chess.git
   cd my_chess
   ```

2. **Set up Virtual Environment** (Recommended)
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\Activate
   # Linux/MacOS:
   # source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch** (Required for AI training / MCTS + model inference)

   PyTorch is platform-dependent, so it is installed separately from `requirements.txt`.

   **Windows + NVIDIA GPU example (`cu124`):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   **Verify GPU availability:**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

## Usage

**Run the GUI Game (Default):**
```bash
python run.py
```

**Run the GUI Human vs AI mode:**
```bash
python run.py ai
```

**Run the GUI Human vs AI mode (AI plays Red):**
```bash
python run.py ai-red
```

**Run the GUI Human vs AI mode (AI plays Black):**
```bash
python run.py ai-black
```

**Run GUI Human vs AI with post-game incremental learning (example: AI plays Black):**
```bash
python run.py ai-black learn
```

**Run automated online training matches (play against play.xiangqi.com Computer):**
```bash
# Default: 1 game with 800 MCTS simulations against Level 1 (Meow):
python run.py online-train

# Select bot level (1 to 10, e.g. Level 3 Huma or Level 10 Hu):
python run.py online-train --level 3 --games 2

# Custom number of games, bot level, simulations, and max moves:
python run.py online-train --games 3 --level 5 --simulations 800 --max-moves 300
```
> **Opponent Difficulty Levels (1-10)**:
> 1: Meow (Beginner) | 2: Coach David | 3: Huma | 4: Chia | 5: Paul  
> 6: Trung | 7: Dr. Png | 8: ZhiXuan | 9: Joep | 10: Hu (Grandmaster bot)
>
> **Prerequisites**:
> Automated online training requires the **ego-lite** (i.e. `ego-browser`) browser environment for DOM control and IPC bridge execution. Ensure that `ego-browser` CLI is installed and available in your environment PATH.
>
> **Game Termination Rules**:
> The session evaluates matches under formal Chinese Chess rules (checkmate/stalemate, 3-fold repetition, 120-halfmove captureless natural draw, or web modal result). Games are no longer artificially truncated at 60 rounds and will play to completion.
>
> **Resolution & Coordinate Auto-Adaptation Note**:
> The automated training agent queries `.square[class*="{rank}-{file}"]` directly from the webpage DOM and invokes `getBoundingClientRect()` at runtime to calculate each square's exact geometric center. Whether your screen resolution is 1080p, 2K, 4K, ultrawide, or using 125%/150% DPI scaling, **all clicks are automatically centered with pixel precision without requiring any manual coordinate calibration!**

**Run the Terminal Mode:**
```bash
python run.py cli
```

**Run the test suite:**
```bash
python -m pytest
```

**Run iterative AI training:**
```bash
python run.py train
```

## Current AI Status
The project now includes the first working AlphaZero-style foundation layer:

- Stable `Move -> action_id` mapping (`90 x 90 = 8100` action space)
- Legal-action mask generation
- Multi-channel board encoding (`15 x 10 x 9`)
- Minimal residual CNN model interface (`policy logits + value`)
- PUCT-style MCTS skeleton
- Self-play sample generation
- One-step policy/value training loop

This does **not** mean the engine is already strong. It means the project has moved from:

- "rule engine + GUI"

to:

- "rule engine + GUI + trainable AI pipeline skeleton"

## AI Development And Verification

### AI-related modules

- [action_space.py](/F:/source_code/my_chess/chess_ai/action_space.py)
- [encoding.py](/F:/source_code/my_chess/chess_ai/encoding.py)
- [dataset.py](/F:/source_code/my_chess/chess_ai/dataset.py)
- [model.py](/F:/source_code/my_chess/chess_ai/model.py)
- [evaluator.py](/F:/source_code/my_chess/chess_ai/evaluator.py)
- [mcts.py](/F:/source_code/my_chess/chess_ai/mcts.py)
- [self_play.py](/F:/source_code/my_chess/chess_ai/self_play.py)
- [train.py](/F:/source_code/my_chess/chess_ai/train.py)

### Recommended installation flow

1. Install base dependencies
```bash
pip install -r requirements.txt
```

2. Install PyTorch for your local environment  
Note:
PyTorch is intentionally not pinned in `requirements.txt` because CPU / CUDA / platform variants differ.

Windows + NVIDIA GPU example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. Verify GPU availability
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Recommended verification commands

Run tests:
```bash
python -m pytest
```

Run iterative self-play training:
```bash
python run.py train
```

Use a larger run explicitly:
```bash
python run.py train --iterations 20 --games-per-iteration 8 --num-simulations 12 --train-steps-per-iteration 24
```

If you see output like:
```text
[iter 1/10] generated=... buffer=... loss=... policy=... value=... checkpoint=...
```

then the current training loop is working:
- iterative self-play sample generation
- repeated policy/value updates
- checkpoint save after each iteration
- resumable training

The default checkpoint path is:
```text
artifacts/checkpoints/latest.pt
```

When that checkpoint exists, GUI AI will load it automatically instead of using random initialization.

The human-vs-AI replay store is persisted at:
```text
artifacts/experience/human_replay.pkl
```

### Current limitations

The AI layer is still in the skeleton phase, not a strong engine yet:

- no trained strong model yet
- self-play scale is still small
- package structure can still be standardized further

## Development
This project follows modern python best practices.
- Type checking: `mypy` (Planned)
- Code formatting: `black`
- Tests: `pytest`
