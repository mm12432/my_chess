# my_chess (AlphaZero Remake)
A Python implementation of Chinese Chess (Xiangqi) with Pygame, currently being upgraded to include an AlphaZero-style AI.

[English](README.md) | [中文](README_CN.md)

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

## Usage

**Run the GUI Game (Default):**
```bash
python run.py
```

**Run the Terminal Mode:**
```bash
python run.py cli
```

## Development
This project follows modern python best practices.
- Type checking: `mypy` (Planned)
- Code formatting: `black`
- Tests: `pytest`
