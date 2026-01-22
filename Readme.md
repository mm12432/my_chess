# MyChess (AlphaZero Remake)
A Python implementation of Chinese Chess (Xiangqi) with Pygame, currently being upgraded to include an AlphaZero-style AI.

[English](README.md) | [中文](README_CN.md)

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mm12432/MyChess.git
   cd MyChess
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
