# AGENTS.md - my_chess 开发指南

## 核心开发命令

- **启动 GUI (默认)**: `python run.py`
- **启动 GUI 人机对战**:
  - 默认 AI: `python run.py ai`
  - AI 执红: `python run.py ai-red`
  - AI 执黑: `python run.py ai-black`
- **GUI 人机对战 + 增量学习**: `python run.py ai-black learn`
- **启动 CLI 模式**: `python run.py cli`
- **启动 AI 训练**: `python run.py train` (支持 `--iterations`, `--games-per-iteration`, `--num-simulations` 参数)
- **运行单元测试**: `python -m pytest`

## 验证与代码质量

- **验证顺序**: `python -m pytest` $\rightarrow$ `black .` $\rightarrow$ `isort .`
- **代码标准**: 严格遵循 PEP 8，使用 `black` 和 `isort` 进行格式化。
- **类型检查**: 计划使用 `mypy`。

## 项目结构与模块

- `chess_core/`: 核心规则引擎，含局面追踪、合法走法、Zobrist Hashing 和 FEN 支持。
- `chess_ui/`: 基于 Pygame 的界面实现，含中文记谱和侧边栏信息面板。
- `chess_ai/`: AlphaZero 风格 AI 实现，包含：
  - `model.py`: 残差 CNN 模型。
  - `mcts.py`: PUCT 风格蒙特卡洛树搜索。
  - `train.py` & `self_play.py`: 训练循环与样本生成。
- `artifacts/`: 存放 AI 产物。
  - 权重路径: `artifacts/checkpoints/latest.pt`
  - 经验池路径: `artifacts/experience/human_replay.pkl`

## 环境与依赖 (Windows)

- **Python**: 3.8+
- **PyTorch**: 不在 `requirements.txt` 中，需根据 CUDA 版本手动安装 (例如 `cu124`)。
- **虚拟环境**: 推荐使用 `.venv`。
- **GPU 验证**: `python -c "import torch; print(torch.cuda.is_available())"`
