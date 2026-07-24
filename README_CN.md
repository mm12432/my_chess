# my_chess (AlphaZero Remake)
一个基于 Pygame 的中国象棋 Python 实现，目前正在升级以包含 AlphaZero 风格的 AI。

[English](README.md) | [中文](README_CN.md)

## v1.2 版本更新日志
- **UI 全面升级**：窗口布局扩展 (800x800 -> 1000x800)，新增侧边栏信息面板，含回合指示器、对弈计时器、标准中文记谱面板和控制按钮（重开、保存 FEN）。
- **规则引擎**：基于 Zobrist Hashing 实现局面追踪，新增长将/长捉检测（三次重复局面违规方判负）和困毙检测（无合法走法即判负）。
- **中文记谱**：实时生成标准中国象棋记谱（如“炮二平五”），在侧边栏滚动显示。
- **FEN 支持**：支持使用标准 FEN 字符串保存和加载棋盘局面。
- **代码质量**：Pylint 评分达到 10.00/10 满分，全面符合 PEP 8 规范，项目从 `MyChess` 统一重命名为 `my_chess`，补全所有模块文档字符串。
- **测试覆盖**：24 个单元测试，覆盖全部 7 种棋子走法规则、困毙判定、长将长捉判负和记谱正确性。
- **坐标系统**：新增 UCCI/ICCS 坐标转换，为后续 AI 对接做准备。

## v1.1 版本更新日志
- **现代化重构**: 将老旧的 Python 2 代码全面升级为 Python 3.8+ 标准（引入类型提示 Type Hints, `dataclasses`, 新版 `super()` 语法）。
- **工程化建设**: 新增 `pyproject.toml` 配置，标准化的 `.gitignore`，并锁定了 `requirements.txt` 依赖版本。
- **基础设施**: 引入 `run.py` 作为统一启动入口，彻底解决了包导入路径问题。
- **代码规范**: 强制执行 `black` 和 `isort` 代码格式化标准。
- **缺陷修复**: 修复了 CLI 命令行模式在 Python 3 下的输入崩溃问题。

## 快速开始

### 前提条件
- Python 3.8+
- Git

### 安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/mm12432/my_chess.git
   cd my_chess
   ```

2. **设置虚拟环境** (推荐)
   ```bash
   # 创建虚拟环境
   python -m venv .venv
   
   # 激活虚拟环境
   # Windows:
   .venv\Scripts\Activate
   # Linux/MacOS:
   # source .venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **安装 PyTorch**（AI 训练 / MCTS + 模型推理需要）

   由于 PyTorch 依赖 CPU / CUDA / 平台环境，不放在 `requirements.txt` 里统一锁死，建议按本机环境单独安装。

   **Windows + NVIDIA GPU 示例（`cu124`）：**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   **验证 GPU 是否可用：**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

## 使用方法

**运行 GUI 游戏 (默认):**
```bash
python run.py
```

**运行 GUI 人机对战模式:**
```bash
python run.py ai
```

**运行 GUI 人机对战模式（AI 执红）:**
```bash
python run.py ai-red
```

**运行 GUI 人机对战模式（AI 执黑）:**
```bash
python run.py ai-black
```

**运行 GUI 人机对战并在局后增量学习（示例：AI 执黑）:**
```bash
python run.py ai-black learn
```

**运行终端模式:**
```bash
python run.py cli
```

**运行测试：**
```bash
python -m pytest
```

**运行正式 AI 训练：**
```bash
python run.py train
```

## 当前 AI 状态
目前项目已经具备第一批可运行的 AlphaZero 风格基础设施：

- 稳定的 `Move -> action_id` 映射（`90 x 90 = 8100` 动作空间）
- 合法动作 mask 生成
- 多通道棋盘编码（`15 x 10 x 9`）
- 最小残差 CNN 模型接口（`policy logits + value`）
- PUCT 风格 MCTS 骨架
- self-play 样本生成
- 一步 policy/value 训练循环

这并不表示当前引擎已经很强，而是表示项目已经从：

- “规则引擎 + GUI”

推进到：

- “规则引擎 + GUI + 可训练 AI 管线骨架”

## AI 开发与验证

### AI 相关模块

- [action_space.py](/F:/source_code/my_chess/chess_ai/action_space.py)
- [encoding.py](/F:/source_code/my_chess/chess_ai/encoding.py)
- [dataset.py](/F:/source_code/my_chess/chess_ai/dataset.py)
- [model.py](/F:/source_code/my_chess/chess_ai/model.py)
- [evaluator.py](/F:/source_code/my_chess/chess_ai/evaluator.py)
- [mcts.py](/F:/source_code/my_chess/chess_ai/mcts.py)
- [self_play.py](/F:/source_code/my_chess/chess_ai/self_play.py)
- [train.py](/F:/source_code/my_chess/chess_ai/train.py)

### 当前推荐安装方式

1. 安装基础依赖
```bash
pip install -r requirements.txt
```

2. 按本机环境安装 PyTorch  
说明：
PyTorch 不写死在 `requirements.txt`，因为 CPU / CUDA / 平台环境相关。

Windows + NVIDIA GPU 示例：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. 验证 GPU
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 当前推荐验证命令

运行测试：
```bash
python -m pytest
```

运行正式 self-play 训练：
```bash
python run.py train
```

加大训练规模可以直接这样跑：
```bash
python run.py train --iterations 50 --games-per-iteration 8 --num-simulations 64
```

如果看到类似输出：
```text
[iter 1/10] generated=... buffer=... loss=... policy=... value=... checkpoint=...
```

表示当前训练主链已经打通：
- self-play 持续生成样本
- policy/value 持续更新
- 每轮训练后保存 checkpoint
- 训练可恢复继续

默认 checkpoint 路径是：
```text
artifacts/checkpoints/latest.pt
```

当这个 checkpoint 存在时，GUI 人机模式会优先加载它，而不是继续使用随机初始化模型。

人机对局增量学习的经验池默认保存在：
```text
artifacts/experience/human_replay.pkl
```

### 当前限制

当前 AI 部分还是骨架阶段，不是成熟引擎：

- 还没有真正训练出有强度的模型
- self-play 规模仍然很小
- 包结构仍可继续标准化

## 开发
本项目遵循现代 Python 最佳实践。
- 类型检查: `mypy` (计划中)
- 代码格式化: `black`
- 测试: `pytest`
