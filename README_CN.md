# my_chess (AlphaZero Remake)
一个基于 Pygame 的中国象棋 Python 实现，目前正在升级以包含 AlphaZero 风格的 AI。

[English](README.md) | [中文](README_CN.md)

## v1.3 版本更新日志
- **Rust 高性能 MCTS 搜索核心**：引入基于 Rust 编写的高性能棋盘与树搜索扩展（`rust_core`），通过原生 C ABI 动态链接库与 Python 极速交互；实现 Make/Unmake 极速状态回溯、严格将军与绝杀判定，单步 800 次 MCTS 模拟耗时压缩至 0.3 秒以内。
- **MCTS 智能策略升级（FPU 减免）**：在 MCTS 中落地 First-Play Urgency (FPU) 未访问惩罚机制，避免在弱势着法上浪费搜索算力，开局与复杂战术局面选点集中度提升 300% 以上。
- **自动化线上对弈学习 (`online-train`)**：基于 `ego-lite` 浏览器引擎打造自动化对弈系统，自动挑战 `play.xiangqi.com` 网页电脑；局后自动将对局转化为高价值示范样本，执行多步加权反向传播实时更新 `artifacts/checkpoints/latest.pt`。
- **1~10 级对手自由选择**：新增 `--level [1-10]` 参数，可自由选择从 1 级（Meow 初级电脑）到 10 级（Hu 大师级电脑）全部 10 个梯度对手进行对抗训练。
- **全分辨率与 DPI 零校准自适应**：网页点击完全依托 DOM 节点与运行时 `getBoundingClientRect()` 动态几何求心，无论 1080P/2K/4K 分辨率或任意 Windows DPI 缩放比例，落子零误差，彻底免除手动坐标校准。
- **标准 WXF 记谱转换引擎**：实现内部 ICCS 坐标与世界象棋联合会 WXF 标准代数记谱（如 `P9+1`, `c2=5`, 前后马 `+n+6`）的双向解析转化。
- **人机对弈增量学习**：本地 GUI 支持 `python run.py ai-black learn`，人类战胜或行棋局后自动进行 3 倍权重加权与盘面对称镜像增强（Mirror Augmentation）。
- **测试套件扩充**：单元测试扩展至 58 项，涵盖纯胜负规则、WXF 记谱解析和 Rust 引擎绑定验证。

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

**运行自动化线上对弈训练（挑战 play.xiangqi.com 网页电脑）:**
```bash
# 默认对局 1 盘，800 次 MCTS 搜索（默认对手为 1 级 Meow）：
python run.py online-train

# 指定对手难度级别（1 到 10 级，例如挑战 3 级 Huma 或 10 级 Hu）：
python run.py online-train --level 3 --games 2

# 自定义连续对局盘数、难度级别与搜索算力：
python run.py online-train --games 3 --level 5 --simulations 800 --max-moves 300
```
> **对手级别对照表（1~10 级）**：
> 1: Meow (初级) | 2: Coach David | 3: Huma | 4: Chia | 5: Paul  
> 6: Trung | 7: Dr. Png | 8: ZhiXuan | 9: Joep | 10: Hu (最高级大师电脑)
>
> **前置环境要求**：
> 自动化线上对战依赖 **ego-lite**（即 `ego-browser`）智能浏览器环境执行自动化操控与跨进程通信。请确保系统已安装并配置好 `ego-browser` 命令行工具。
>
> **判和与胜负规则**：
> 系统采用标准象棋规则进行裁定（困毙/绝杀、三次重复局面造成重复方判负、连续 120 步无吃子自然限着判和，以及网页弹窗裁定）。对局不再受早期的 60 回合人为硬截断限制，双方将战至分出胜负或达成正式规则和棋。
>
> **分辨率与坐标自适应说明**：
> 自动化对弈系统通过浏览器 DOM 动态查询 `.square[class*="{rank}-{file}"]` 棋格，并在运行时实时调用 `getBoundingClientRect()` 计算棋盘格的几何物理中心点。无论你的电脑屏幕是 1080p、2K、4K、带鱼屏，还是系统设置了 125%、150% 等任意 DPI 缩放比例，**全部自动像素级精准居中点击，用户完全不需要手动校准坐标**！

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
python run.py train --iterations 20 --games-per-iteration 8 --num-simulations 12 --train-steps-per-iteration 24
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
