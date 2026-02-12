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

## 使用方法

**运行 GUI 游戏 (默认):**
```bash
python run.py
```

**运行终端模式:**
```bash
python run.py cli
```

## 开发
本项目遵循现代 Python 最佳实践。
- 类型检查: `mypy` (计划中)
- 代码格式化: `black`
- 测试: `pytest`
