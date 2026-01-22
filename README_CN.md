# MyChess (AlphaZero Remake)
一个基于 Pygame 的中国象棋 Python 实现，目前正在升级以包含 AlphaZero 风格的 AI。

[English](README.md) | [中文](README_CN.md)

## 快速开始

### 前提条件
- Python 3.8+
- Git

### 安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/mm12432/MyChess.git
   cd MyChess
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
