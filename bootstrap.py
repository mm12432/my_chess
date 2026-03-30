"""Runtime import bootstrap helpers for direct project-root execution.

This project currently uses `my_chess.*` imports while the repository root is
itself the package directory. When running scripts directly from the project
root, Python needs the parent directory on `sys.path` so `my_chess` resolves
consistently.
"""

from __future__ import annotations

import sys
from pathlib import Path


def setup_project_imports() -> None:
    """Ensures the repository parent directory is importable as `my_chess`."""
    root_dir = Path(__file__).resolve().parent
    parent_dir = root_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
