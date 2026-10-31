"""Launch the canonical Amazon-50K compact materializer from ``src``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[3] / "src" / "data" / "materialization" / "build_amazon50k_compact_light3.py"

if __name__ == "__main__":
    sys.path.insert(0, str(SOURCE.parent))
    runpy.run_path(str(SOURCE), run_name="__main__")
