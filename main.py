"""Compatibility entry point for SubspaceAD."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    model = Path(__file__).resolve().parent / "models" / "subspacead"
    sys.path.insert(0, str(model))
    runpy.run_path(str(model / "main.py"), run_name="__main__")
