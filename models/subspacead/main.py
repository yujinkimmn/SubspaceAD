"""Run the existing SubspaceAD model from the shared models directory."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "main.py"), run_name="__main__")
