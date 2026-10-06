from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "Fine_layout_generation"

subprocess.run(
    [sys.executable, "Inference.py"],
    cwd=MODULE,
    check=True,
)
