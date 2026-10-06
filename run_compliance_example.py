from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "Compliance_check"

subprocess.run(
    [sys.executable, "run_standard_check.py"],
    cwd=MODULE,
    check=True,
)
