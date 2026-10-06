from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "Energy_simulation"

subprocess.run(
    [sys.executable, "run_energy_simulation.py"],
    cwd=MODULE,
    check=True,
)
