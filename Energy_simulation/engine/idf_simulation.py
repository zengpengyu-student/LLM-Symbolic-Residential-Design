"""Run an IDF with EnergyPlus and return its ESO output path.

This parameterized helper follows the source project's city-specific
idf_simulation scripts, while keeping outputs in the caller's temporary folder.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_energyplus(
    energyplus_exe: Path,
    epw_path: Path,
    idf_path: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(energyplus_exe),
        "-w",
        str(epw_path),
        "-d",
        str(output_dir),
        str(idf_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    eso_path = output_dir / "eplusout.eso"
    if result.returncode != 0 or not eso_path.is_file():
        error_path = output_dir / "eplusout.err"
        error_text = ""
        if error_path.is_file():
            error_text = error_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        details = "\n".join(part for part in (result.stderr.strip(), error_text.strip()) if part)
        raise RuntimeError(
            f"EnergyPlus failed (return code {result.returncode}).\n"
            f"Command: {' '.join(command)}\n{details}"
        )
    return eso_path
