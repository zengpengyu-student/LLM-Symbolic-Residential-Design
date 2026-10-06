"""Run the annual building-energy simulation workflow for one vector JSON in four climates.

Only the final energy metrics are printed. Converted JSON, IDF, and
EnergyPlus outputs are written to a temporary directory and removed afterward.
"""

from __future__ import annotations

import io
import json
import os
import runpy
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ENGINE_DIR = HERE / "engine"

# Change this path to the vector JSON you want to evaluate.
INPUT_JSON_PATH = PROJECT_ROOT / "test" / "Energy_test" / "0.json"

ENERGYPLUS_DIR = Path(os.environ.get("ENERGYPLUS_DIR", ""))
ENERGYPLUS_EXE = ENERGYPLUS_DIR / "energyplus.exe"
ENERGYPLUS_IDD = ENERGYPLUS_DIR / "Energy+.idd"
ENERGYPLUS_VERSION = "23.1"

CITIES = (
    ("Australia / Sydney", "Sydney.epw"),
    ("Canada / Toronto", "Toronto.epw"),
    ("China / Shenzhen", "Shenzhen.epw"),
    ("London", "London.epw"),
)

def _load_standard_idf_writer():
    """Load the standard IDF writer and set the EnergyPlus version."""
    script_path = ENGINE_DIR / "json_to_idf_standard.py"
    namespace = runpy.run_path(str(script_path))
    writer = namespace["json_to_idf"]
    writer_globals = writer.__globals__
    writer_globals["EP_VERSION"] = ENERGYPLUS_VERSION
    return writer


def load_source_functions():
    """Load the vector converter, ESO parser, and EnergyPlus launcher."""
    converter_ns = runpy.run_path(str(ENGINE_DIR / "vector_to_energy_json.py"))
    parser_ns = runpy.run_path(str(ENGINE_DIR / "parse_eso.py"))
    simulation_ns = runpy.run_path(str(ENGINE_DIR / "idf_simulation.py"))
    return (
        converter_ns["convert_floor_plan_format"],
        _load_standard_idf_writer(),
        parser_ns["parse_eso_file"],
        simulation_ns["run_energyplus"],
    )


def _energy_component_kwh(frame, suffix: str) -> float:
    columns = [column for column in frame.columns if str(column).split("_")[-1] == suffix]
    if not columns:
        raise RuntimeError(f"ESO parser found no {suffix} energy variables.")
    # The source step-5 script converts summed joules to kWh in the same way.
    return float(frame[columns].sum().sum() / 3_600_000.0)


def main() -> None:
    input_path = INPUT_JSON_PATH.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Vector JSON not found: {input_path}")
    if not ENERGYPLUS_EXE.is_file():
        raise FileNotFoundError(
            f"EnergyPlus executable not found: {ENERGYPLUS_EXE}\n"
            "Edit ENERGYPLUS_DIR near the top of this file or set the ENERGYPLUS_DIR environment variable."
        )
    if not ENERGYPLUS_IDD.is_file():
        raise FileNotFoundError(f"EnergyPlus IDD not found: {ENERGYPLUS_IDD}")

    epw_paths = [(name, ENGINE_DIR / "EPW" / epw_name) for name, epw_name in CITIES]
    missing_epw = [str(path) for _, path in epw_paths if not path.is_file()]
    if missing_epw:
        raise FileNotFoundError("Missing city weather file(s):\n" + "\n".join(missing_epw))

    with input_path.open("r", encoding="utf-8") as source_file:
        original_vector = json.load(source_file)
    required_keys = {"Pic", "door", "window"}
    missing_keys = sorted(required_keys - set(original_vector))
    if missing_keys:
        raise ValueError(f"Input JSON is missing required source keys: {', '.join(missing_keys)}")

    convert_vector, json_to_idf, parse_eso_file, run_energyplus = load_source_functions()
    converted_vector = convert_vector(original_vector)

    with tempfile.TemporaryDirectory(prefix="energyplus_run_") as temp_root:
        work_dir = Path(temp_root)
        converted_json_path = work_dir / f"{input_path.stem}_energy_format.json"
        idf_path = work_dir / f"{input_path.stem}.idf"
        with converted_json_path.open("w", encoding="utf-8") as converted_file:
            json.dump(converted_vector, converted_file, ensure_ascii=False)

        # The source writer creates a small seed IDF through tempfile before
        # saving the completed IDF. Keep that intermediate inside this scratch
        # directory too, so it is removed with the simulation outputs.
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = str(work_dir)
        try:
            with redirect_stdout(io.StringIO()):
                json_to_idf(converted_json_path, ENERGYPLUS_IDD, idf_path)
        finally:
            tempfile.tempdir = previous_tempdir

        for city_name, epw_path in epw_paths:
            print(f"\nRunning EnergyPlus: {city_name}")
            output_dir = work_dir / "energyplus_output" / epw_path.stem
            eso_path = run_energyplus(ENERGYPLUS_EXE, epw_path, idf_path, output_dir)
            frame, _, _ = parse_eso_file(eso_path)
            if frame.empty:
                raise RuntimeError(f"ESO parser returned no time-series data for {city_name}: {eso_path}")

            heating_kwh = _energy_component_kwh(frame, "heating")
            cooling_kwh = _energy_component_kwh(frame, "cooling")
            lighting_kwh = _energy_component_kwh(frame, "lighting")
            annual_energy_kwh = heating_kwh / 3.0 + cooling_kwh / 3.0 + lighting_kwh

            print(f"Heating load: {heating_kwh:.2f} kWh_th/year")
            print(f"Cooling load: {cooling_kwh:.2f} kWh_th/year")
            print(f"Lighting electricity: {lighting_kwh:.2f} kWh/year")
            print(
                "Annual energy (step-5 method: heating/3 + cooling/3 + lighting): "
                f"{annual_energy_kwh:.11f} kWh/year"
            )

    print("\nAll four city simulations completed. Temporary files were removed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEnergy simulation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
