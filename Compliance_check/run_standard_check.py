"""Run one compliance check and print the result as JSON."""

import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

DETECTION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DETECTION_DIR.parent
TEST_DIR = PROJECT_DIR / "test" / "Compliance_check"
sys.path.insert(0, str(DETECTION_DIR))

from vector_renderer import render_floorplan

# Update these paths when testing another case.
VECTOR_JSON_PATH = TEST_DIR / "floorplan_vectors.json"
EDGE_MASK_PATH = VECTOR_JSON_PATH.with_name("edge_256.png")
CASE_METADATA_PATH = VECTOR_JSON_PATH.with_name("case_metadata.json")
CHECKER_PATH = DETECTION_DIR / "toronto_compliance_checker.py"


def _vector_room_count(vector_data, room_type):
    value = vector_data.get(room_type, [])
    if isinstance(value, str):
        return 1
    if isinstance(value, list):
        return sum(isinstance(path, str) for path in value)
    return 0


def run_check():
    sys.path.insert(0, str(DETECTION_DIR))
    spec = importlib.util.spec_from_file_location("toronto_compliance_checker", CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load checker: {CHECKER_PATH}")

    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    with VECTOR_JSON_PATH.open("r", encoding="utf-8") as stream:
        vector_data = json.load(stream)
    if CASE_METADATA_PATH.is_file():
        with CASE_METADATA_PATH.open("r", encoding="utf-8") as stream:
            case_metadata = json.load(stream)
        room_number_original = case_metadata["room_number_original"]
    else:
        room_counts = [
            _vector_room_count(vector_data, room_type)
            for room_type in ("living_room", "bedroom", "bathroom", "kitchen", "balcony")
        ]
        room_number_original = str(room_counts)

    image = render_floorplan(vector_data)
    original_data = {"room_number_original": room_number_original}
    return checker.analyse_image(
        None,
        original_data,
        edge_path=EDGE_MASK_PATH if EDGE_MASK_PATH.is_file() else None,
        vector_data=vector_data,
        image=image,
        image_name=vector_data.get("Pic", "floorplan.png"),
    )


def main():
    print(json.dumps(run_check(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
