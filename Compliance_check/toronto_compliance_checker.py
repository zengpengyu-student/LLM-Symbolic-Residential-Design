"""Toronto building-code compliance checks for floor-plan images."""

from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional, List, Any
from math import hypot
import re

import cv2
import numpy as np
import yaml

import FireTest2


def _suppress_firetest_visualization(*_args, **_kwargs):
    return None


FireTest2.draw_firetest = _suppress_firetest_visualization
max_distance_class = FireTest2.max_distance_class

from SETUP import PIXELS_PER_M

SCRIPT_DIR = Path(__file__).resolve().parent
                                   
                                                       
                                                       
CONFIG_PATH = SCRIPT_DIR / "config_from_Toronto.yaml"
                                 

                                          

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG: Dict[str, Any] = load_config(CONFIG_PATH)
HOUSE_CLASSES: List[Dict[str, Any]] = CONFIG.get("house_classes", [])
_HOUSE_CLASS_BY_NAME: Dict[str, Dict[str, Any]] = {
    c["name"]: c for c in HOUSE_CLASSES if "name" in c
}

depth_cfg = CONFIG.get("depth_ratio", {}) or {}
DEPTH_RATIO_RANGE: Tuple[float, float] = (
    float(depth_cfg.get("min", 1.5)),
    float(depth_cfg.get("max", 2.5)),
)

TOPOLOGY_CONFIG: Dict[str, Any] = CONFIG.get("topology", {}) or {}

FIRE_CONFIG: Dict[str, Any] = CONFIG.get("fire_corridor", {}) or {}
FIRE_RANGE: Tuple[float, float] = (
    float(FIRE_CONFIG.get("min", 0.0)),
    float(FIRE_CONFIG.get("max", 0.0)),
)

                                                                  

BGR_THRESHOLDS: Dict[str, Tuple[List[int], List[int]]] = {
        
    "wall": ([0, 0, 245], [20, 20, 255]),
    "window": ([0, 100, 0], [20, 150, 20]),
    "door": ([245, 0, 0], [255, 20, 20]),
           
    "living_room": ([219, 231, 234], [239, 251, 254]),
    "bedroom": ([160, 234, 243], [180, 254, 255]),
    "balcony": ([124, 205, 197], [144, 225, 217]),
    "kitchen": ([203, 205, 224], [223, 225, 244]),
    "bathroom": ([241, 222, 194], [255, 242, 214]),
}

                                                    

def count_non_white_pixels(img, white_threshold=(255, 255, 255)):
                                 
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                            
    white_b, white_g, white_r = white_threshold
                                    
    white_mask = (img[:, :, 0] == white_b) &\
                 (img[:, :, 1] == white_g) &\
                 (img[:, :, 2] == white_r)

                                
    total_pixels = img.shape[0] * img.shape[1]
    white_pixels = np.sum(white_mask)
    non_white_pixels = total_pixels - white_pixels

    return non_white_pixels

def create_mask(img, lo, hi):
    return cv2.inRange(img, np.array(lo, np.uint8), np.array(hi, np.uint8))


def classify_house(total_m2: float) -> Optional[str]:
    for cls_cfg in HOUSE_CLASSES:
        total_cfg = cls_cfg.get("total_area") or {}
        lo = total_cfg.get("min")
        hi = total_cfg.get("max")
        if lo == 0 and hi == 0:
            return cls_cfg["name"]
        elif lo == 0:
            if total_m2 <= float(hi):
                return cls_cfg["name"]
            else:
                continue
        elif hi == 0:
            if float(lo) <= total_m2:
                return cls_cfg["name"]
            else:
                continue
        elif float(lo) <= total_m2 <= float(hi):
            return cls_cfg["name"]
    return None


def _get_room_spec(house_class: str, room_type: str) -> Optional[Dict[str, Any]]:
    hc = _HOUSE_CLASS_BY_NAME.get(house_class)
    if not hc:
        return None
    rooms_spec = hc.get("rooms") or {}
    spec = rooms_spec.get(room_type)
    if spec is None:
        return None
    return spec


def _parse_min_max(cfg: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(cfg, dict):
        return None
    lo = cfg.get("min")
    hi = cfg.get("max")
    if lo is None or hi is None:
        return None
    try:
        return float(lo), float(hi)
    except (TypeError, ValueError):
        return None


def _format_min_max(rng: Optional[Tuple[float, float]]) -> Optional[str]:
    if rng is None:
        return None
    lo, hi = rng
    def format_bound(value):
        number = float(value)
        if number.is_integer() and number != 0:
            return str(int(number))
        return str(number)
    return f"{format_bound(lo)}-{format_bound(hi)}"


def compare_to_min_max(value: Optional[float], rng: Optional[Tuple[float, float]]) -> Optional[str]:
    if value is None or rng is None:
        return None
    lo, hi = rng
    if lo == 0 and hi == 0:
        return "ok"
    elif lo == 0:
        if value > hi:
            return "above_max"
        else:
            return "ok"
    elif hi == 0:
        if value < lo:
            return "below_min"
        else:
            return "ok"
        
    if value < lo:
        return "below_min"
    if value > hi:
        return "above_max"
    return "ok"
        


def apply_room_spec(room: Dict[str, Any], spec: Dict[str, Any]):
    area_rng = _parse_min_max(spec.get("area")) or _parse_min_max(spec)
    width_rng = _parse_min_max(spec.get("width"))
    depth_rng = _parse_min_max(spec.get("depth"))

    d1 = room.get("depth")
    d2 = room.get("width")
    if isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
        d_long, d_short = (d1, d2) if d1 >= d2 else (d2, d1)
        if depth_rng is not None and width_rng is None:
            room["depth"] = d_short
            room["width"] = d_long
        else:
            room["depth"] = d_long
            room["width"] = d_short

    room["expected_area"] = _format_min_max(area_rng)
    room["area_status"] = compare_to_min_max(room.get("area_m2"), area_rng)

    room["expected_width"] = _format_min_max(width_rng)
    room["width_status"] = compare_to_min_max(room.get("width"), width_rng)

    room["expected_depth"] = _format_min_max(depth_rng)
    room["depth_status"] = compare_to_min_max(room.get("depth"), depth_rng)


def calc_depth_ratio(contour) -> Optional[float]:
    _, _, ratio = calc_depth_width_ratio(contour)
    return ratio


def calc_depth_width_ratio(contour) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if contour is None or len(contour) < 3:
        return None, None, None
    rect = cv2.minAreaRect(contour)
    w_px, h_px = rect[1]
    if w_px == 0 or h_px == 0:
        return None, None, None
    long_px, short_px = max(w_px, h_px), min(w_px, h_px)
    depth_m = long_px / PIXELS_PER_M
    width_m = short_px / PIXELS_PER_M
    ratio = long_px / short_px
    return round(depth_m, 2), round(width_m, 2), round(ratio, 2)


def judge_ratio(ratio: Optional[float]):
    return compare_to_min_max(ratio, DEPTH_RATIO_RANGE)


                                                                 


def detect_doors(img, min_area=1):
    lo, hi = BGR_THRESHOLDS["door"]
    mask = create_mask(img, lo, hi)
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doors = []
    for c in ctrs:
        if cv2.contourArea(c) < min_area:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        orient = "h" if w > h else "v"
        doors.append({"contour": c, "center": (int(cx), int(cy)), "orient": orient})
    return doors


def detect_door_segments(img, min_len=5, line_gap=2, dilate_k=3):
    lo, hi = BGR_THRESHOLDS["door"]
    door_mask = create_mask(img, lo, hi)

    if dilate_k:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        door_mask = cv2.dilate(door_mask, k, iterations=1)

    edges = cv2.Canny(door_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=min_len, maxLineGap=line_gap)
    if lines is None:
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=6,
            minLineLength=max(3, min_len // 2),
            maxLineGap=line_gap + 1,
        )
    if lines is None:
        return []

    door_segs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        v = np.array([x2 - x1, y2 - y1], dtype=float)
        L = np.linalg.norm(v)
        if L < min_len:
            continue
        n = np.array([-v[1], v[0]])
        n = 2 * n / np.linalg.norm(n)
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        quad = np.stack([p1 + n, p2 + n, p2 - n, p1 - n]).astype(int)
        contour = quad.reshape((-1, 1, 2))
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        orient = "h" if abs(v[0]) > abs(v[1]) else "v"
        door_segs.append({"contour": contour, "center": (cx, cy), "orient": orient})
    return door_segs


def _merge_doors_by_center(doors, merge_dist=4):
    uniq = []
    for d in doors:
        c = np.array(d["center"], dtype=float)
        hit = False
        for u in uniq:
            uc = np.array(u["center"], dtype=float)
            if np.linalg.norm(c - uc) <= merge_dist:
                hit = True
                break
        if not hit:
            uniq.append(d)
    return uniq


def detect_all_doors(img):
    segs = detect_door_segments(img)
    blobs = detect_doors(img)
    all_doors = segs + blobs
    return _merge_doors_by_center(all_doors, merge_dist=4)


                                               


def _find_long_edges(box):
    segs = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    lens = [hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in segs]
    max_l = max(lens)
    return [segs[i] for i, l in enumerate(lens) if abs(l - max_l) < 1e-2 * max_l]


def _probe_room(m, n, h, w, room_contours, delta_max, step):
    for t in range(1, delta_max + 1, step):
        pt = (m + t * n).astype(int)
        x, y = int(pt[0]), int(pt[1])
        if h and (y < 0 or y >= h or x < 0 or x >= w):
            break
        hit = next(
            (idx for idx, c in enumerate(room_contours) if cv2.pointPolygonTest(c, (x, y), False) >= 0),
            None,
        )
        if hit is not None:
            return hit
    return None


def _get_hits(m, n, h, w, room_contours, delta_max, step):
    hits = []
    for sign in (+1, -1):
        hit = _probe_room(m, sign * n, h, w, room_contours, delta_max, step)
        hits.append(hit)
    return hits


def build_edges_by_doors(doors, room_contours, rooms, img_shape, delta_max=10, step=1):
    H, W = img_shape
    edges, entries = set(), set()

    for d in doors:
        cnt = d["contour"]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(int)
        long_edges = _find_long_edges(box)

        if not long_edges:
            continue

        center = np.array(rect[0], dtype=float)

        p0, p1 = long_edges[0]
        v = np.array(p1) - np.array(p0)
        if max(v) <= 4:
            continue
        n = np.array([-v[1], v[0]], dtype=float)
        if np.linalg.norm(n) == 0:
            continue
        n /= np.linalg.norm(n)

        local_delta = max(delta_max, int(max(rect[1]) // 2) + 8)

        a, b = _get_hits(center, n, H, W, room_contours, local_delta, step)

        if a is not None and b is not None and a != b:
            edges.add(tuple(sorted((rooms[a]["id"], rooms[b]["id"]))))
            continue

        found = set()
        has_outside = False
        for q0, q1 in long_edges:
            m_edge = (np.array(q0) + np.array(q1)) / 2.0
            aa, bb = _get_hits(m_edge, n, H, W, room_contours, local_delta, step)
            if aa is not None:
                found.add(aa)
            else:
                has_outside = True
            if bb is not None:
                found.add(bb)
            else:
                has_outside = True

        if len(found) >= 2:
            ids = [rooms[i]["id"] for i in sorted(found)[:2]]
            edges.add(tuple(sorted(ids)))
            continue
        elif len(found) == 1 and has_outside:
            idx = next(iter(found))
            entries.add(rooms[idx]["id"])
            continue

        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        radial_found = set()
        radial_outside = False
        for th in angles:
            dir_vec = np.array([np.cos(th), np.sin(th)], dtype=float)
            aa, bb = _get_hits(center, dir_vec, H, W, room_contours, local_delta * 2, step)
            if aa is not None:
                radial_found.add(aa)
            else:
                radial_outside = True
            if bb is not None:
                radial_found.add(bb)
            else:
                radial_outside = True

        if len(radial_found) >= 2:
            ids = [rooms[i]["id"] for i in sorted(radial_found)[:2]]
            edges.add(tuple(sorted(ids)))
        elif len(radial_found) == 1 and radial_outside:
            idx = next(iter(radial_found))
            entries.add(rooms[idx]["id"])

    return sorted(edges), sorted(entries)


                                                           


def check_topology(rooms, adjacency, entry_rooms):
    id2type = {r["id"]: r["type"] for r in rooms if "id" in r}
    id2sub = {r["id"]: r.get("subtype", "") for r in rooms if "id" in r}

    adj_set = {tuple(sorted(edge)) for edge in adjacency}

    missing: List[str] = []
    illegal: List[str] = []

    topo = TOPOLOGY_CONFIG
    required_rules = topo.get("required", []) or []
    forbidden_rules = topo.get("forbidden", []) or []
    entry_cfg = topo.get("entry", {}) or {}

                                              
    for rule in required_rules:
        from_type = rule.get("from_type")
        to_type = rule.get("to_type")
        from_sub = rule.get("from_subtype")
        to_sub = rule.get("to_subtype")
        if not from_type or not to_type:
            continue

        for rid, tp in id2type.items():
            if tp != from_type:
                continue
            if from_sub and id2sub.get(rid) != from_sub:
                continue

            has_conn = False
            for a, b in adj_set:
                if rid not in (a, b):
                    continue
                other = b if rid == a else a
                otp = id2type.get(other)
                osub = id2sub.get(other)
                if otp != to_type:
                    continue
                if to_sub and osub != to_sub:
                    continue
                has_conn = True
                break

            if not has_conn:
                missing.append(f"{rid}->{to_type}")

                                      
    def _match_endpoint(tp: Optional[str], sub: str, rule_type: Optional[str], rule_sub: Optional[str]) -> bool:
        if rule_type and tp != rule_type:
            return False
        if rule_sub and sub != rule_sub:
            return False
        return True

    for a, b in adj_set:
        ta, tb = id2type.get(a), id2type.get(b)
        sa, sb = id2sub.get(a, ""), id2sub.get(b, "")
        for rule in forbidden_rules:
            type_a = rule.get("type_a")
            type_b = rule.get("type_b")
            sub_a = rule.get("a_subtype")
            sub_b = rule.get("b_subtype")
            if not type_a or not type_b:
                continue

            if (
                _match_endpoint(ta, sa, type_a, sub_a)
                and _match_endpoint(tb, sb, type_b, sub_b)
            ) or (
                _match_endpoint(ta, sa, type_b, sub_b)
                and _match_endpoint(tb, sb, type_a, sub_a)
            ):
                illegal.append(str((a, b)))
                break

           
    require_entry = bool(entry_cfg.get("require", True))
    allowed_entry_types = set(entry_cfg.get("allowed_entry_room_types", []))

    if require_entry and not entry_rooms:
        if allowed_entry_types:
            types_str = "/".join(sorted(allowed_entry_types))
            missing.append(f"entry_to_{types_str}_missing")
        else:
            missing.append("entry_missing")

    illegal_entry = [
        rid
        for rid in entry_rooms
        if allowed_entry_types and id2type.get(rid) not in allowed_entry_types
    ]
    illegal.extend(illegal_entry)

    status = "ok" if not missing and not illegal else "fail"
    return {"missing_required": missing, "illegal_edges": illegal, "status": status}


                                                        


def contour_to_svg_path(contour, simplify_epsilon: float = 1.0) -> str:
    if contour is None or len(contour) == 0:
        return ""
    cnt = contour
    if simplify_epsilon and simplify_epsilon > 0:
        peri = cv2.arcLength(cnt, True)
        eps = simplify_epsilon if peri == 0 else simplify_epsilon
        cnt = cv2.approxPolyDP(cnt, eps, True)
    pts = cnt.reshape(-1, 2)
    coords = " ".join(f"{int(x)},{int(y)}" for x, y in pts)
    return f"M{coords} Z"


                                           


def calculate_total_area_and_masks(img, px_per_m2):
    non_white_pixels = count_non_white_pixels(img)
    total_area_m2 = non_white_pixels / px_per_m2
    masks = {}
    for cls, (lo, hi) in BGR_THRESHOLDS.items():
        if cls in {"wall", "door", "window"}:
            continue
        mask = create_mask(img, lo, hi)
        masks[cls] = mask
                                                                                         
                           
                                         
                             
                         
                                                 
    return total_area_m2, masks


def build_room_info(area_m2, cls, house_class, contour, rid: str):
    depth, width, ratio = calc_depth_width_ratio(contour)
    ratio_status = judge_ratio(ratio)

    room_info = {
        "type": cls,
        "id": rid,
        "subtype": None,
        "vector_path": contour_to_svg_path(contour, simplify_epsilon=1.0),
        "area_m2": round(area_m2, 2),
        "expected_area": None,
        "area_status": None,
        "depth": depth,
        "expected_depth": None,
        "depth_status": None,
        "width": width,
        "expected_width": None,
        "width_status": None,
        "depth_ratio": ratio,
        "expected_depth_ratio": None if DEPTH_RATIO_RANGE == (0.0, 0.0) else _format_min_max(DEPTH_RATIO_RANGE),
        "ratio_status": None if DEPTH_RATIO_RANGE == (0.0, 0.0) else ratio_status,
    }

    if house_class and house_class in _HOUSE_CLASS_BY_NAME:
        spec = _get_room_spec(house_class, cls)
        if isinstance(spec, dict) and (_parse_min_max(spec.get("area")) or ("min" in spec and "max" in spec)):
            apply_room_spec(room_info, spec)
    return room_info


def analyse_rooms(masks, px_per_m2, house_class):
    rooms, room_contours = [], []
    type_counter = defaultdict(int)
    for cls, mask in masks.items():
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area_px = cv2.contourArea(c)
            if area_px < 50:
                continue
            area_m2 = area_px / px_per_m2

            type_counter[cls] += 1
            rid = f"{cls}_{type_counter[cls]}"

            room_info = build_room_info(area_m2, cls, house_class, c, rid)
            rooms.append(room_info)
            room_contours.append(c)

                            
    bed_rooms = [r for r in rooms if r["type"] == "bedroom"]
    if bed_rooms and house_class in _HOUSE_CLASS_BY_NAME:
        master = max(bed_rooms, key=lambda r: r["area_m2"])
        bed_spec = _get_room_spec(house_class, "bedroom") or {}

        for r in bed_rooms:
            r["subtype"] = "master" if r is master else "secondary"
            subtype = r.get("subtype")
            sub_rule = bed_spec.get(subtype) if subtype else None
            if isinstance(sub_rule, dict):
                apply_room_spec(r, sub_rule)

    return rooms, room_contours


def analyse_vector_rooms(vector_data, px_per_m2, house_class):
    rooms, room_contours = [], []
    type_counter = defaultdict(int)

    for cls in ("living_room", "bedroom", "balcony", "kitchen", "bathroom"):
        for vector_path in vector_data.get(cls, []) or []:
            coordinates = re.findall(r"(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)", vector_path)
            if len(coordinates) < 3:
                continue

            contour = np.asarray(coordinates, dtype=np.float32).reshape(-1, 1, 2)
            area_px = cv2.contourArea(contour)
            if area_px < 50:
                continue

            type_counter[cls] += 1
            room_id = f"{cls}_{type_counter[cls]}"
            room = build_room_info(area_px / px_per_m2, cls, house_class, contour, room_id)
            depth, width, _ = calc_depth_width_ratio(contour)
            room.update(
                {
                    "vector_path": vector_path,
                    "area_m2": round(area_px / px_per_m2, 1),
                    "depth": round(depth, 1) if depth is not None else None,
                    "width": round(width, 1) if width is not None else None,
                    "depth_ratio": round(depth / width, 1) if depth is not None and width else None,
                }
            )

            spec = _get_room_spec(house_class, cls) if house_class else None
            if isinstance(spec, dict) and (_parse_min_max(spec.get("area")) or ("min" in spec and "max" in spec)):
                apply_room_spec(room, spec)

            rooms.append(room)
            room_contours.append(contour)

    bedrooms = [room for room in rooms if room["type"] == "bedroom"]
    if bedrooms and house_class in _HOUSE_CLASS_BY_NAME:
        master = max(bedrooms, key=lambda room: room["area_m2"])
        bedroom_spec = _get_room_spec(house_class, "bedroom") or {}
        for room in bedrooms:
            subtype = "master" if room is master else "secondary"
            room["subtype"] = subtype
            subtype_spec = bedroom_spec.get(subtype)
            if isinstance(subtype_spec, dict):
                apply_room_spec(room, subtype_spec)

    return rooms, room_contours


def update_bath_subtypes(rooms, adjacency, house_class):
    id2room = {r["id"]: r for r in rooms if "id" in r}
    mbr_ids = [r["id"] for r in rooms if r["type"] == "bedroom" and r.get("subtype") == "master"]

    master_bath_ids = set()
    for a, b in adjacency:
        if a in mbr_ids and id2room[b]["type"] == "bathroom":
            master_bath_ids.add(b)
        if b in mbr_ids and id2room[a]["type"] == "bathroom":
            master_bath_ids.add(a)

    if house_class not in _HOUSE_CLASS_BY_NAME:
        return

    bath_spec = _get_room_spec(house_class, "bathroom") or {}

    for rid, rm in id2room.items():
        if rm["type"] != "bathroom":
            continue
        subtype = "master" if rid in master_bath_ids else "other"
        sub_rule = bath_spec.get(subtype)
        rm["subtype"] = subtype
        if isinstance(sub_rule, dict):
            apply_room_spec(rm, sub_rule)


def update_balcony_subtypes(rooms, adjacency, house_class):
    if house_class not in _HOUSE_CLASS_BY_NAME:
        return

    balc_spec = _get_room_spec(house_class, "balcony")
    if not isinstance(balc_spec, dict):
        return

    id2room = {r["id"]: r for r in rooms if "id" in r}
    kitchen_ids = {r["id"] for r in rooms if r.get("type") == "kitchen" and "id" in r}
    balcony_ids = {r["id"] for r in rooms if r.get("type") == "balcony" and "id" in r}

    if not balcony_ids:
        return

    if _parse_min_max(balc_spec.get("area")) or ("min" in balc_spec and "max" in balc_spec):
        for rid in balcony_ids:
            rm = id2room.get(rid)
            if rm:
                apply_room_spec(rm, balc_spec)
        return

    main_rule = balc_spec.get("main")
    service_rule = balc_spec.get("service")
    main_area_rng = (
        (_parse_min_max(main_rule.get("area")) or _parse_min_max(main_rule)) if isinstance(main_rule, dict) else None
    )
    service_area_rng = (
        (_parse_min_max(service_rule.get("area")) or _parse_min_max(service_rule))
        if isinstance(service_rule, dict)
        else None
    )

    kitchen_adjacent = set()
    for a, b in adjacency:
        if a in kitchen_ids and b in balcony_ids:
            kitchen_adjacent.add(b)
        elif b in kitchen_ids and a in balcony_ids:
            kitchen_adjacent.add(a)

    service_ids = set()
    if kitchen_adjacent:
        if len(kitchen_adjacent) >= 2:
            candidate = min(
                kitchen_adjacent,
                key=lambda rid: id2room.get(rid, {}).get("area_m2", float("inf")),
            )
            service_ids.add(candidate)
        else:
            rid = next(iter(kitchen_adjacent))
            rm = id2room.get(rid)
            if rm and service_area_rng and rm.get("area_m2") is not None and rm["area_m2"] > service_area_rng[1]:
                pass
            else:
                service_ids.add(rid)

                                            
    if service_area_rng and main_area_rng and service_area_rng[1] <= main_area_rng[0]:
        for rid in balcony_ids:
            if rid in service_ids or rid in kitchen_adjacent:
                continue
            rm = id2room.get(rid)
            if rm and rm.get("area_m2") is not None and rm["area_m2"] <= service_area_rng[1]:
                service_ids.add(rid)

    for rid in balcony_ids:
        rm = id2room.get(rid)
        if not rm:
            continue
        subtype = "service" if rid in service_ids else "main"
        rm["subtype"] = subtype
        sub_rule = balc_spec.get(subtype)
        if isinstance(sub_rule, dict):
            apply_room_spec(rm, sub_rule)

def check_mask_coverage(mask1_path, mask2_path, white_threshold=245):
                           
                                                        
                                                        
    img1 = cv2.imdecode(np.fromfile(str(mask1_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if isinstance(mask2_path, np.ndarray):
        if mask2_path.ndim == 2:
            img2 = mask2_path
        else:
            img2 = cv2.cvtColor(mask2_path, cv2.COLOR_BGR2GRAY)
    else:
        img2 = cv2.imdecode(np.fromfile(str(mask2_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

              
    if img1 is None:
        raise ValueError(f"Unable to read image: {mask1_path}")
    if img2 is None:
        raise ValueError(f"Unable to read image: {mask2_path}")

                                       
    if img1.shape != img2.shape:
                                        
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)

                                                      
                                                   
    _, mask1 = cv2.threshold(img1, white_threshold, 255, cv2.THRESH_BINARY_INV)
    _, mask2 = cv2.threshold(img2, white_threshold, 255, cv2.THRESH_BINARY_INV)

                                        
         
                         
                                 
    non_zero_2 = np.nonzero(mask2)                                 
    if len(non_zero_2[0]) == 0:
                                     
        return True

                         
    coverage_check = mask1[non_zero_2]                           
    all_covered = np.all(coverage_check == 255)                  

                      
                          
    debug_img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if not all_covered:
                     
        uncovered_idx = np.where(coverage_check != 255)[0]
        uncovered_rows = non_zero_2[0][uncovered_idx]
        uncovered_cols = non_zero_2[1][uncovered_idx]
                 
        debug_img[uncovered_rows, uncovered_cols] = [0, 0, 255]         
                                                                          

    return all_covered

def room_num(image):
    lower_gray_BGR, upper_gray_BGR = BGR_THRESHOLDS["living_room"]
    lower_yellow_BGR, upper_yellow_BGR = BGR_THRESHOLDS["bedroom"]
    lower_blue_BGR, upper_blue_BGR = BGR_THRESHOLDS["bathroom"]
    lower_pink_BGR, upper_pink_BGR = BGR_THRESHOLDS["kitchen"]
    lower_green_BGR, upper_green_BGR = BGR_THRESHOLDS["balcony"]
    def mask_room_choose(mask_room):
        contours, _ = cv2.findContours(mask_room, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result_num = 0
        center = []
        for i, contour in enumerate(contours):
                         
            contour_area = cv2.contourArea(contour)
                                    
            if contour_area < 200:
                continue
            result_num += 1
        return result_num
    mask_livingroom = cv2.inRange(image, np.array(lower_gray_BGR), np.array(upper_gray_BGR))
    num_livingroom = mask_room_choose(mask_livingroom)
    mask_toilet = cv2.inRange(image, np.array(lower_blue_BGR), np.array(upper_blue_BGR))
    num_toilet = mask_room_choose(mask_toilet)
    mask_bedroom = cv2.inRange(image, np.array(lower_yellow_BGR), np.array(upper_yellow_BGR))
    num_bedroom = mask_room_choose(mask_bedroom)
    mask_kitchen = cv2.inRange(image, np.array(lower_pink_BGR), np.array(upper_pink_BGR))
    num_kitchen = mask_room_choose(mask_kitchen)
    mask_balcony = cv2.inRange(image, np.array(lower_green_BGR), np.array(upper_green_BGR))
    num_balcony = mask_room_choose(mask_balcony)

    room_n = [num_livingroom, num_bedroom, num_toilet, num_kitchen, num_balcony]

    return str(room_n)

def analyse_image(
    image_path: Optional[Path],
    original_json,
    edge_path: Optional[Path] = None,
    vector_data: Optional[Dict[str, Any]] = None,
    image: Optional[np.ndarray] = None,
    image_name: Optional[str] = None,
):
                                      
    if image is None:
        if image_path is None:
            raise ValueError("Provide either an image array or an image path.")
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_path)
    else:
        img = np.asarray(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("The rendered image must be a grayscale or BGR image.")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if image_name:
        img_name = Path(image_name).name
    elif image_path is not None:
        img_name = Path(image_path).name
    elif vector_data and vector_data.get("Pic"):
        img_name = Path(vector_data["Pic"]).name
    else:
        img_name = "floorplan.png"
    img_zero = np.zeros_like(img)
                                                                   

    px_per_m2 = PIXELS_PER_M ** 2

    total_area_m2, masks = calculate_total_area_and_masks(img, px_per_m2)
    total_area_m2 = round(total_area_m2, 2)
    house_class = classify_house(total_area_m2)
    if vector_data is None:
        rooms, room_contours = analyse_rooms(masks, px_per_m2, house_class)
    else:
        rooms, room_contours = analyse_vector_rooms(vector_data, px_per_m2, house_class)

    room_number_original = original_json['room_number_original']
    room_number_change = room_num(img)
    if room_number_original == room_number_change:
        room_number_judge = "ok"
    else:
        room_number_judge = "fail"

    try:
        firetest = max_distance_class(img, Path(img_name)) / PIXELS_PER_M
        firetest = float(format(firetest, ".2f"))
        max_firetest = FIRE_RANGE[1]
        if max_firetest > 5.0 and firetest > max_firetest:
            firetest_judge = "fail"
        else:
            firetest_judge = "ok"
    except:
        firetest = None
        firetest_judge = "fail"

    if edge_path is None:
        is_covered = None
    else:
        try:
            is_covered = check_mask_coverage(edge_path, img)
        except Exception:
            is_covered = False

    doors = detect_all_doors(img)
    H, W = img.shape[:2]
    edges, entry = build_edges_by_doors(doors, room_contours, rooms, (H, W))

                
    update_bath_subtypes(rooms, edges, house_class)
    update_balcony_subtypes(rooms, edges, house_class)

    for r in rooms:
        if r.get("subtype") is None:
            r.pop("subtype", None)

    result = {
        "file": img_name,
        "total_area_m2": total_area_m2,
        "house_class": house_class or "out_of_range",
        "Fire_safety_distance": firetest,
        "Fire_safety_judge": firetest_judge,
        "Plot_test": str(is_covered) if is_covered is not None else None,
        "room_number_original": room_number_original,
        "room_number_change": room_number_change,
        "room_number_judge": room_number_judge,
        "rooms": rooms,
    }

    if edges:
        result["adjacency"] = edges
    if entry:
        result["entry_rooms"] = entry

    if "adjacency" in result:
        topo = check_topology(rooms, result["adjacency"], result.get("entry_rooms", []))
        result["graph_check"] = topo
    return result
