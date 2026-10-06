from __future__ import annotations
import ast
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from eppy.modeleditor import IDF


EP_VERSION = "23.1"
SCALE = 1.0 / 12.0
ZONE_HEIGHT = 3.0
DOOR_HEIGHT = 2.1
WIN_HEIGHT  = 1.2
WIN_SILL    = 0.9
MATCH_TOL   = 0.15
ADJACENCY_TOL = 0.15
ADJACENCY_ANGLE_TOL = 0.02
MIN_ADJACENT_WALL = 0.05
MIN_WALL_SEGMENT = 0.05
EDGE_MARGIN = 0.18
MIN_OPENING = 0.35
Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

OPAQUE_U_VALUE = 0.3
WINDOW_U = 1.5
WINDOW_SHGC = 0.6
WINDOW_TVIS = 0.70
INTERIOR_WALL_CONDUCTIVITY = 1.80
HVAC_COP = 3.0
AIR_CHANGES_PER_HOUR = 0.5
INCLUDE_INTERNAL_LOADS = True

LIGHT_RADIANT_FRACTION = 0.42
EQUIP_RADIANT_FRACTION = 0.30
OCCUPANT_DENSITY = 0.03
ACTIVITY_LEVEL_W_PER_PERSON = 90.0
EQUIPMENT_POWER_DENSITY_W_M2 = 6.7
LIGHTING_POWER_DENSITY_W_M2 = 6.5

LIGHTING_SCHEDULE = [
    0.011316031, 0.011316031, 0.011316031, 0.011316031,
    0.033948092, 0.073554200, 0.079212150, 0.073554200,
    0.033948092, 0.022632061, 0.022632061, 0.022632061,
    0.022632061, 0.022632061, 0.022632061, 0.039606108,
    0.079212150, 0.113160307, 0.152766415, 0.181056492,
    0.181056492, 0.124476338, 0.067896184, 0.028290077,
]

EQUIPMENT_SCHEDULE = [
    0.45, 0.41, 0.39, 0.38, 0.38, 0.43, 0.54, 0.65,
    0.66, 0.67, 0.69, 0.70, 0.69, 0.66, 0.65, 0.68,
    0.80, 1.00, 1.00, 0.93, 0.89, 0.85, 0.71, 0.58,
]

OCCUPANCY_SCHEDULE = [
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.85,
    0.39, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
    0.30, 0.52, 0.87, 0.87, 0.87, 1.00, 1.00, 1.00,
]

HEATING_SETPOINT = 20.0
COOLING_SETPOINT = 25.0
BALCONY_HAS_LOADS = False


def create_blank_idf(idd_path: Path, ep_version: str) -> IDF:
    IDF.setiddname(str(idd_path))
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".idf", delete=False, encoding="utf-8")
    tmp.write(f"Version,{ep_version};\n")
    tmp.flush()
    tmp.close()
    return IDF(tmp.name)


def safe_add(idf: IDF, objtype: str, **fields):
    try:
        return idf.newidfobject(objtype, **fields)
    except Exception:
        return None


def load_paths_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value).strip()
    if not s or s in ("None", "[]", "''", '""'):
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            clean = s.strip("[]")
            parts = re.split(r"""\s*['"]\s*,\s*['"]\s*""", clean.strip("'\""))
            return [p for p in parts if p.strip()]
    return [s]


def parse_svg_path(path: str) -> List[Vec2]:
    s = str(path).strip()
    s = re.sub(r"[mMlLzZ]\s*", "", s)
    s = s.replace("Z", "").replace("z", "").strip()
    tokens = [t for t in s.split() if t.strip()]

    pts: List[Vec2] = []
    for tok in tokens:
        if "," in tok:
            xs, ys = tok.split(",", 1)
        else:
            parts = tok.split()
            if len(parts) < 2:
                continue
            xs, ys = parts[0], parts[1]
        try:
            pts.append((float(xs) * SCALE, float(ys) * SCALE))
        except Exception:
            continue

    if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
        pts = pts[:-1]

    cleaned: List[Vec2] = []
    for p in pts:
        if not cleaned or abs(cleaned[-1][0] - p[0]) > 1e-9 or abs(cleaned[-1][1] - p[1]) > 1e-9:
            cleaned.append(p)
    return cleaned


def polygon_area_signed(poly: List[Vec2]) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def polygon_area(poly: List[Vec2]) -> float:
    return abs(polygon_area_signed(poly))


def ensure_ccw(poly: List[Vec2]) -> List[Vec2]:
    return poly if polygon_area_signed(poly) > 0 else list(reversed(poly))


def round_pt(p: Vec2, nd: int = 4) -> Vec2:
    return (round(p[0], nd), round(p[1], nd))


def edge_key(a: Vec2, b: Vec2, nd: int = 4) -> Tuple[Vec2, Vec2]:
    ra, rb = round_pt(a, nd), round_pt(b, nd)
    return tuple(sorted((ra, rb)))


def dist_point_to_segment(p: Vec2, a: Vec2, b: Vec2) -> Tuple[float, float]:
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv < 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    t = (wx * vx + wy * vy) / vv
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * vx, ay + t * vy
    return math.hypot(px - cx, py - cy), t


def project_param_on_line(p: Vec2, a: Vec2, b: Vec2) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    vv = vx * vx + vy * vy
    if vv < 1e-12:
        return 0.0
    return ((px - ax) * vx + (py - ay) * vy) / vv


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def extract_zones(data: Dict[str, Any]) -> Dict[str, List[Vec2]]:
    zones: Dict[str, List[Vec2]] = {}
    for k, v in data.items():
        kl = str(k).lower()
        if any(t in kl for t in ["livingroom", "bedroom", "toilet", "kitchen", "balcony"]):
            paths = load_paths_field(v)
            if not paths:
                continue
            poly = parse_svg_path(paths[0])
            if len(poly) >= 3:
                zones[str(k)] = ensure_ccw(poly)
    return zones


def extract_segments(data: Dict[str, Any], key_substring: str) -> List[Tuple[Vec2, Vec2]]:
    segs: List[Tuple[Vec2, Vec2]] = []
    for k, v in data.items():
        if key_substring not in str(k).lower():
            continue
        for p in load_paths_field(v):
            pts = parse_svg_path(p)
            uniq: List[Vec2] = []
            for pt in pts:
                if not any(math.hypot(pt[0] - q[0], pt[1] - q[1]) < 1e-9 for q in uniq):
                    uniq.append(pt)
            if len(uniq) >= 2:
                segs.append((uniq[0], uniq[1]))
    return segs


def build_adjacency(zones: Dict[str, List[Vec2]]) -> Dict[Tuple[str, int], Tuple[str, int]]:
    edge_occ: Dict[Tuple[Vec2, Vec2], List[Tuple[str, int]]] = {}
    for zn, poly in zones.items():
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            edge_occ.setdefault(edge_key(a, b), []).append((zn, i))

    adj: Dict[Tuple[str, int], Tuple[str, int]] = {}
    for occs in edge_occ.values():
        if len(occs) == 2:
            (z1, i1), (z2, i2) = occs
            adj[(z1, i1)] = (z2, i2)
            adj[(z2, i2)] = (z1, i1)
    return adj


def point_on_segment(a: Vec2, b: Vec2, t: float) -> Vec2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def segment_length(a: Vec2, b: Vec2) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def normalized_interval(a: float, b: float) -> Tuple[float, float]:
    lo, hi = sorted((max(0.0, min(1.0, a)), max(0.0, min(1.0, b))))
    return lo, hi


def add_split_point(points: List[float], value: float, tol: float = 1e-7) -> None:
    value = max(0.0, min(1.0, value))
    if not any(abs(value - p) <= tol for p in points):
        points.append(value)


def edge_overlap_record(edge_a: Dict[str, Any], edge_b: Dict[str, Any], pair_id: int) -> Optional[Dict[str, Any]]:
    a, b = edge_a["a"], edge_a["b"]
    c, d = edge_b["a"], edge_b["b"]
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = d[0] - c[0], d[1] - c[1]
    len_a = math.hypot(ux, uy)
    len_b = math.hypot(vx, vy)
    if len_a < 1e-9 or len_b < 1e-9:
        return None

    u = (ux / len_a, uy / len_a)
    v = (vx / len_b, vy / len_b)
    dot = u[0] * v[0] + u[1] * v[1]

    if abs(dot) < 0.65:
        return None

    cross_norm = abs(ux * vy - uy * vx) / (len_a * len_b)
    if cross_norm > ADJACENCY_ANGLE_TOL:
        return None

    def signed_line_dist(p: Vec2) -> float:
        return ((p[0] - a[0]) * uy - (p[1] - a[1]) * ux) / len_a

    if max(abs(signed_line_dist(c)), abs(signed_line_dist(d))) > ADJACENCY_TOL:
        return None

    proj_c = ((c[0] - a[0]) * u[0] + (c[1] - a[1]) * u[1])
    proj_d = ((d[0] - a[0]) * u[0] + (d[1] - a[1]) * u[1])
    overlap_0 = max(0.0, min(proj_c, proj_d))
    overlap_1 = min(len_a, max(proj_c, proj_d))
    if overlap_1 - overlap_0 < MIN_ADJACENT_WALL:
        return None

    q0 = (a[0] + u[0] * overlap_0, a[1] + u[1] * overlap_0)
    q1 = (a[0] + u[0] * overlap_1, a[1] + u[1] * overlap_1)

    def project_t_on_edge_b(p: Vec2) -> float:
        dist = ((p[0] - c[0]) * v[0] + (p[1] - c[1]) * v[1])
        return dist / len_b

    a_t0, a_t1 = normalized_interval(overlap_0 / len_a, overlap_1 / len_a)
    b_t0, b_t1 = normalized_interval(project_t_on_edge_b(q0), project_t_on_edge_b(q1))
    if (a_t1 - a_t0) * len_a < MIN_ADJACENT_WALL or (b_t1 - b_t0) * len_b < MIN_ADJACENT_WALL:
        return None

    return {
        "id": pair_id,
        "edge_a_key": edge_a["key"],
        "edge_b_key": edge_b["key"],
        "edge_a_t0": a_t0,
        "edge_a_t1": a_t1,
        "edge_b_t0": b_t0,
        "edge_b_t1": b_t1,
        "common_a": q0,
        "common_b": q1,
        "same_direction": dot >= 0.0,
    }


def make_edge_list(zones: Dict[str, List[Vec2]]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for zk, poly in zones.items():
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            edges.append({"key": (zk, i), "zone_key": zk, "edge_idx": i, "a": a, "b": b})
    return edges


def build_overlap_records(zones: Dict[str, List[Vec2]]) -> List[Dict[str, Any]]:
    edges = make_edge_list(zones)
    records: List[Dict[str, Any]] = []
    pair_id = 1
    for i, edge_a in enumerate(edges):
        for edge_b in edges[i + 1:]:
            if edge_a["zone_key"] == edge_b["zone_key"]:
                continue
            rec = edge_overlap_record(edge_a, edge_b, pair_id)
            if rec is None:
                continue
            records.append(rec)
            pair_id += 1
    return records


def interval_record_for_edge(records: List[Dict[str, Any]], edge_key: Tuple[str, int], mid_t: float) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_len = -1.0
    for rec in records:
        if rec["edge_a_key"] == edge_key:
            t0, t1 = rec["edge_a_t0"], rec["edge_a_t1"]
        elif rec["edge_b_key"] == edge_key:
            t0, t1 = rec["edge_b_t0"], rec["edge_b_t1"]
        else:
            continue
        if t0 - 1e-7 <= mid_t <= t1 + 1e-7:
            span = t1 - t0
            if span > best_len:
                best = rec
                best_len = span
    return best


def common_point_for_record(rec: Dict[str, Any], frac: float) -> Vec2:
    frac = max(0.0, min(1.0, frac))
    a, b = rec["common_a"], rec["common_b"]
    return (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)


def record_frac_for_edge(rec: Dict[str, Any], edge_key: Tuple[str, int], t: float) -> float:
    if rec["edge_a_key"] == edge_key:
        t0, t1 = rec["edge_a_t0"], rec["edge_a_t1"]
        return (t - t0) / max(t1 - t0, 1e-9)

    t0, t1 = rec["edge_b_t0"], rec["edge_b_t1"]
    frac = (t - t0) / max(t1 - t0, 1e-9)
    if not rec["same_direction"]:
        frac = 1.0 - frac
    return frac


def build_wall_segments(zones: Dict[str, List[Vec2]], zone_names_map: Dict[str, str]) -> List[Dict[str, Any]]:
    edges = make_edge_list(zones)
    edge_map = {edge["key"]: edge for edge in edges}
    records = build_overlap_records(zones)

    split_points: Dict[Tuple[str, int], List[float]] = {edge["key"]: [0.0, 1.0] for edge in edges}
    for rec in records:
        add_split_point(split_points[rec["edge_a_key"]], rec["edge_a_t0"])
        add_split_point(split_points[rec["edge_a_key"]], rec["edge_a_t1"])
        add_split_point(split_points[rec["edge_b_key"]], rec["edge_b_t0"])
        add_split_point(split_points[rec["edge_b_key"]], rec["edge_b_t1"])

    wall_segments: List[Dict[str, Any]] = []
    pair_groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}

    for edge in edges:
        edge_key = edge["key"]
        points = sorted(split_points[edge_key])
        intervals = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
        intervals = [(t0, t1) for t0, t1 in intervals if (t1 - t0) * segment_length(edge["a"], edge["b"]) >= MIN_WALL_SEGMENT]
        multiple_segments = len(intervals) > 1
        local_idx = 1

        for t0, t1 in intervals:
            mid_t = (t0 + t1) * 0.5
            rec = interval_record_for_edge(records, edge_key, mid_t)
            base_name = f"WALL_{zone_names_map[edge['zone_key']]}_{edge['edge_idx'] + 1}"
            name = f"{base_name}_S{local_idx}" if multiple_segments else base_name
            local_idx += 1

            if rec is None:
                seg_a = point_on_segment(edge["a"], edge["b"], t0)
                seg_b = point_on_segment(edge["a"], edge["b"], t1)
                wall = {
                    "name": name,
                    "zone_key": edge["zone_key"],
                    "zone_name": zone_names_map[edge["zone_key"]],
                    "edge_idx": edge["edge_idx"],
                    "a": seg_a,
                    "b": seg_b,
                    "bc": "Outdoors",
                    "bc_obj": "",
                    "sun": "SunExposed",
                    "wind": "WindExposed",
                    "construction": "CONST_EWALL",
                    "pair_token": None,
                }
            else:
                f0 = record_frac_for_edge(rec, edge_key, t0)
                f1 = record_frac_for_edge(rec, edge_key, t1)
                seg_a = common_point_for_record(rec, f0)
                seg_b = common_point_for_record(rec, f1)
                pair_token = (rec["id"], round(min(f0, f1) * 1000000), round(max(f0, f1) * 1000000))
                wall = {
                    "name": name,
                    "zone_key": edge["zone_key"],
                    "zone_name": zone_names_map[edge["zone_key"]],
                    "edge_idx": edge["edge_idx"],
                    "a": seg_a,
                    "b": seg_b,
                    "bc": "Surface",
                    "bc_obj": "",
                    "sun": "NoSun",
                    "wind": "NoWind",
                    "construction": "CONST_IWALL",
                    "pair_token": pair_token,
                }
                pair_groups.setdefault(pair_token, []).append(wall)

            wall_segments.append(wall)

    for walls in pair_groups.values():
        if len(walls) != 2:
            for wall in walls:
                wall.update({
                    "bc": "Outdoors",
                    "bc_obj": "",
                    "sun": "SunExposed",
                    "wind": "WindExposed",
                    "construction": "CONST_EWALL",
                    "pair_token": None,
                })
            continue
        a, b = walls
        a["bc_obj"] = b["name"]
        b["bc_obj"] = a["name"]
        a["paired_name"] = b["name"]
        b["paired_name"] = a["name"]
        a["paired_zone_key"] = b["zone_key"]
        b["paired_zone_key"] = a["zone_key"]
        a["paired_a"], a["paired_b"] = b["a"], b["b"]
        b["paired_a"], b["paired_b"] = a["a"], a["b"]

    return wall_segments


def make_opening_vertices(seg: Tuple[Vec2, Vec2], wall_a: Vec2, wall_b: Vec2, sill: float, height: float) -> Optional[List[Vec3]]:
    (p1, p2) = seg
    ax, ay = wall_a
    bx, by = wall_b
    L = math.hypot(bx - ax, by - ay)
    if L < 0.80:
        return None

    t1 = project_param_on_line(p1, wall_a, wall_b)
    t2 = project_param_on_line(p2, wall_a, wall_b)
    if t2 < t1:
        t1, t2 = t2, t1

    t1 = max(0.0, min(1.0, t1))
    t2 = max(0.0, min(1.0, t2))

    m = EDGE_MARGIN / L
    t1 = max(m, min(1.0 - m, t1))
    t2 = max(m, min(1.0 - m, t2))

    if (t2 - t1) * L < MIN_OPENING:
        return None

    q1 = (ax + (bx - ax) * t1, ay + (by - ay) * t1)
    q2 = (ax + (bx - ax) * t2, ay + (by - ay) * t2)

    sill = max(0.0, sill)
    max_h = max(0.2, ZONE_HEIGHT - 0.05 - sill)
    h = min(height, max_h)
    if h < 0.2:
        return None

    z0, z1 = sill, sill + h
    return [(q1[0], q1[1], z0), (q2[0], q2[1], z0), (q2[0], q2[1], z1), (q1[0], q1[1], z1)]


def find_host_wall(seg: Tuple[Vec2, Vec2], wall_index: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    (p1, p2) = seg
    mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    best = None
    best_d = 1e9
    for w in wall_index:
        d, _ = dist_point_to_segment(mid, w["a"], w["b"])
        if d < best_d:
            best_d = d
            best = w
    if best is None or best_d > MATCH_TOL:
        return None
    return best


def add_vertices_fields(obj, verts: List[Vec3]) -> None:
    obj.Number_of_Vertices = len(verts)
    for i, (x, y, z) in enumerate(verts, start=1):
        setattr(obj, f"Vertex_{i}_Xcoordinate", x)
        setattr(obj, f"Vertex_{i}_Ycoordinate", y)
        setattr(obj, f"Vertex_{i}_Zcoordinate", z)


def add_output_variables(idf: IDF) -> None:

    meter_vars = [
        "Cooling:Electricity",
        "Heating:Electricity",
        "Heating:NaturalGas",
        "Fans:Electricity",
        "Pumps:Electricity",
    ]
    for mv in meter_vars:
        safe_add(idf, "Output:Meter:MeterFileOnly", Key_Name=mv, Reporting_Frequency="Hourly")

    vars_hourly = [
        "Zone Lights Electricity Energy",
        "Zone Mean Air Temperature",
        "Zone Ideal Loads Zone Total Heating Energy",
        "Zone Ideal Loads Zone Total Cooling Energy",
    ]
    for vname in vars_hourly:
        safe_add(idf, "Output:Variable", Key_Value="*", Variable_Name=vname, Reporting_Frequency="Hourly")


def clean_zonecontrol_thermostat_blocks(idf_path: Path) -> None:
    text = idf_path.read_text(encoding="utf-8", errors="ignore")
    parts = text.split(";")
    cleaned_parts: List[str] = []

    def strip_comment(line: str) -> str:
        return line.split("!-", 1)[0].strip()

    for part in parts:
        if not part.strip():
            cleaned_parts.append(part)
            continue
        if not part.lstrip().upper().startswith("ZONECONTROL:THERMOSTAT"):
            cleaned_parts.append(part)
            continue

        fields: List[str] = []
        for line in part.splitlines():
            no_comment = strip_comment(line)
            if not no_comment:
                continue
            for item in no_comment.split(","):
                item = item.strip()
                if item:
                    fields.append(item)

        if len(fields) >= 6 and fields[0].upper() == "ZONECONTROL:THERMOSTAT":
            obj = [
                "ZoneControl:Thermostat,",
                f"  {fields[1]},",
                f"  {fields[2]},",
                f"  {fields[3]},",
                f"  {fields[4]},",
                f"  {fields[5]}",
            ]
            cleaned_parts.append("\n".join(obj))
        else:
            cleaned_parts.append(part)

    with open(idf_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(";".join(cleaned_parts))


def append_schedules_raw(out_idf_path: Path) -> None:
    raw: List[str] = []
    raw.append("")
    raw.append("! ==== Added by script: residential schedules (raw) ====")

    def add_schedule_compact(name: str, values: List[float]) -> None:
        if len(values) != 24:
            raise ValueError(f"{name} must have 24 hourly values.")
        raw.append("Schedule:Compact,")
        raw.append(f"  {name},")
        raw.append("  AnyNumber,")
        raw.append("  Through: 12/31,")
        raw.append("  For: AllDays,")
        for h, value in enumerate(values, start=1):
            end_hour = f"{h:02d}:00" if h < 24 else "24:00"
            term = ";" if h == 24 else ","
            raw.append(f"  Until: {end_hour}, {value:.9g}{term}")
        raw.append("")

    add_schedule_compact("LightsSch_Prototype", LIGHTING_SCHEDULE)
    add_schedule_compact("EquipSch_Prototype", EQUIPMENT_SCHEDULE)
    add_schedule_compact("OccSch_Prototype", OCCUPANCY_SCHEDULE)


    add_schedule_compact("HVACAvail_Bedroom", [1]*10 + [0]*9 + [1]*5)


    add_schedule_compact("HVACAvail_Other", [0]*7 + [1]*16 + [0])

    raw.append("Schedule:Constant,")
    raw.append("  InfilSch_AlwaysOn,")
    raw.append("  Fraction,")
    raw.append("  1;")
    raw.append("")

    raw.append("Schedule:Constant,")
    raw.append("  Activity_90W,")
    raw.append("  AnyNumber,")
    raw.append(f"  {ACTIVITY_LEVEL_W_PER_PERSON:.3f};")
    raw.append("")

    with open(out_idf_path, "a", encoding="utf-8") as f:
        f.write("\n".join(raw) + "\n")


def append_loads_raw(out_idf_path: Path,
                     zone_names: List[str],
                     zone_areas: Dict[str, float]) -> None:
    raw: List[str] = []
    raw.append("")
    raw.append("! ==== Added by script: residential Lights/People/Equip/Infiltration (raw) ====")

    for zn in zone_names:
        zl = zn.lower()
        area = max(0.01, float(zone_areas[zn]))

        if "kitchen" in zl:
            ztype = "kitchen"
        elif "toilet" in zl or "bath" in zl or "wc" in zl:
            ztype = "toilet"
        elif "bedroom" in zl:
            ztype = "bedroom"
        elif "balcony" in zl:
            ztype = "balcony"
        else:
            ztype = "livingroom"

        conditioned_zone = not (ztype == "balcony" and not BALCONY_HAS_LOADS)

        if conditioned_zone:
            lighting_level = float(LIGHTING_POWER_DENSITY_W_M2) * area

            raw.append("Lights,")
            raw.append(f"  Lights_{zn},")
            raw.append(f"  {zn},")
            raw.append("  LightsSch_Prototype,")
            raw.append("  LightingLevel,")
            raw.append(f"  {lighting_level:.3f},")
            raw.append("  ,")
            raw.append("  ,")
            raw.append("  0.00,")
            raw.append(f"  {LIGHT_RADIANT_FRACTION:.3f},")
            raw.append("  0.180,")
            raw.append("  0.000,")
            raw.append("  GeneralLights;")
            raw.append("")

        if not INCLUDE_INTERNAL_LOADS:
            continue

        if conditioned_zone:
            raw.append("People,")
            raw.append(f"  People_{zn},")
            raw.append(f"  {zn},")
            raw.append("  OccSch_Prototype,")
            raw.append("  People/Area,")
            raw.append("  ,")
            raw.append(f"  {OCCUPANT_DENSITY:.5f},")
            raw.append("  ,")
            raw.append("  0.300,")
            raw.append("  ,")
            raw.append("  Activity_90W;")
            raw.append("")

            equip_level = float(EQUIPMENT_POWER_DENSITY_W_M2) * area
            if equip_level > 0.0:
                raw.append("ElectricEquipment,")
                raw.append(f"  Equip_{zn},")
                raw.append(f"  {zn},")
                raw.append("  EquipSch_Prototype,")
                raw.append("  EquipmentLevel,")
                raw.append(f"  {equip_level:.3f},")
                raw.append("  ,")
                raw.append("  ,")
                raw.append("  0.000,")
                raw.append(f"  {EQUIP_RADIANT_FRACTION:.3f},")
                raw.append("  0.000,")
                raw.append("  PlugLoads;")
                raw.append("")

        raw.append("ZoneInfiltration:DesignFlowRate,")
        raw.append(f"  Infil_{zn},")
        raw.append(f"  {zn},")
        raw.append("  InfilSch_AlwaysOn,")
        raw.append("  AirChanges/Hour,")
        raw.append("  ,")
        raw.append("  ,")
        raw.append("  ,")
        raw.append(f"  {AIR_CHANGES_PER_HOUR:.3f},")
        raw.append("  1.0,")
        raw.append("  0.0,")
        raw.append("  0.0,")
        raw.append("  0.0;")
        raw.append("")

    with open(out_idf_path, "a", encoding="utf-8") as f:
        f.write("\n".join(raw) + "\n")


def json_to_idf(json_path: Path, idd_path: Path, out_idf: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    zones_xy = extract_zones(data)
    if not zones_xy:
        raise ValueError("No room outlines were parsed. Expected keys containing livingroom, bedroom, toilet, kitchen, or balcony.")

    zone_keys = sorted(zones_xy.keys())
    zone_names_map = {zk: safe_name(zk) for zk in zone_keys}
    zone_names_list = [zone_names_map[zk] for zk in zone_keys]
    zones_area_map = {zone_names_map[zk]: polygon_area(zones_xy[zk]) for zk in zone_keys}
    wall_segments = build_wall_segments(zones_xy, zone_names_map)
    wall_index: List[Dict[str, Any]] = [dict(wall) for wall in wall_segments]

    door_segs = extract_segments(data, "door")
    win_segs  = extract_segments(data, "window")

    door_hosts = [(seg, find_host_wall(seg, wall_index)) for seg in door_segs]
    door_hosts = [(seg, host) for seg, host in door_hosts if host is not None]

    win_hosts = [(seg, find_host_wall(seg, wall_index)) for seg in win_segs]
    win_hosts = [(seg, host) for seg, host in win_hosts if host is not None]

    idf = create_blank_idf(idd_path, EP_VERSION)
    if idf.idfobjects.get("VERSION"):
        idf.idfobjects["VERSION"][0].Version_Identifier = EP_VERSION


    safe_add(idf, "SimulationControl",
             Do_Zone_Sizing_Calculation="Yes",
             Do_System_Sizing_Calculation="Yes",
             Do_Plant_Sizing_Calculation="Yes",
             Run_Simulation_for_Sizing_Periods="No",
             Run_Simulation_for_Weather_File_Run_Periods="Yes")


    safe_add(idf, "Building",
             Name="Building",
             North_Axis=0.0,
             Terrain="Suburbs",
             Loads_Convergence_Tolerance_Value=0.04,
             Temperature_Convergence_Tolerance_Value=0.25,
             Solar_Distribution="FullExterior",
             Maximum_Number_of_Warmup_Days=25,
             Minimum_Number_of_Warmup_Days=6)


    safe_add(idf, "GlobalGeometryRules",
             Starting_Vertex_Position="LowerLeftCorner",
             Vertex_Entry_Direction="Counterclockwise",
             Coordinate_System="Relative")

    safe_add(idf, "Timestep", Number_of_Timesteps_per_Hour=6)


    safe_add(idf, "Site:GroundReflectance",
             January_Ground_Reflectance=0.2,
             February_Ground_Reflectance=0.2,
             March_Ground_Reflectance=0.2,
             April_Ground_Reflectance=0.2,
             May_Ground_Reflectance=0.2,
             June_Ground_Reflectance=0.2,
             July_Ground_Reflectance=0.2,
             August_Ground_Reflectance=0.2,
             September_Ground_Reflectance=0.2,
             October_Ground_Reflectance=0.2,
             November_Ground_Reflectance=0.2,
             December_Ground_Reflectance=0.2)

    safe_add(idf, "Site:GroundReflectance:SnowModifier",
             Ground_Reflected_Solar_Modifier=1.0,
             Daylighting_Ground_Reflected_Solar_Modifier=1.0)


    safe_add(idf, "Site:WeatherStation",
             Wind_Sensor_Height_Above_Ground=10.0,
             Wind_Speed_Profile_Exponent=0.22,
             Wind_Speed_Profile_Boundary_Layer_Thickness=370.0,
             Air_Temperature_Sensor_Height_Above_Ground=1.5)


    safe_add(idf, "ShadowCalculation",
             Shading_Calculation_Method="PixelCounting",
             Shading_Calculation_Update_Frequency_Method="Periodic",
             Shading_Calculation_Update_Frequency=20,
             Maximum_Figures_in_Shadow_Overlap_Calculations=15000,
             Pixel_Counting_Resolution=512,
             Sky_Diffuse_Modeling_Algorithm="SimpleSkyDiffuseModeling")


    safe_add(idf, "HeatBalanceAlgorithm",
             Algorithm="ConductionTransferFunction",
             Surface_Temperature_Upper_Limit=200,
             Minimum_Surface_Convection_Heat_Transfer_Coefficient_Value=0.1,
             Maximum_Surface_Convection_Heat_Transfer_Coefficient_Value=1000)

    safe_add(idf, "SurfaceConvectionAlgorithm:Inside",  Algorithm="TARP")
    safe_add(idf, "SurfaceConvectionAlgorithm:Outside", Algorithm="DOE-2")


    safe_add(idf, "ConvergenceLimits",
             Minimum_System_Timestep=0,
             Maximum_HVAC_Iterations=50,
             Minimum_Plant_Iterations=3,
             Maximum_Plant_Iterations=9)


    safe_add(idf, "Site:GroundTemperature:BuildingSurface",
             January_Surface_Ground_Temperature=18,
             February_Surface_Ground_Temperature=18,
             March_Surface_Ground_Temperature=18,
             April_Surface_Ground_Temperature=18,
             May_Surface_Ground_Temperature=18,
             June_Surface_Ground_Temperature=18,
             July_Surface_Ground_Temperature=18,
             August_Surface_Ground_Temperature=18,
             September_Surface_Ground_Temperature=18,
             October_Surface_Ground_Temperature=18,
             November_Surface_Ground_Temperature=18,
             December_Surface_Ground_Temperature=18)

    safe_add(idf, "RunPeriod",
             Name="Annual",
             Begin_Month=1, Begin_Day_of_Month=1,
             End_Month=12, End_Day_of_Month=31,
             Day_of_Week_for_Start_Day="Sunday",
             Use_Weather_File_Holidays_and_Special_Days="No",
             Use_Weather_File_Daylight_Saving_Period="No",
             Apply_Weekend_Holiday_Rule="No",
             Use_Weather_File_Rain_Indicators="Yes",
             Use_Weather_File_Snow_Indicators="Yes")


    safe_add(idf, "SizingPeriod:WeatherFileConditionType",
             Name="Extreme Summer Weather Period for Design",
             Period_Selection="SummerExtreme",
             Use_Weather_File_Daylight_Saving_Period="No",
             Use_Weather_File_Rain_and_Snow_Indicators="No")

    safe_add(idf, "SizingPeriod:WeatherFileConditionType",
             Name="Extreme Winter Weather Period for Design",
             Period_Selection="WinterExtreme",
             Use_Weather_File_Daylight_Saving_Period="No",
             Use_Weather_File_Rain_and_Snow_Indicators="No")

    safe_add(idf, "Sizing:Parameters",
             Heating_Sizing_Factor=1.5,
             Cooling_Sizing_Factor=1.5)


    safe_add(idf, "ScheduleTypeLimits", Name="Any Number")
    safe_add(idf, "ScheduleTypeLimits", Name="AnyNumber")
    safe_add(idf, "ScheduleTypeLimits", Name="Fraction",
             Lower_Limit_Value=0.0, Upper_Limit_Value=1.0, Numeric_Type="CONTINUOUS")
    safe_add(idf, "ScheduleTypeLimits", Name="Temperature",
             Lower_Limit_Value=-60, Upper_Limit_Value=200, Numeric_Type="CONTINUOUS")
    safe_add(idf, "ScheduleTypeLimits", Name="On/Off",
             Lower_Limit_Value=0, Upper_Limit_Value=1, Numeric_Type="DISCRETE")

    safe_add(idf, "Schedule:Constant", Name="AlwaysOn",
             Schedule_Type_Limits_Name="Fraction", Hourly_Value=1)

    safe_add(idf, "Schedule:Constant", Name="AllOn",
             Schedule_Type_Limits_Name="Fraction", Hourly_Value=1)
    safe_add(idf, "Schedule:Constant", Name="On",
             Schedule_Type_Limits_Name="On/Off", Hourly_Value=1)
    safe_add(idf, "Schedule:Constant", Name="Off",
             Schedule_Type_Limits_Name="On/Off", Hourly_Value=0)


    safe_add(idf, "Material:NoMass",
             Name="MAT_OPAQUE_EQUIV_U",
             Roughness="MediumRough",
             Thermal_Resistance=1.0 / float(OPAQUE_U_VALUE),
             Thermal_Absorptance=0.9,
             Solar_Absorptance=0.7,
             Visible_Absorptance=0.7)


    MAT_THICK = 0.2
    safe_add(idf, "Material",
             Name="MAT_IWALL_0.2_CONST_IWALL",
             Roughness="MediumRough",
             Thickness=MAT_THICK,
             Conductivity=float(INTERIOR_WALL_CONDUCTIVITY),
             Density=2400, Specific_Heat=840,
             Thermal_Absorptance=0.9, Solar_Absorptance=0.7, Visible_Absorptance=0.7)

    safe_add(idf, "Material",
             Name="MAT_DOOR",
             Roughness="MediumSmooth",
             Thickness=MAT_THICK,
             Conductivity=2.0,
             Density=2400, Specific_Heat=840,
             Thermal_Absorptance=0.9, Solar_Absorptance=0.6, Visible_Absorptance=0.6)

    safe_add(idf, "Construction", Name="CONST_EWALL",         Outside_Layer="MAT_OPAQUE_EQUIV_U")
    safe_add(idf, "Construction", Name="CONST_EWALL_FLIPPED", Outside_Layer="MAT_OPAQUE_EQUIV_U")
    safe_add(idf, "Construction", Name="CONST_IWALL",         Outside_Layer="MAT_IWALL_0.2_CONST_IWALL")
    safe_add(idf, "Construction", Name="CONST_IWALL_FLIPPED", Outside_Layer="MAT_IWALL_0.2_CONST_IWALL")
    safe_add(idf, "Construction", Name="CONST_ROOF",          Outside_Layer="MAT_OPAQUE_EQUIV_U")
    safe_add(idf, "Construction", Name="CONST_ROOF_FLIPPED",  Outside_Layer="MAT_OPAQUE_EQUIV_U")
    safe_add(idf, "Construction", Name="CONST_FLOOR",         Outside_Layer="MAT_OPAQUE_EQUIV_U")
    safe_add(idf, "Construction", Name="CONST_FLOOR_FLIPPED", Outside_Layer="MAT_OPAQUE_EQUIV_U")
    safe_add(idf, "Construction", Name="CONST_DOOR",          Outside_Layer="MAT_DOOR")

    safe_add(idf, "WindowMaterial:SimpleGlazingSystem",
             Name="GLZWindow",
             UFactor=float(WINDOW_U),
             Solar_Heat_Gain_Coefficient=float(WINDOW_SHGC),
             Visible_Transmittance=float(WINDOW_TVIS))
    safe_add(idf, "Construction", Name="GLZ",         Outside_Layer="GLZWindow")
    safe_add(idf, "Construction", Name="CONST_WINDOW", Outside_Layer="GLZWindow")


    for zn in zone_names_list:
        safe_add(idf, "Zone", Name=zn)

        inlet_node  = f"Node {zn} In"
        zone_node   = f"Node {zn} Zone"
        return_node = f"Node {zn} Out"
        inlet_list  = f"{zn} Inlets"
        equip_list  = f"{zn} Equipment"
        ideal_name  = f"{zn} Ideal Loads Air"


        safe_add(idf, "NodeList",
                 Name=inlet_list,
                 Node_1_Name=inlet_node)


        safe_add(idf, "ZoneHVAC:EquipmentList",
                 Name=equip_list,
                 Load_Distribution_Scheme="UniformLoad",
                 Zone_Equipment_1_Object_Type="ZoneHVAC:IdealLoadsAirSystem",
                 Zone_Equipment_1_Name=ideal_name,
                 Zone_Equipment_1_Cooling_Sequence=1,
                 Zone_Equipment_1_Heating_or_NoLoad_Sequence=1)

        safe_add(idf, "ZoneHVAC:EquipmentConnections",
                 Zone_Name=zn,
                 Zone_Conditioning_Equipment_List_Name=equip_list,
                 Zone_Air_Inlet_Node_or_NodeList_Name=inlet_list,
                 Zone_Air_Node_Name=zone_node,
                 Zone_Return_Air_Node_or_NodeList_Name=return_node)


        safe_add(idf, "ZoneHVAC:IdealLoadsAirSystem",
                 Name=ideal_name,
                 Zone_Supply_Air_Node_Name=inlet_node,
                 Maximum_Heating_Supply_Air_Temperature=35,
                 Minimum_Cooling_Supply_Air_Temperature=12,
                 Maximum_Heating_Supply_Air_Humidity_Ratio=0.01,
                 Minimum_Cooling_Supply_Air_Humidity_Ratio=0.008,
                 Heating_Limit="LimitFlowRateAndCapacity",
                 Maximum_Heating_Air_Flow_Rate="Autosize",
                 Maximum_Sensible_Heating_Capacity="Autosize",
                 Cooling_Limit="LimitFlowRateAndCapacity",
                 Maximum_Cooling_Air_Flow_Rate="Autosize",
                 Maximum_Total_Cooling_Capacity="Autosize",
                 Heating_Availability_Schedule_Name=("HVACAvail_Bedroom" if "bedroom" in zn.lower() else "HVACAvail_Other"),
                 Cooling_Availability_Schedule_Name=("HVACAvail_Bedroom" if "bedroom" in zn.lower() else "HVACAvail_Other"),
                 Dehumidification_Control_Type="None",
                 Humidification_Control_Type="None",
                 Outdoor_Air_Economizer_Type="NoEconomizer",
                 Heat_Recovery_Type="None",
                 Sensible_Heat_Recovery_Effectiveness=0.7,
                 Latent_Heat_Recovery_Effectiveness=0.65)


        safe_add(idf, "Sizing:Zone",
                 Zone_or_ZoneList_Name=zn,
                 Zone_Cooling_Design_Supply_Air_Temperature_Input_Method="SupplyAirTemperature",
                 Zone_Cooling_Design_Supply_Air_Temperature=12,
                 Zone_Heating_Design_Supply_Air_Temperature_Input_Method="SupplyAirTemperature",
                 Zone_Heating_Design_Supply_Air_Temperature=35,
                 Zone_Cooling_Design_Supply_Air_Humidity_Ratio=0.009,
                 Zone_Heating_Design_Supply_Air_Humidity_Ratio=0.004,
                 Zone_Heating_Sizing_Factor=1.15,
                 Zone_Cooling_Sizing_Factor=1.15,
                 Cooling_Design_Air_Flow_Method="DesignDay",
                 Heating_Design_Air_Flow_Method="DesignDay")


        heat_sch = f"{zn}Heating_SP_Sch"
        cool_sch = f"{zn}Cooling_SP_Sch"
        ctrl_sch = f"{zn}_ZoneControlTypeSched"
        dual_sp  = f"Dual Setpoint - Zone {zn}"

        safe_add(idf, "Schedule:Constant",
                 Name=heat_sch,
                 Schedule_Type_Limits_Name="Temperature",
                 Hourly_Value=HEATING_SETPOINT)
        safe_add(idf, "Schedule:Constant",
                 Name=cool_sch,
                 Schedule_Type_Limits_Name="Temperature",
                 Hourly_Value=COOLING_SETPOINT)
        safe_add(idf, "Schedule:Constant",
                 Name=ctrl_sch,
                 Schedule_Type_Limits_Name="Any Number",
                 Hourly_Value=4)

        safe_add(idf, "ThermostatSetpoint:DualSetpoint",
                 Name=dual_sp,
                 Heating_Setpoint_Temperature_Schedule_Name=heat_sch,
                 Cooling_Setpoint_Temperature_Schedule_Name=cool_sch)

        safe_add(idf, "ZoneControl:Thermostat",
                 Name=f"{zn} Thermostat",
                 Zone_or_ZoneList_Name=zn,
                 Control_Type_Schedule_Name=ctrl_sch,
                 Control_1_Object_Type="ThermostatSetpoint:DualSetpoint",
                 Control_1_Name=dual_sp)


        humid_min_sch = f"{zn}HumidMin"
        humid_max_sch = f"{zn}HumidMax"
        safe_add(idf, "Schedule:Constant",
                 Name=humid_min_sch,
                 Schedule_Type_Limits_Name="Any Number",
                 Hourly_Value=0)
        safe_add(idf, "Schedule:Constant",
                 Name=humid_max_sch,
                 Schedule_Type_Limits_Name="Any Number",
                 Hourly_Value=100)
        safe_add(idf, "ZoneControl:Humidistat",
                 Name=f"{zn}Humidistat",
                 Zone_Name=zn,
                 Humidifying_Relative_Humidity_Setpoint_Schedule_Name=humid_min_sch,
                 Dehumidifying_Relative_Humidity_Setpoint_Schedule_Name=humid_max_sch)


    for zk in zone_keys:
        zn = zone_names_map[zk]
        poly = zones_xy[zk]

        roof = safe_add(idf, "BuildingSurface:Detailed",
                        Name=f"ROOF_{zn}",
                        Surface_Type="Roof",
                        Construction_Name="CONST_ROOF",
                        Zone_Name=zn,
                        Outside_Boundary_Condition="Outdoors",
                        Sun_Exposure="SunExposed",
                        Wind_Exposure="WindExposed",
                        View_Factor_to_Ground="autocalculate")
        if roof:
            add_vertices_fields(roof, [(x, y, ZONE_HEIGHT) for (x, y) in poly])

        floor_poly = list(reversed(poly))
        floor = safe_add(idf, "BuildingSurface:Detailed",
                         Name=f"FLOOR_{zn}",
                         Surface_Type="Floor",
                         Construction_Name="CONST_FLOOR",
                         Zone_Name=zn,
                         Outside_Boundary_Condition="Ground",
                         Sun_Exposure="NoSun",
                         Wind_Exposure="NoWind",
                         View_Factor_to_Ground="autocalculate")
        if floor:
            add_vertices_fields(floor, [(x, y, 0.0) for (x, y) in floor_poly])

        for i in range(0):
            a = poly[i]
            b = poly[(i + 1) % n]
            wname = wall_name_map[(zk, i)]

            if (zk, i) in adj:
                other_zk, other_i = adj[(zk, i)]
                bc = "Surface"
                bc_obj = wall_name_map[(other_zk, other_i)]
                sun, wind = "NoSun", "NoWind"
                constr = "CONST_IWALL"
            else:
                bc = "Outdoors"
                bc_obj = ""
                sun, wind = "SunExposed", "WindExposed"
                constr = "CONST_EWALL"

            wall = safe_add(idf, "BuildingSurface:Detailed",
                            Name=wname,
                            Surface_Type="Wall",
                            Construction_Name=constr,
                            Zone_Name=zn,
                            Outside_Boundary_Condition=bc,
                            Outside_Boundary_Condition_Object=bc_obj,
                            Sun_Exposure=sun,
                            Wind_Exposure=wind,
                            View_Factor_to_Ground="autocalculate")
            if wall:


                add_vertices_fields(wall, [
                    (a[0], a[1], 0.0),
                    (b[0], b[1], 0.0),
                    (b[0], b[1], ZONE_HEIGHT),
                    (a[0], a[1], ZONE_HEIGHT)
                ])


    for wall_info in wall_segments:
        a = wall_info["a"]
        b = wall_info["b"]
        wall = safe_add(idf, "BuildingSurface:Detailed",
                        Name=wall_info["name"],
                        Surface_Type="Wall",
                        Construction_Name=wall_info["construction"],
                        Zone_Name=wall_info["zone_name"],
                        Outside_Boundary_Condition=wall_info["bc"],
                        Outside_Boundary_Condition_Object=wall_info["bc_obj"],
                        Sun_Exposure=wall_info["sun"],
                        Wind_Exposure=wall_info["wind"],
                        View_Factor_to_Ground="autocalculate")
        if wall:

            add_vertices_fields(wall, [
                (a[0], a[1], 0.0),
                (b[0], b[1], 0.0),
                (b[0], b[1], ZONE_HEIGHT),
                (a[0], a[1], ZONE_HEIGHT)
            ])

    def add_fen(kind: str, seg: Tuple[Vec2, Vec2], host: Dict[str, Any],
                sill: float, height: float, constr: str, prefix: str, idx: int) -> bool:
        zk = host["zone_key"]
        wall_a, wall_b = host["a"], host["b"]
        host_wall = host["name"]
        verts1 = make_opening_vertices(seg, wall_a, wall_b, sill, height)
        if not verts1:
            return False

        name1 = f"{prefix}_{zone_names_map[zk]}_{idx}"

        if host.get("bc") == "Surface" and host.get("paired_name"):
            other_zk = host["paired_zone_key"]
            oa = host["paired_a"]
            ob = host["paired_b"]
            other_wall = host["paired_name"]
            verts2 = make_opening_vertices(seg, oa, ob, sill, height)
            if not verts2:
                return False

            name2 = f"{prefix}_{zone_names_map[other_zk]}_{idx}"

            f1 = safe_add(idf, "FenestrationSurface:Detailed",
                          Name=name1, Surface_Type=kind,
                          Construction_Name=constr,
                          Building_Surface_Name=host_wall,
                          Outside_Boundary_Condition_Object=name2,
                          View_Factor_to_Ground="autocalculate",
                          Multiplier=1)
            if f1:
                add_vertices_fields(f1, verts1)

            f2 = safe_add(idf, "FenestrationSurface:Detailed",
                          Name=name2, Surface_Type=kind,
                          Construction_Name=constr,
                          Building_Surface_Name=other_wall,
                          Outside_Boundary_Condition_Object=name1,
                          View_Factor_to_Ground="autocalculate",
                          Multiplier=1)
            if f2:
                add_vertices_fields(f2, verts2)
        else:
            f = safe_add(idf, "FenestrationSurface:Detailed",
                         Name=name1, Surface_Type=kind,
                         Construction_Name=constr,
                         Building_Surface_Name=host_wall,
                         View_Factor_to_Ground="autocalculate",
                         Multiplier=1)
            if f:
                add_vertices_fields(f, verts1)
        return True

    for i, (seg, host) in enumerate(door_hosts, start=1):
        add_fen("Door", seg, host, 0.0, DOOR_HEIGHT, "CONST_DOOR", "DOOR", i)

    for i, (seg, host) in enumerate(win_hosts, start=1):
        add_fen("Window", seg, host, WIN_SILL, WIN_HEIGHT, "CONST_WINDOW", "WIN", i)


    safe_add(idf, "Output:Diagnostics", Key_1="DisplayAdvancedReportVariables")
    safe_add(idf, "OutputControl:Table:Style", Column_Separator="Comma")
    safe_add(idf, "Output:Table:SummaryReports",
             Report_1_Name="EnvelopeSummary",
             Report_2_Name="AnnualBuildingUtilityPerformanceSummary",
             Report_3_Name="InputVerificationandResultsSummary",
             Report_4_Name="SourceEnergyEndUseComponentsSummary",
             Report_5_Name="ShadingSummary")
    add_output_variables(idf)

    out_idf.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(out_idf))

    append_schedules_raw(out_idf)
    append_loads_raw(out_idf, zone_names_list, zones_area_map)
    clean_zonecontrol_thermostat_blocks(out_idf)
