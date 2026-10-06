from collections import Counter

def extract_most_frequent_coords(svg_path):
    coords_str = svg_path.replace('M', '').replace('Z', '').strip()
    point_list = [p for p in coords_str.split() if ',' in p]
    x_coords = []
    y_coords = []
    for point in point_list:
        x, y = point.split(',')
        x_coords.append(int(float(x)))
        y_coords.append(int(float(y)))
    most_x = Counter(x_coords).most_common(1)[0][0]
    most_y = Counter(y_coords).most_common(1)[0][0]
    return most_x, most_y

def split_polyline_by_inflection_rule(json_data):
    """
    Process polyline doors strictly according to the example rules:
    1. Identify junctions (repeated coordinate points)
    2. Keep the shared coordinate at each junction unchanged
    3. For the varying coordinate, subtract 1 when it is greater than the opposite side and add 1 when it is smaller, leaving a 1-pixel gap
    Example: "M196,132 196,156 216,156 196,156 Z"
          → First segment: 196,132 → 196,155 (156 > 132, so 156 - 1)
          → Second segment: 216,156 → 197,156 (216 > 196, so 196 + 1)
          → Third segment: adjust the relevant segment from 196,156 according to the rule
    """
    original_doors = json_data.get("door", [])
    new_doors = []

    for door in original_doors:
        # 1. Parse the path and extract integer coordinates (only process paths starting with M; remove the Z symbol)
        if not door.startswith('M'):
            new_doors.append(door)
            continue

        coords_part = door[1:].replace('Z', '').strip()
        points_str = coords_part.split()
        points = []
        for p in points_str:
            if ',' in p:
                x, y = p.split(',')
                points.append((int(float(x)), int(float(y))))

        # 2. Filter invalid paths and straight lines; process only polylines (at least 2 segments, i.e. at least 3 points)
        if len(points) < 2:
            new_doors.append(door)
            continue
        if len(points) == 2:
            sx, sy = points[0]
            ex, ey = points[1]
            new_doors.append(f"M{sx},{sy} {ex},{ey}")
            continue

        mx, my = extract_most_frequent_coords(door)
        # 3. Core: process each pair of adjacent segments according to the example rules (split and adjust segment by segment)
        #points = points[:3]
        for i in range(len(points)//2):
            # Get the start and end points of the current segment (points[i] → points[i+1])
            p1 = points[i * 2]
            p2 = points[i * 2 + 1]
            x1, y1 = p1
            x2, y2 = p2

            new_x1, new_y1 = p1
            new_x2, new_y2 = p2

            # Step 1: identify the current segment's shared and varying coordinates
            if x1 == x2:
                if y1 < my:
                    new_y2 -= 3
                elif y1 > my:
                    new_y2 += 3
                if y2 < my:
                    new_y1 -= 3
                elif y2 > my:
                    new_y1 += 3
            elif y1 == y2:
                if x1 < mx:
                    new_x2 -= 3
                elif x1 > mx:
                    new_x2 += 3
                if x2 < mx:
                    new_x1 -= 3
                elif x2 > mx:
                    new_x1 += 3
            else:
                # Non-horizontal or non-vertical segment (leave unchanged for now)
                new_doors.append(f"M{new_x1},{new_y1} {new_x2},{new_y2}")
                continue


            new_doors.append(f"M{new_x1},{new_y1} {new_x2},{new_y2}")

    # 4. Build and return the result (preserve the original JSON format; update only the door field)
    result = json_data.copy()
    result["door"] = new_doors
    return result

def split_adherent_doors_with_gap(json_data):
    """
    Process connected door paths in the JSON by splitting them at junctions into two straight segments with a 1-pixel gap; preserve the original format
    Example: split (10,10)-(10,60) at (10,30) into (10,10)-(10,29) and (10,31)-(10,60)
    :param json_data: input floor-plan data dictionary
    :return: processed dictionary
    """
    # Step 1: extract all room corners (as integer (x, y) tuples)
    corner_points = set()

    # Iterate over all rooms (exclude non-room fields)
    room_keys = [k for k in json_data.keys() if k not in ["Pic", "door", "window"]]
    for room_key in room_keys:
        room_paths = json_data.get(room_key, [])
        for path in room_paths:
            if path.startswith("M") and path.endswith("Z"):
                # Extract and parse the coordinate string
                coord_str = path[1:-1].strip()
                point_parts = coord_str.split()
                for part in point_parts:
                    if "," in part:
                        try:
                            x = int(float(part.split(",")[0]))
                            y = int(float(part.split(",")[1]))
                            corner_points.add((x, y))
                        except (ValueError, TypeError):
                            continue

    # Step 2: define a helper to parse a single door path into a list of coordinate points
    def parse_door_path(door_path):
        if not door_path.startswith("M"):
            return []
        coord_str = door_path[1:].replace("Z", "").strip()
        point_list = []
        for part in coord_str.split():
            if "," in part:
                try:
                    x = int(float(part.split(",")[0]))
                    y = int(float(part.split(",")[1]))
                    point_list.append((x, y))
                except (ValueError, TypeError):
                    continue
        return point_list

    # Step 3: define a helper to find junctions inside a straight segment and split it with a 1-pixel gap
    def split_door_with_1px_gap(p1, p2, corners):
        """
        Find junctions between p1 and p2, splitting the segment into straight pieces with 1-pixel gaps
        :param p1: start point (x1, y1)
        :param p2: end point (x2, y2)
        :param corners: set of room junctions
        :return: list of split SVG paths
        """
        split_segments = []
        x1, y1 = p1
        x2, y2 = p2

        # Process only horizontal or vertical segments (the usual shape of floor-plan doors)
        is_vertical = (x1 == x2)
        is_horizontal = (y1 == y2)
        if not (is_vertical or is_horizontal):
            # For a non-horizontal or non-vertical segment, return it unchanged (diagonal support can be added later)
            seg = f"M{x1},{y1} {x2},{y2}"
            return [seg]

        # Find all junctions strictly between the segment endpoints
        middle_corners = []
        for (cx, cy) in corners:
            # Exclude the endpoints
            if (cx, cy) == (x1, y1) or (cx, cy) == (x2, y2):
                continue

            # Vertical segment (fixed x, y within the interval)
            if is_vertical and cx == x1:
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                if y_min < cy < y_max:
                    middle_corners.append((cx, cy))

            # Horizontal segment (fixed y, x within the interval)
            if is_horizontal and cy == y1:
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                if x_min < cx < x_max:
                    middle_corners.append((cx, cy))

        # If there are no intermediate junctions, return the original segment
        if not middle_corners:
            seg = f"M{x1},{y1} {x2},{y2}"
            return [seg]

        # Sort junctions in path order to keep the split order correct
        if is_vertical:
            # Sort vertically by y-coordinate
            middle_corners.sort(key=lambda p: p[1])
            # Handle the segment direction (top to bottom or bottom to top)
            start_y, end_y = (y1, y2) if y1 < y2 else (y2, y1)
            current_start = (x1, y1)
        else:
            # Sort horizontally by x-coordinate
            middle_corners.sort(key=lambda p: p[0])
            # Handle the segment direction (left to right or right to left)
            start_x, end_x = (x1, x2) if x1 < x2 else (x2, x1)
            current_start = (x1, y1)

        # Core: split the segment and leave a 1-pixel gap
        for corner in middle_corners:
            cx, cy = corner
            cs_x, cs_y = current_start

            # Generate the first segment's endpoint (1 pixel before the junction)
            if is_vertical:
                # Vertical: junction y-coordinate ±1
                if cs_y < cy:  # Top to bottom: first segment ends at (cx, cy-1)
                    first_end = (cx, cy - 2)
                else:  # Bottom to top: first segment ends at (cx, cy + 1)
                    first_end = (cx, cy + 2)
            else:
                # Horizontal: junction x-coordinate ±1
                if cs_x < cx:  # Left to right: first segment ends at (cx-1, cy)
                    first_end = (cx - 2, cy)
                else:  # Right to left: first segment ends at (cx + 1, cy)
                    first_end = (cx + 2, cy)

            # Add the first segment
            seg1 = f"M{cs_x},{cs_y} {first_end[0]},{first_end[1]}"
            split_segments.append(seg1)

            # Generate the second segment's start point (1 pixel after the junction, leaving a 1-pixel gap)
            if is_vertical:
                if cs_y < cy:  # Top to bottom: second segment starts at (cx, cy + 1)
                    current_start = (cx, cy + 1)
                else:  # Bottom to top: second segment starts at (cx, cy - 1)
                    current_start = (cx, cy - 1)
            else:
                if cs_x < cx:  # Left to right: second segment starts at (cx + 1, cy)
                    current_start = (cx + 1, cy)
                else:  # Right to left: second segment starts at (cx - 1, cy)
                    current_start = (cx - 1, cy)

        # Add the last segment (from the final start point to the original endpoint)
        final_seg = f"M{current_start[0]},{current_start[1]} {x2},{y2}"
        split_segments.append(final_seg)

        return split_segments

    # Step 4: process all door paths
    original_doors = json_data.get("door", [])
    processed_doors = []

    for door in original_doors:
        door_points = parse_door_path(door)
        if len(door_points) != 2:
            processed_doors.append(door)
            continue

        # Process each continuous straight segment (split cornered doors into individual straight segments first)
        for i in range(len(door_points) - 1):
            p_start = door_points[i]
            p_end = door_points[i + 1]
            # Split the current straight segment and leave a 1-pixel gap
            split_segs = split_door_with_1px_gap(p_start, p_end, corner_points)
            processed_doors.extend(split_segs)

    # Step 5: build the return value (preserve the original format)
    result = json_data.copy()
    result["door"] = processed_doors
    return result
