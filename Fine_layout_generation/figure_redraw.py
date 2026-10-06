import numpy as np
import cv2
import os
import json
from PIL import Image, ImageDraw
import shutil
from collections import Counter

def draw_vertex_boxes(img, vertices, color=(0, 0, 255), box_size=3, thickness=-1):
    """
    Draw a filled square at each vertex on the given image.

    :param img:        target image (np.ndarray)
    :param vertices:   (N, 2) array of vertex coordinates [[x1, y1], ...]
    :param color:      BGR color tuple; default: red (0, 0, 255)
    :param box_size:   Square side length (pixels); default: 3
    :param thickness:  Rectangle fill thickness; -1 means filled
    """
    vertices = np.asarray(vertices, dtype=int)
    half = box_size // 2
    for x, y in vertices:
        pt1 = (x - half, y - half)
        pt2 = (x + half, y + half)
        cv2.rectangle(img, pt1, pt2, color, thickness=thickness)
    return img

def draw_smooth_lines_pil(img_cv2, vertices, color, thickness=3):
    """
    Use PIL to draw smooth lines (instead of cv2.polylines) to reduce aliasing
    :param img_cv2: OpenCV canvas (BGR)
    :param vertices: coordinate array of shape (n, 2), dtype int32
    :param color: BGR color → convert to PIL RGB color
    :param thickness: line thickness
    :return: drawn canvas in OpenCV format
    """
    # 1. OpenCV BGR → PIL RGB
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    # 2. Convert coordinates to PIL format (a list of tuples)
    points = [(int(v[0]), int(v[1])) for v in vertices]

    # 3. Draw lines with PIL (antialiasing is enabled by default)
    # Note: PIL uses RGB colors, so reverse the OpenCV BGR channel order
    pil_color = (color[2], color[1], color[0])
    draw.line(
        points,
        fill=pil_color,
        width=thickness,
        joint="round"  # Round the corners to further reduce aliasing
    )

    # 4. PIL → OpenCV BGR
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv2

# Read the JSON file
# -------Read a complete JSON file------------
#with open('../stage3_output/test_vec.json', 'r') as file:
#    datas = [json.loads(line) for line in file.readlines()]
path_base = f'../test/Fine_output/cache_json/'
json_all = os.listdir(path_base)
datas = []
for json_name in json_all:
    with open(f'{path_base}/{json_name}', 'r', encoding='utf-8') as file:
        datas.extend(json.loads(line) for line in file)

# Ensure the output directory exists
output_dir = "../test/Fine_output"
os.makedirs(output_dir, exist_ok=True)

for data in datas:
    pic_name = data['Pic']
    del data['Pic']
    #del data['door']
    #del data['window']
    # ----------------------
    # 1. Updated room data (merge entries of the same type)
    # ----------------------
    '''
    room_data = {
        "living_room": ["M118,54 118,82 67,82 67,138 187,138 187,118 166,118 166,82 140,82 140,54 Z"],
        "bedroom": ["M140,138 140,200 187,200 187,138 Z",
                    "M118,138 118,172 88,172 88,227 140,227 140,138 Z",
                    "M75,31 75,78 118,78 118,31 Z"],
        "kitchen": ["M166,78 166,118 192,118 192,78 Z"],
        "bathroom": ["M88,138 88,172 118,172 118,138 Z",
                   "M118,31 118,54 143,54 143,31 Z"],
        "balcony": ["M146,200 146,227 193,227 193,200 Z"]
    }
    '''
    room_data = data

    # ----------------------
    # 2. Color definitions (BGR format for OpenCV)
    # ----------------------
    room_colors = {
        "living_room": (229, 242, 244),  # living room
        "bathroom": (251, 233, 204),  # bathroom
        "bedroom": (171, 244, 253),  # bedroom
        "kitchen": (213, 215, 233),  # kitchen
        "balcony": (134, 216, 208),  # balcony
        "door": (255, 0, 0),  # blue (BGR)
        "window": (0, 133, 0)  # green (BGR)
    }


    # ----------------------
    # 3. Parse SVG paths into OpenCV polygon coordinates
    # ----------------------
    def parse_path(path_str):
        """Parse an SVG path string into an array of polygon vertices with dtype int32."""
        # Remove 'M' and 'Z' from the path and extract coordinate pairs
        clean_str = path_str.replace("M", "").replace("Z", "").strip()
        coords = clean_str.split()  # Split into ["x1,y1", "x2,y2", ...]

        points = []
        for coord in coords:
            x, y = coord.split(",")
            #points.append((int(float(x)//4), int(float(y)//4)))  # Convert to integer coordinates
            points.append((int(float(x)), int(float(y))))  # Convert to integer coordinates

        return np.array(points, dtype=np.int32)  # Return an OpenCV-compatible array


    # ----------------------
    # 4. Compute image dimensions to fit all room coordinates
    # ----------------------
    # Collect all vertices to determine the coordinate bounds
    all_vertices = []
    for room_type in room_data:
        for path_str in room_data[room_type]:
            vertices = parse_path(path_str)
            all_vertices.extend(vertices)

    all_vertices = np.array(all_vertices)
    min_x, min_y = all_vertices.min(axis=0) - 10  # Leave a 10-pixel margin at the edges
    max_x, max_y = all_vertices.max(axis=0) + 10

    # Fix the image size (keep it consistent with the original code)
    img_width = 256
    img_height = 256

    # ----------------------
    # 5. Compose all rooms into a single image
    # ----------------------
    # Create a blank black canvas (3-channel BGR)
    combined_img = np.zeros((img_height, img_width, 3), dtype=np.uint8) + 255

    # Draw each room type onto the same canvas in sequence
    for room_type in room_data:
        # Get the fill color for the current room
        fill_color = room_colors[room_type]

        # Draw all polygons of this type
        for path_str in room_data[room_type]:
            vertices = parse_path(path_str)
            # Preserve the original coordinate system (uncomment the next line to align with the top-left corner if needed)
            # adjusted_vertices = vertices - [min_x, min_y]
            adjusted_vertices = vertices

            # Fill the polygon (overlay it on the canvas)
            cv2.fillPoly(combined_img, [adjusted_vertices], color=fill_color)
            # Draw a red outline to emphasize the contour
            #combined_img = draw_smooth_lines_pil(combined_img, adjusted_vertices, (0, 0, 255), thickness=3)
            cv2.polylines(combined_img, [adjusted_vertices], isClosed=True, color=(0, 0, 255), thickness=2)
            combined_img = draw_vertex_boxes(combined_img, adjusted_vertices, color=(0, 0, 255))

    # Step 2: draw doors (blue lines, thickness 1)
    door_color = room_colors["door"]
    for path_str in room_data["door"]:
        vertices = parse_path(path_str)
        if vertices.size < 2:  # Skip invalid lines (fewer than 2 points)
            continue
        adjusted_vertices = vertices
        # Draw door lines (open paths)
        combined_img = draw_smooth_lines_pil(combined_img, adjusted_vertices, door_color, thickness=3)
        combined_img = draw_vertex_boxes(combined_img, adjusted_vertices, color=door_color)
        #cv2.polylines(combined_img, [adjusted_vertices], isClosed=False, color=door_color, thickness=2)

    # Step 3: draw windows (green lines, thickness 1)
    window_color = room_colors["window"]
    for path_str in room_data["window"]:
        vertices = parse_path(path_str)
        if vertices.size < 2:  # Skip invalid lines (fewer than 2 points)
            continue
        adjusted_vertices = vertices
        # Draw window lines (open paths)
        combined_img = draw_smooth_lines_pil(combined_img, adjusted_vertices, window_color, thickness=3)
        combined_img = draw_vertex_boxes(combined_img, adjusted_vertices, color=window_color)
        #cv2.polylines(combined_img, [adjusted_vertices], isClosed=False, color=window_color, thickness=2)

    # ----------------------
    # 6. Save the composited image
    # ----------------------
    combined_save_path = os.path.join(output_dir, pic_name)
    cv2.imwrite(combined_save_path, combined_img)
    print(f"Composite image saved: {combined_save_path}")

# Optional: display the composited image (requires a GUI, such as a local environment)
# cv2.imshow("All Rooms Combined", combined_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
