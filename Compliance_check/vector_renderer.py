"""Render a floor-plan image from vector JSON data."""

import re
from typing import Any, Dict, Iterable

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


ROOM_COLORS = {
    "living_room": (229, 242, 244),
    "bathroom": (251, 233, 204),
    "bedroom": (171, 244, 253),
    "kitchen": (213, 215, 233),
    "balcony": (134, 216, 208),
}
DOOR_COLOR = (255, 0, 0)
WINDOW_COLOR = (0, 133, 0)


def parse_vector_path(path: str) -> np.ndarray:
    """Convert comma-separated vector coordinates into integer image points."""
    coordinates = re.findall(r"(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)", str(path))
    points = [(int(float(x)), int(float(y))) for x, y in coordinates]
    return np.asarray(points, dtype=np.int32).reshape(-1, 2)


def draw_vertex_boxes(
    image: np.ndarray,
    vertices: np.ndarray,
    color: tuple,
    box_size: int = 3,
) -> np.ndarray:
    half = box_size // 2
    for x, y in vertices:
        cv2.rectangle(
            image,
            (int(x) - half, int(y) - half),
            (int(x) + half, int(y) + half),
            color,
            thickness=-1,
        )
    return image


def draw_smooth_lines(image: np.ndarray, vertices: np.ndarray, color: tuple) -> np.ndarray:
    if len(vertices) < 2:
        return image

    if Image is not None and ImageDraw is not None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        points = [(int(x), int(y)) for x, y in vertices]
        rgb_color = (color[2], color[1], color[0])
        draw.line(points, fill=rgb_color, width=3, joint="round")
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    for start, end in zip(vertices, vertices[1:]):
        cv2.line(
            image,
            tuple(map(int, start)),
            tuple(map(int, end)),
            color,
            thickness=3,
            lineType=cv2.LINE_8,
        )
    return image


def _paths(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return (item for item in value if isinstance(item, str))
    return ()


def render_floorplan(
    vector_data: Dict[str, Any],
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    """Render vectors with the colors and drawing order used by the source script."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    for room_type, paths in vector_data.items():
        fill_color = ROOM_COLORS.get(room_type)
        if fill_color is None:
            continue
        for path in _paths(paths):
            vertices = parse_vector_path(path)
            if len(vertices) < 3:
                continue
            polygon = vertices.reshape(-1, 1, 2)
            cv2.fillPoly(image, [polygon], color=fill_color)
            cv2.polylines(image, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
            draw_vertex_boxes(image, vertices, color=(0, 0, 255))

    for key, color in (("door", DOOR_COLOR), ("window", WINDOW_COLOR)):
        for path in _paths(vector_data.get(key, [])):
            vertices = parse_vector_path(path)
            if len(vertices) < 2:
                continue
            image = draw_smooth_lines(image, vertices, color)
            draw_vertex_boxes(image, vertices, color=color)

    return image
