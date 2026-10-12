"""Perception for the agent: detect coloured cubes in the overhead camera image.

The VLM/LLM planner never sees simulator state. It gets the overhead image plus this
detector's object list (colour, pixel centroid, estimated table position), the same thing a
real overhead webcam plus a colour-blob detector would provide on the physical workcell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_workcell.sim.robot import SO101

# Overhead camera as defined in workcell.xml: straight down from (0.25, 0, 0.75), fovy 55 deg,
# image x-axis = world -y, image y-axis = world -x.
CAM_POS = np.array([0.25, 0.0, 0.75])
CAM_FOVY_DEG = 55.0
TABLE_Z = 0.0125  # cube centre height

# HSV hue windows (OpenCV-style 0..180) for the cube colours in the scene materials
# No yellow window: the SO-101 itself is yellow and would be detected as an object.
COLOR_HUES = {"red": (0, 10), "green": (40, 85), "blue": (95, 135)}


@dataclass
class Detection:
    color: str
    pixel: tuple[float, float]
    table_xy: tuple[float, float]
    area_px: int

    def describe(self) -> str:
        return f"{self.color} cube at table position x={self.table_xy[0]:.2f} m, y={self.table_xy[1]:.2f} m"


def pixel_to_table(u: float, v: float, width: int, height: int, z: float = TABLE_Z) -> tuple[float, float]:
    """Back-project an overhead-camera pixel to the table plane at height z."""
    f = 0.5 * height / np.tan(np.radians(CAM_FOVY_DEG) / 2)
    depth = CAM_POS[2] - z
    cx, cy = (width - 1) / 2, (height - 1) / 2
    # camera frame: right = world -y, down = world -x
    world_y = -(u - cx) / f * depth
    world_x = CAM_POS[0] - (v - cy) / f * depth
    return float(world_x), float(world_y)


def _rgb_to_hsv(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = img.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    diff = mx - mn + 1e-9
    h = np.zeros_like(mx)
    m = mx == r
    h[m] = (60 * ((g - b) / diff) % 360)[m]
    m = mx == g
    h[m] = (60 * ((b - r) / diff) + 120)[m]
    m = mx == b
    h[m] = (60 * ((r - g) / diff) + 240)[m]
    s = diff / (mx + 1e-9)
    return h / 2.0, s, mx  # hue in 0..180 like OpenCV


def detect_cubes(img: np.ndarray, *, min_area: int = 30) -> list[Detection]:
    """Colour-blob detector. Returns one detection per colour (largest blob)."""
    h, s, v = _rgb_to_hsv(img)
    height, width = img.shape[:2]
    out = []
    saturated = (s > 0.35) & (v > 0.35)
    for color, (lo, hi) in COLOR_HUES.items():
        mask = saturated & (h >= lo) & (h <= hi)
        if color == "red":
            mask |= saturated & (h >= 170)
        # bins are large flat regions of similar hue; cubes are small. Reject blobs that are
        # too large to be a cube by a crude connected-component pass.
        ys, xs = np.nonzero(mask)
        if len(xs) < min_area:
            continue
        # a 2.5 cm cube covers ~250 px at this camera; the 9 cm bins cover ~3000 px
        comp = _largest_small_component(mask, max_area=int(0.004 * height * width))
        if comp is None:
            continue
        cy_, cx_ = comp
        u, v_ = float(np.mean(cx_)), float(np.mean(cy_))
        out.append(Detection(color, (u, v_), pixel_to_table(u, v_, width, height), int(len(cx_))))
    return out


def _largest_small_component(mask: np.ndarray, max_area: int, min_area: int = 40):
    """Largest 4-connected component that looks like a cube: area in [min_area, max_area],
    near-square bounding box, and mostly filled (bin walls are long thin strips)."""
    visited = np.zeros_like(mask, dtype=bool)
    best = None
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        comp_y, comp_x = [], []
        while stack:
            y, x = stack.pop()
            comp_y.append(y)
            comp_x.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        n = len(comp_x)
        if not (min_area <= n <= max_area):
            continue
        h = max(comp_y) - min(comp_y) + 1
        w = max(comp_x) - min(comp_x) + 1
        aspect = w / h
        fill = n / (w * h)
        if not (0.6 <= aspect <= 1.7 and fill >= 0.55):
            continue
        if best is None or n > len(best[0]):
            best = (np.array(comp_y), np.array(comp_x))
    return best


def observe(robot: SO101, width: int = 640, height: int = 480) -> tuple[np.ndarray, list[Detection]]:
    img = robot.render("overhead", width, height)
    return img, detect_cubes(img)


def scene_description(dets: list[Detection]) -> str:
    bins = "Bins: 'bin_left' (blue tray, at x=0.17 m, y=+0.16 m, on the arm's left) and 'bin_right' (orange tray, at x=0.17 m, y=-0.16 m, on the arm's right)."
    if not dets:
        return bins + " No cubes detected."
    return bins + " Detected objects: " + "; ".join(d.describe() for d in dets) + "."
