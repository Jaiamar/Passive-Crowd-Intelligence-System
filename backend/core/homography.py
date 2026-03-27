"""
Homography-based Polygon Area Calibration.
Maps pixel coordinates to real-world ground plane coordinates using a Homography matrix.
"""
from __future__ import annotations
from typing import Optional, List
import cv2
import numpy as np


# Default world reference points (meters): corresponds to a 10x10 meter ground plane
# These represent the corners of a known calibration region.
DEFAULT_WORLD_POINTS = np.array([
    [0.0,  0.0],
    [10.0, 0.0],
    [10.0, 10.0],
    [0.0,  10.0],
], dtype=np.float32)


def compute_homography(
    pixel_points: List[List[float]],
    world_points: Optional[List[List[float]]] = None
) -> np.ndarray:
    """
    Compute the Homography matrix H that maps pixel coords -> world coords.

    Args:
        pixel_points: 4 pixel coordinates [[x1,y1], ..., [x4,y4]]
        world_points: Optional 4 real-world reference points in meters.
                      Defaults to a 10x10m calibration area.
    Returns:
        3x3 Homography matrix H
    """
    if len(pixel_points) < 4:
        raise ValueError("At least 4 pixel points are required to compute homography.")
    src = np.array(pixel_points[:4], dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32) if world_points is not None else DEFAULT_WORLD_POINTS
    H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Could not compute homography. Ensure 4 non-collinear points are provided.")
    return H


def pixel_to_world(H: np.ndarray, pixel_points: List[List[float]]) -> np.ndarray:
    """
    Transform pixel coordinates to world (meter) coordinates using Homography H.

    Args:
        H: 3x3 Homography matrix
        pixel_points: [[x1,y1], [x2,y2], ...]
    Returns:
        world_points: Nx2 array of real-world coordinates in meters
    """
    pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, H)
    return world.reshape(-1, 2)


def polygon_area_meters(world_points: np.ndarray) -> float:
    """
    Compute the area (m²) of a polygon defined by real-world coordinates.
    Uses the Shoelace formula.

    Args:
        world_points: Nx2 array of (X, Y) coordinates in meters
    Returns:
        Area in square meters
    """
    n = len(world_points)
    if n < 3:
        return 0.0
    x = world_points[:, 0]
    y = world_points[:, 1]
    area = 0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )
    return float(area)


def calibrate_area(
    pixel_polygon: List[List[float]],
    world_ref_points: Optional[List[List[float]]] = None
) -> dict:
    """
    Full pipeline: given pixel polygon + optional world reference points,
    compute the real-world area in m².

    Args:
        pixel_polygon: List of pixel coords forming the monitoring zone polygon (4+ points)
        world_ref_points: Optional world-space reference corners for the 4 calibration pixel points.
    Returns:
        dict with 'area_m2', 'world_coords', 'homography'
    """
    if len(pixel_polygon) < 4:
        raise ValueError("At least 4 points required for calibration.")
    H = compute_homography(pixel_polygon[:4], world_ref_points)
    world_coords = pixel_to_world(H, pixel_polygon)
    area = polygon_area_meters(world_coords)
    return {
        "area_m2": round(area, 4),
        "world_coords": world_coords.tolist(),
        "homography": H.tolist(),
    }


# ------- Level of Service (LoS) Logic -------

def compute_los(n_people: int, area_m2: float) -> dict:
    """
    Compute crowd density and Fruin Level of Service.

    Fruin LoS thresholds (people/m²):
    LoS A-B: < 1.08    -> Normal Flow
    LoS C-D: 1.08-2.17 -> Crowd
    LoS E:   2.17-3.57 -> More Crowd  (Yellow Alert)
    LoS F:   > 3.57    -> Very Risky  (Red Alert)
    """
    if area_m2 <= 0:
        return {
            "density": 0.0,
            "los_level": "N/A",
            "risk_label": "No Area Defined",
            "alert_level": "none",
            "n_people": n_people,
            "area_m2": 0.0,
        }

    density = n_people / area_m2

    if density < 1.08:
        los_level, risk_label, alert_level = "A-B", "Normal Flow", "none"
    elif density < 2.17:
        los_level, risk_label, alert_level = "C-D", "Crowd", "notify"
    elif density < 3.57:
        los_level, risk_label, alert_level = "E", "More Crowd", "yellow"
    else:
        los_level, risk_label, alert_level = "F", "Very Risky Crowd", "red"

    return {
        "density": round(density, 4),
        "los_level": los_level,
        "risk_label": risk_label,
        "alert_level": alert_level,
        "n_people": n_people,
        "area_m2": round(area_m2, 4),
    }

