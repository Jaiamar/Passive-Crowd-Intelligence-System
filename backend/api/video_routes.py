"""
Video API Routes - WebSocket endpoint for real-time YOLO person detection.
Streams YOLO-annotated frames and crowd analytics to the frontend.
Defaults to webcam (device 0). Falls back to demo.mp4 if DEMO_VIDEO env var is set.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.requests import Request

from core.homography import calibrate_area, compute_los
from core.yolo_detector import YOLO26Detector

router = APIRouter()
detector = YOLO26Detector()

# Demo video override (set DEMO_VIDEO env var to a .mp4 path to use a file instead)
DEMO_VIDEO_PATH = os.environ.get("DEMO_VIDEO", "")

# Camera device index override. Set CAMERA_DEVICE=0 for laptop webcam, 1 for Iriun, etc.
# Defaults to -1 which means auto-detect (picks the highest available index so Iriun wins).
_cam_env = os.environ.get("CAMERA_DEVICE", "").strip()
CAMERA_DEVICE = int(_cam_env) if _cam_env else -1


def find_best_camera() -> int:
    """
    Scan device indices 0-4 and return the highest one that opens successfully.
    Iriun registers after the built-in webcam, so the highest index = Iriun.
    Returns -1 if no camera found.
    """
    if CAMERA_DEVICE >= 0:
        print(f"[VideoWS] Using fixed camera device: {CAMERA_DEVICE}")
        return CAMERA_DEVICE
    best = -1
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            best = idx
            print(f"[VideoWS] Camera device {idx} available")
        cap.release()
    print(f"[VideoWS] Auto-selected camera device: {best}")
    return best

# Per-session calibration data
calibration_store: dict[str, dict] = {}


@router.get("/api/cameras")
async def list_cameras():
    """List all available camera devices (index 0-4)."""
    available = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append({
                "index": idx,
                "label": f"Device {idx}" + (" (Built-in / Laptop Webcam)" if idx == 0 else f" (External / Iriun #{idx})"),
            })
            cap.release()
    return {"cameras": available, "auto_selected": find_best_camera()}


@router.post("/api/calibrate")
async def calibrate_zone(request: Request):
    """
    Accept pixel polygon + optional world reference points.
    Returns real-world area (m²) and homography matrix.
    """
    try:
        payload = await request.json()
        pixel_polygon = payload.get("pixel_polygon", [])
        world_ref = payload.get("world_ref_points", None)
        if len(pixel_polygon) < 4:
            return {"error": "At least 4 points required for calibration"}
        result = calibrate_area(pixel_polygon, world_ref)
        session_id = payload.get("session_id", "default")
        calibration_store[session_id] = result
        return {"status": "ok", **result}
    except Exception as e:
        return {"error": str(e)}


def count_in_polygon(detections: list[dict], polygon: list[list[float]]) -> int:
    """Count detections whose centroid falls inside the polygon."""
    if not polygon or len(polygon) < 3:
        return len(detections)
    pts = np.array(polygon, dtype=np.int32)
    count = 0
    for d in detections:
        if cv2.pointPolygonTest(pts, (float(d["cx"]), float(d["cy"])), False) >= 0:
            count += 1
    return count


def draw_overlay(
    frame: np.ndarray,
    n_in_zone: int,
    total: int,
    los: dict,
    polygon: list[list[float]],
) -> np.ndarray:
    """Draw YOLO detection overlay, polygon, and LoS info onto frame."""
    h, w = frame.shape[:2]

    # Draw calibration polygon
    if len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 100, 255))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 140, 255), 2)
        for pt in polygon:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 200, 255), -1)

    # Alert color
    color_map = {
        "none":   (0, 220, 100),
        "notify": (0, 165, 255),
        "yellow": (0, 210, 255),
        "red":    (0, 60, 255),
    }
    alert_color = color_map.get(los.get("alert_level", "none"), (255, 255, 255))

    # Top-left info panel
    panel_h, panel_w = 140, 480
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), alert_color, 1)

    texts = [
        (f"YOLO26  |  Total Detected: {total}", (255, 255, 255), 0.75, 2),
        (f"In Zone: {n_in_zone}  |  Area: {los.get('area_m2', 100):.1f} m2", (200, 200, 200), 0.70, 1),
        (f"Density: {los.get('density', 0):.4f} p/m2", (200, 200, 200), 0.70, 1),
        (f"LoS {los.get('los_level','N/A')} - {los.get('risk_label','')}", alert_color, 0.78, 2),
    ]
    for i, (txt, col, scale, thickness) in enumerate(texts):
        cv2.putText(frame, txt, (12, 28 + i * 30),
                    cv2.FONT_HERSHEY_DUPLEX, scale, col, thickness, cv2.LINE_AA)

    # Bottom-right: big person count badge
    badge_text = f"{n_in_zone}"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 3.0, 4)
    bx, by = w - tw - 24, h - 20
    cv2.putText(frame, badge_text, (bx, by),
                cv2.FONT_HERSHEY_DUPLEX, 3.0, alert_color, 4, cv2.LINE_AA)
    cv2.putText(frame, "PEOPLE", (bx, by + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


@router.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    """
    Main WebSocket stream endpoint.
    - Reads from webcam (device 0) by default, or DEMO_VIDEO file if set.
    - Runs YOLO26 on every frame.
    - Accepts JSON commands: { type: 'calibrate', polygon: [...], area_m2: float }
    - Sends: { frame (base64 JPEG), detections, n_in_zone, total_detections, los, area_m2 }
    """
    await websocket.accept()

    # Determine video source
    if DEMO_VIDEO_PATH and os.path.exists(DEMO_VIDEO_PATH):
        cap = cv2.VideoCapture(DEMO_VIDEO_PATH)
        source_label = f"VIDEO: {DEMO_VIDEO_PATH}"
        webcam_ok = cap.isOpened()
    else:
        device_idx = find_best_camera()
        if device_idx < 0:
            cap = None
            webcam_ok = False
            source_label = "No camera found"
        else:
            cap = cv2.VideoCapture(device_idx)
            webcam_ok = cap.isOpened()
            source_label = f"Camera device {device_idx}"

    if webcam_ok and cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[VideoWS] Source opened: {source_label}")
    else:
        print(f"[VideoWS] WARNING: Could not open {source_label}. Sending placeholder frames.")

    current_polygon: list[list[float]] = []
    current_area_m2: float = 100.0

    try:
        while True:
            # ── Non-blocking command receive ─────────────────────────
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                msg = json.loads(raw)
                if msg.get("type") == "calibrate":
                    current_polygon = msg.get("polygon", [])
                    current_area_m2 = float(msg.get("area_m2", 100.0))
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            # ── Grab frame ───────────────────────────────────────────
            if webcam_ok:
                ret, frame = cap.read()
                if not ret:
                    # For video files: loop; for webcam: wait
                    if DEMO_VIDEO_PATH:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.05)
                        continue
            else:
                # Placeholder frame
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "No camera / video source found.",
                    (80, 360),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (80, 80, 200), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Set DEMO_VIDEO env var or connect a webcam.",
                    (80, 410),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 1, cv2.LINE_AA,
                )

            # ── YOLO detection ───────────────────────────────────────
            detections = detector.detect(frame)
            frame = detector.annotate_frame(frame, detections)

            # ── Zone counting & LoS ──────────────────────────────────
            n_in_zone = count_in_polygon(detections, current_polygon)
            total = len(detections)
            los = compute_los(n_in_zone, current_area_m2)

            # ── Draw overlay ─────────────────────────────────────────
            frame = draw_overlay(frame, n_in_zone, total, los, current_polygon)

            # ── Encode & send ────────────────────────────────────────
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64 = base64.b64encode(buf).decode("utf-8")

            await websocket.send_text(json.dumps({
                "frame": b64,
                "detections": detections,
                "n_in_zone": n_in_zone,
                "total_detections": total,
                "los": los,
                "area_m2": current_area_m2,
            }))

            # ~15 FPS
            await asyncio.sleep(0.066)

    except WebSocketDisconnect:
        print("[VideoWS] Client disconnected.")
    finally:
        if webcam_ok and cap is not None:
            cap.release()

