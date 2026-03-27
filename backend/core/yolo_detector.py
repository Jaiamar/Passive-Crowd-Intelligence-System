"""
YOLO26 Detector - Wraps the latest Ultralytics YOLO model (YOLOv11/YOLOv8)
as a conceptual YOLO26 class for person detection.
"""
import numpy as np
from ultralytics import YOLO

YOLO26_WEIGHTS = "yolo11n.pt"  # Use latest available Ultralytics model


class YOLO26Detector:
    """
    YOLO26 person detector using the latest Ultralytics engine.
    Detects class 0 (person) in frames and returns bounding boxes.
    """

    def __init__(self, weights: str = YOLO26_WEIGHTS, conf_threshold: float = 0.4):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR frame.
        Returns list of dicts: {x1, y1, x2, y2, conf, cx, cy}
        """
        results = self.model(frame, conf=self.conf_threshold, classes=[0], verbose=False)
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detections.append({
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "conf": round(conf, 3),
                        "cx": int((x1 + x2) / 2),
                        "cy": int((y1 + y2) / 2),
                    })
        return detections

    def annotate_frame(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes on frame."""
        import cv2
        for d in detections:
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'{d["conf"]:.2f}',
                (d["x1"], d["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return frame
