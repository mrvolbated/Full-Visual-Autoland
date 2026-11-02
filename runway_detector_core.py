import os
import logging
from datetime import datetime, timezone
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


class RunwayDetectorCore:
    """
    Passive Mother Script for Runway Pose Detection.
    Does not contain hardcoded inputs or visualization logic.
    Waits to be called with an image path or numpy frame array and returns
    screen-space 4-corner coordinates.
    """
    def __init__(self, model_path="models/runway_pose_6k_best.pt", default_conf=0.75, imgsz=1280, device=0):
        self.logger = logging.getLogger("RunwayDetectorCore.InputLogger")
        self.model_path = model_path
        self.default_conf = default_conf
        self.imgsz = imgsz
        self.device = device
        self.model = None

    def _ensure_model_loaded(self):
        """Lazy-loads the YOLO model only when first called."""
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
            self.logger.info(f"Loading YOLOv8-pose weights from: {self.model_path}")
            self.model = YOLO(self.model_path)

    def log_input(self, source_input, source_id=None, frame_index=None):
        """Logs and validates the incoming request from any external caller script."""
        input_type = "FILE_PATH" if isinstance(source_input, str) else "MEMORY_FRAME_ARRAY"
        resolved_id = source_id or (os.path.basename(source_input) if isinstance(source_input, str) else "sim_frame")
        self.logger.info(f"Triggered -> Type: {input_type} | ID: {resolved_id} | Frame: {frame_index}")
        return input_type, resolved_id

    def detect(self, source_input, source_id=None, frame_index=0, conf_threshold=None):
        """
        Main passive entry point called by external scripts.
        Returns a standardized dictionary of runway screen coordinates.
        """
        self._ensure_model_loaded()
        conf_gate = conf_threshold if conf_threshold is not None else self.default_conf
        input_type, resolved_id = self.log_input(source_input, source_id, frame_index)

        results = self.model.predict(
            source=source_input,
            imgsz=self.imgsz,
            conf=conf_gate,
            device=self.device,
            verbose=False
        )
        result = results[0]
        orig_h, orig_w = result.orig_shape

        payload = {
            "schema": "runway-screen-coords/v1.0",
            "source_id": resolved_id,
            "input_type": input_type,
            "frame_index": frame_index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "image_size_px": [int(orig_w), int(orig_h)],
            "conf_threshold_used": float(conf_gate),
            "runway_detected": False,
            "confidence": 0.0,
            "corners_px": None,
            "corners_ordered_list": None,
            "bounding_box_xyxy": None
        }

        # Fallback if no runway meets confidence gate
        if len(result.boxes) == 0 or result.keypoints is None or len(result.keypoints) == 0:
            return payload

        # Select highest-confidence runway candidate
        best_idx = int(result.boxes.conf.argmax().cpu().numpy())
        confidence = float(result.boxes.conf[best_idx].cpu().numpy())
        corners = result.keypoints.xy[best_idx].cpu().numpy().tolist()  # [NR, NL, FL, FR]
        bbox = result.boxes.xyxy[best_idx].cpu().numpy().tolist()

        payload.update({
            "runway_detected": True,
            "confidence": round(confidence, 4),
            "corners_px": {
                "near_right": [round(corners[0][0], 2), round(corners[0][1], 2)],
                "near_left":  [round(corners[1][0], 2), round(corners[1][1], 2)],
                "far_left":   [round(corners[2][0], 2), round(corners[2][1], 2)],
                "far_right":  [round(corners[3][0], 2), round(corners[3][1], 2)]
            },
            "corners_ordered_list": [
                [round(pt[0], 2), round(pt[1], 2)] for pt in corners
            ],
            "bounding_box_xyxy": [round(val, 2) for val in bbox]
        })

        return payload