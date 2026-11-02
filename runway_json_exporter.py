import os
import json
import argparse
import cv2
from pathlib import Path
from runway_detector_core import RunwayDetectorCore


class RunwayJsonExporter:
    """
    Passive-Active bridge that calls RunwayDetectorCore and exports
    the resulting screen coordinates into a standardized .json file.
    """
    def __init__(self, detector: RunwayDetectorCore = None, output_dir="runs/coords_json"):
        self.detector = detector or RunwayDetectorCore()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_from_image(self, image_path, custom_json_path=None, conf_threshold=None):
        """Calls mother detector on an image file and writes the coordinates JSON."""
        scenario_id = Path(image_path).stem
        detection_dict = self.detector.detect(
            source_input=image_path,
            source_id=scenario_id,
            frame_index=0,
            conf_threshold=conf_threshold
        )

        out_path = custom_json_path or os.path.join(self.output_dir, f"{scenario_id}_coords.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detection_dict, f, indent=2)

        return out_path, detection_dict

    def export_from_sim_frame(self, frame_bgr, scenario_id, frame_index=0, custom_json_path=None, conf_threshold=None):
        """Calls mother detector on an in-memory simulation frame and saves the JSON."""
        detection_dict = self.detector.detect(
            source_input=frame_bgr,
            source_id=scenario_id,
            frame_index=frame_index,
            conf_threshold=conf_threshold
        )

        out_path = custom_json_path or os.path.join(
            self.output_dir, f"{scenario_id}_frame_{frame_index:05d}_coords.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detection_dict, f, indent=2)

        return out_path, detection_dict

    def export_from_video(self, video_path, conf_threshold=None):
        """Processes a video file and writes a JSON coordinate file per frame or bundle."""
        cap = cv2.VideoCapture(video_path)
        video_id = Path(video_path).stem
        video_out_dir = os.path.join(self.output_dir, video_id)
        os.makedirs(video_out_dir, exist_ok=True)

        frame_idx = 0
        saved_paths = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            json_path = os.path.join(video_out_dir, f"{video_id}_frame_{frame_idx:05d}_coords.json")
            self.export_from_sim_frame(frame, video_id, frame_idx, json_path, conf_threshold)
            saved_paths.append(json_path)

        cap.release()
        return saved_paths


# Active CLI Execution Mode (when run directly from terminal)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Runway Coordinates to JSON")
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--model", default="models/runway_pose_6k_best.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold")
    parser.add_argument("--outdir", default="runs/coords_json", help="Output directory for JSONs")
    args = parser.parse_args()

    core = RunwayDetectorCore(model_path=args.model, default_conf=args.conf)
    exporter = RunwayJsonExporter(detector=core, output_dir=args.outdir)

    ext = Path(args.input).suffix.lower()
    if ext in [".mp4", ".mkv", ".avi", ".mov"]:
        paths = exporter.export_from_video(args.input, conf_threshold=args.conf)
        print(f"Exported {len(paths)} frame JSONs to {args.outdir}")
    else:
        json_file, data = exporter.export_from_image(args.input, conf_threshold=args.conf)
        print(f"Exported coordinates JSON to: {json_file} | Detected: {data['runway_detected']}")