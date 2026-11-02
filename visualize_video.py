import os
import time
import cv2
import numpy as np
from runway_detector_core import RunwayDetectorCore

def visualize_landing_video(video_path, model_path="models/runway_pose_6k_best.pt", conf_threshold=0.75, show_live=True):
    detector = RunwayDetectorCore(model_path=model_path, default_conf=conf_threshold)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs("runs/inference", exist_ok=True)
    out_path = "runs/inference/video_visualized.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (width, height))

    frame_idx = 0
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # Call passive mother detector for current frame
        res = detector.detect(source_input=frame, source_id="video_stream", frame_index=frame_idx)

        if res["runway_detected"]:
            pts = np.array(res["corners_ordered_list"], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            hud = f"RUNWAY LOCKED | Conf: {res['confidence']:.2f} | FPS: {fps:.1f}"
            cv2.putText(frame, hud, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            hud = f"SEARCHING... | FPS: {fps:.1f}"
            cv2.putText(frame, hud, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)
        if show_live:
            cv2.imshow("Modular Runway Visualizer", cv2.resize(frame, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved annotated video to: {out_path}")

if __name__ == "__main__":
    VIDEO_FILE = "runway_video_raw_2026-09-07 11-08-00.mkv"
    visualize_landing_video(VIDEO_FILE)