import os
import cv2
import numpy as np
from runway_detector_core import RunwayDetectorCore

def visualize_single_image(image_path, model_path="models/runway_pose_6k_best.pt", conf_threshold=0.75):
    # Initialize passive mother script
    detector = RunwayDetectorCore(model_path=model_path, default_conf=conf_threshold)

    # Trigger mother script with image input
    result_dict = detector.detect(source_input=image_path)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return

    if result_dict["runway_detected"]:
        pts = np.array(result_dict["corners_ordered_list"], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        colors = [(0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0)]
        labels = ["NR", "NL", "FL", "FR"]
        for pt, col, lbl in zip(pts, colors, labels):
            cv2.circle(img, tuple(pt), 5, col, -1)
            cv2.putText(img, lbl, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        hud = f"RUNWAY LOCKED | Conf: {result_dict['confidence']:.2f}"
        cv2.putText(img, hud, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        hud = f"NO RUNWAY DETECTED (Conf < {conf_threshold})"
        cv2.putText(img, hud, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    os.makedirs("runs/inference", exist_ok=True)
    save_path = os.path.join("runs/inference", f"vis_{os.path.basename(image_path)}")
    cv2.imwrite(save_path, img)
    print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    TEST_IMG = r"C:\Users\User\Desktop\ILSData-main\ScenerioExports\weather\Broken_Clouds\usable\images\AutoScenario_001.png"
    visualize_single_image(TEST_IMG)