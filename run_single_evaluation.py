import os
import time
import json
import cv2
import numpy as np

from standardize_telemetry import parse_msfs_v3_json
from runway_detector_core import RunwayDetectorCore
from runway_json_exporter import RunwayJsonExporter
from glide_path_core import GlidePathVectorCore


def draw_bezier_arc(img, p_start, p_control, p_end, color=(255, 215, 0), thickness=3, num_steps=60):
    """
    Draws a smooth Quadratic Bezier curve from p_start (camera center)
    to p_end (runway touchdown point).
    """
    t_vals = np.linspace(0.0, 1.0, num_steps)
    curve_points = []
    for t in t_vals:
        x = (1 - t)**2 * p_start[0] + 2 * (1 - t) * t * p_control[0] + (t**2) * p_end[0]
        y = (1 - t)**2 * p_start[1] + 2 * (1 - t) * t * p_control[1] + (t**2) * p_end[1]
        curve_points.append([int(round(x)), int(round(y))])

    pts_array = np.array(curve_points, dtype=np.int32)
    cv2.polylines(img, [pts_array], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Directional chevrons along the trajectory
    for idx in [int(num_steps * 0.35), int(num_steps * 0.70)]:
        if idx + 1 < len(curve_points):
            pt1 = tuple(curve_points[idx])
            pt2 = tuple(curve_points[idx + 1])
            cv2.arrowedLine(img, pt1, pt2, color, thickness, cv2.LINE_AA, tipLength=0.4)


def visualize_glide_path_verification(image_path, coords_json_path, guidance_output, exec_time_ms, output_path):
    """
    Renders the Multi-Phase Autoland HUD (APPR -> ALIGN -> FLARE -> ROLLOUT)
    and the distortion-free 2-stage ground capture corridor onto the runway centerline.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    with open(coords_json_path, "r", encoding="utf-8") as f:
        coords_data = json.load(f)

    h, w = img.shape[:2]
    cam_center = (int(w / 2), int(h / 2))

    # 1. Draw Aircraft Nose Bore-Sight Crosshair (Horizon Reference)
    reticle_col = (220, 220, 220)
    cv2.circle(img, cam_center, 7, reticle_col, 1, cv2.LINE_AA)
    cv2.line(img, (cam_center[0] - 16, cam_center[1]), (cam_center[0] + 16, cam_center[1]), reticle_col, 1, cv2.LINE_AA)
    cv2.line(img, (cam_center[0], cam_center[1] - 16), (cam_center[0], cam_center[1] + 16), reticle_col, 1, cv2.LINE_AA)
    cv2.putText(img, "NOSE HDG", (cam_center[0] + 12, cam_center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, reticle_col, 1, cv2.LINE_AA)

    if coords_data.get("runway_detected") and guidance_output.get("vision_guidance_valid"):
        corners = np.array(coords_data["corners_ordered_list"], dtype=np.float64)
        corners_int = corners.astype(np.int32)

        # 2. Draw Runway Boundary Polygon & Extended Centerline Reference
        cv2.polylines(img, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        near_mid = 0.5 * (corners[0] + corners[1])
        far_mid = 0.5 * (corners[2] + corners[3])

        rwy_screen_dir = near_mid - far_mid
        extended_pt = near_mid + 2.2 * rwy_screen_dir
        cv2.line(img, (int(far_mid[0]), int(far_mid[1])),
                 (int(extended_pt[0]), int(extended_pt[1])),
                 (0, 140, 0), 1, cv2.LINE_AA)

        # 3. Draw Segmented Autoland Ground Corridor (APPR Capture -> ALIGN/FLARE Straight-In)
        screen_targets = guidance_output.get("screen_targets_px", {})
        appr_pts = screen_targets.get("appr_capture_corridor_px", [])
        align_pts = screen_targets.get("align_flare_centerline_px", [])

        if len(appr_pts) >= 2:
            appr_arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in appr_pts], dtype=np.int32)
            cv2.polylines(img, [appr_arr], isClosed=False, color=(255, 220, 0), thickness=3, lineType=cv2.LINE_AA)
            # Label start of APPR capture corridor
            cv2.putText(img, "APPR LOC CAPTURE", (appr_arr[0][0] + 10, appr_arr[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1, cv2.LINE_AA)

        if len(align_pts) >= 2:
            align_arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in align_pts], dtype=np.int32)
            cv2.polylines(img, [align_arr], isClosed=False, color=(0, 180, 255), thickness=3, lineType=cv2.LINE_AA)
            # Mark Alignment Gate where APPR hands off to ALIGN along centerline
            gate_pt = tuple(align_arr[0])
            cv2.circle(img, gate_pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "ALIGN GATE", (gate_pt[0] - 85, gate_pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Draw Touchdown Zone Bullseye
        td_pt_raw = screen_targets.get("touchdown_point_px", near_mid.tolist())
        touchdown_pt = (int(round(td_pt_raw[0])), int(round(td_pt_raw[1])))
        cv2.circle(img, touchdown_pt, 5, (0, 100, 255), -1, cv2.LINE_AA)
        cv2.circle(img, touchdown_pt, 10, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "TDZ (FLARE->ROLLOUT)", (touchdown_pt[0] + 14, touchdown_pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 165, 255), 1, cv2.LINE_AA)

        # 5. Draw Flight Mode Annunciator (FMA) & Telemetry HUD
        phase = guidance_output["autoland_phase_state"]
        vis_est = guidance_output["visual_estimates"]
        gp_vec = guidance_output["glide_path_vector"]
        ils_dev = guidance_output["virtual_ils_deviations"]
        gt_cmp = guidance_output.get("ground_truth_comparison")

        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (770, 320), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.76, img, 0.24, 0, img)

        active_fma = phase["active_mode"]
        hud_lines = [
            (f"FMA MODE: [{active_fma}] | Source: {phase['guidance_source']} | Core: {exec_time_ms:.2f} ms", (0, 255, 0)),
            (f"Phase Logic: {phase['phase_description']}", (180, 255, 255)),
            (f"Alt AGL: {vis_est['fused_agl_altitude_ft']:.0f} ft ({vis_est['fused_agl_altitude_m']:.1f} m) | WoW Sensor: {phase['wow_active']} (Poll: {phase['wow_polling_rate_ms']:.0f} ms)", (255, 255, 255)),
            (f"Dist to Threshold: {vis_est['distance_to_threshold_m']:.1f} m | Lat Offset: {vis_est['lateral_offset_m']:+.1f} m | Rel Yaw: {vis_est['relative_yaw_to_runway_deg']:+.2f} deg", (255, 255, 255)),
            (f"Virtual ILS -> LOC Err: {ils_dev['localizer_error_deg']:+.2f} deg | GS Err: {ils_dev['glideslope_error_deg']:+.2f} deg", (0, 215, 255)),
            (f"Cmds -> Pitch: {gp_vec['commanded_pitch_deg']:+.2f} deg | Bank: {gp_vec['commanded_roll_bank_deg']:+.1f} deg | Rudder Decrab: {gp_vec['commanded_rudder_decrab_deg']:+.1f} deg | Brake: {gp_vec['commanded_autobrake_pct']:.0f}%", (255, 200, 100))
        ]

        if gt_cmp:
            hud_lines.append((
                f"GT Comparison -> Dist Err: {gt_cmp['distance_error_m']:+.1f} m ({gt_cmp['distance_error_pct']:.1f}%)",
                (180, 255, 180)
            ))

        for i, (text, col) in enumerate(hud_lines):
            cv2.putText(img, text, (35, 50 + i * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.49, col, 1, cv2.LINE_AA)

    else:
        cv2.putText(img, f"NO VISUAL GLIDE PATH LOCK | Core Time: {exec_time_ms:.2f} ms",
                    (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved Glide Path Verification Image to: {output_path}")
if __name__ == "__main__":
    # Corrected paths pointing to Heavy_Rain for AutoScenario_001
    WEATHER_FOLDER = "Broken_Clouds"
    RAW_TELEMETRY = rf"C:\Users\User\Desktop\ILSData-main\ScenerioExports\weather\{WEATHER_FOLDER}\usable\telemetry\AutoScenario_405.json"
    IMAGE_PATH = rf"C:\Users\User\Desktop\ILSData-main\ScenerioExports\weather\{WEATHER_FOLDER}\usable\images\AutoScenario_405.png"
    MODEL_PATH = r"C:\Users\User\Desktop\ILSData-main\models\runway_pose_batch2_best.pt"

    os.makedirs("runs/std_telemetry", exist_ok=True)
    STD_TELEMETRY_JSON = "runs/std_telemetry/AutoScenario_405_telemetry_std.json"

    # Step A: Standardize telemetry JSON
    std_telemetry_data = parse_msfs_v3_json(RAW_TELEMETRY, WEATHER_FOLDER)
    with open(STD_TELEMETRY_JSON, "w", encoding="utf-8") as f:
        json.dump(std_telemetry_data, f, indent=2)

    # Step B: Export runway coordinates via passive mother detector
    detector = RunwayDetectorCore(model_path=MODEL_PATH, default_conf=0.75)
    exporter = RunwayJsonExporter(detector=detector, output_dir="runs/coords_json")
    coords_json_path, _ = exporter.export_from_image(IMAGE_PATH)

    # Step C: Compute Glide Path to Touchdown Point & time execution
    glide_solver = GlidePathVectorCore(target_glideslope_deg=3.0, touchdown_offset_m=300.0)

    t_start = time.perf_counter()
    guidance_output = glide_solver.compute_glide_path(
        telemetry_json_path=STD_TELEMETRY_JSON,
        coords_json_path=coords_json_path
    )
    t_end = time.perf_counter()

    glide_core_time_ms = (t_end - t_start) * 1000.0
    guidance_output["glide_path_core_execution_ms"] = round(glide_core_time_ms, 4)

    print(f"\n[TIMING] GlidePathVectorCore finished in {glide_core_time_ms:.3f} ms")
    print("\n=== COMPUTED GLIDE PATH & POSE OUTPUT ===")
    print(json.dumps(guidance_output, indent=2))

    # Step D: Render Verification Image with Arc terminating at Touchdown Point
    VIS_OUTPUT_PATH = "runs/inference/AutoScenario_405_glide_path_verified.png"
    visualize_glide_path_verification(
        image_path=IMAGE_PATH,
        coords_json_path=coords_json_path,
        guidance_output=guidance_output,
        exec_time_ms=glide_core_time_ms,
        output_path=VIS_OUTPUT_PATH
    )