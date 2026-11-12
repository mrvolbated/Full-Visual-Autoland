import os
import time
import json
import math
import cv2
import numpy as np

from standardize_telemetry import compute_camera_intrinsics
from runway_detector_core import RunwayDetectorCore
from runway_json_exporter import RunwayJsonExporter
from glide_path_core import GlidePathVectorCore


class OpticalStateFilter:
    """
    Extracts and smooths camera Pitch, Roll, and AGL Altitude directly from
    the 4 detected runway corners. Guarantees valid ground projection even when
    the camera is pitched nose-down and the runway sits above cy (Y < 540 px).
    """
    def __init__(self, alpha_attitude=0.45, alpha_agl=0.30):
        self.alpha_att = alpha_attitude
        self.alpha_agl = alpha_agl
        self.pitch_deg = -3.0
        self.roll_deg = 0.0
        self.agl_m = 180.0
        self.initialized = False

    def update_from_corners(self, corners_ordered_list, fx, fy, cx, cy, rwy_width_m=45.0, gs_deg=3.0):
        pts = np.array(corners_ordered_list, dtype=np.float64)
        nr, nl, fl, fr = pts[0], pts[1], pts[2], pts[3]

        # 1. Optical Roll from Near Threshold slope (NR -> NL)
        dx_t = nr[0] - nl[0]
        dy_t = nr[1] - nl[1]
        raw_roll = self.roll_deg
        if abs(dx_t) > 5.0:
            raw_roll = float(np.clip(-math.degrees(math.atan2(dy_t, dx_t)), -30.0, 30.0))

        # 2. Optical Pitch from Sideline Vanishing Point (Horizon Line)
        near_mid_y = 0.5 * (nr[1] + nl[1])
        far_mid_y = 0.5 * (fl[1] + fr[1])

        v_left = fl - nl
        v_right = fr - nr
        denom = v_left[0] * v_right[1] - v_left[1] * v_right[0]

        if abs(denom) > 1e-3:
            t_vp = ((nr[0] - nl[0]) * v_right[1] - (nr[1] - nl[1]) * v_right[0]) / denom
            vp_y = nl[1] + t_vp * v_left[1]
            # Horizon must logically sit above the far threshold on screen
            vp_y = min(vp_y, far_mid_y - 12.0)
        else:
            vp_y = far_mid_y - 35.0

        # Camera pitch angle from horizon vertical pixel position:
        # When vp_y < cy (horizon is in upper half of screen), camera is pitched DOWN (negative pitch)
        raw_pitch = float(np.clip(math.degrees(math.atan2(vp_y - cy, fy)), -15.0, 15.0))

        # CRITICAL SAFETY CONSTRAINT: Ensure both Near & Far thresholds sit below the optical horizon
        # in GlidePathVectorCore._project_pixel_to_ground (where down_dir = y_unrolled*cos(theta) - sin(theta))
        far_ray_angle_deg = math.degrees(math.atan2(far_mid_y - cy, fy))
        max_valid_pitch_deg = far_ray_angle_deg - 1.8  # Guarantees at least 1.8 deg downward grazing angle
        if raw_pitch > max_valid_pitch_deg:
            raw_pitch = max_valid_pitch_deg

        # 3. Dynamic AGL from optical threshold width
        w_near_px = max(float(np.linalg.norm(nr - nl)), 8.0)
        est_slant_m = (rwy_width_m * fx) / w_near_px
        raw_agl = max(est_slant_m * math.tan(math.radians(gs_deg)), 0.5)

        if not self.initialized:
            self.pitch_deg = raw_pitch
            self.roll_deg = raw_roll
            self.agl_m = raw_agl
            self.initialized = True
        else:
            self.pitch_deg = self.alpha_att * raw_pitch + (1.0 - self.alpha_att) * self.pitch_deg
            # Keep safety clamp active after smoothing
            if self.pitch_deg > max_valid_pitch_deg:
                self.pitch_deg = max_valid_pitch_deg
            self.roll_deg = self.alpha_att * raw_roll + (1.0 - self.alpha_att) * self.roll_deg
            self.agl_m = self.alpha_agl * raw_agl + (1.0 - self.alpha_agl) * self.agl_m

        return self.pitch_deg, self.roll_deg, self.agl_m


def create_dynamic_frame_telemetry_json(
    frame_idx, width_px, height_px, coords_dict, state_filter: OpticalStateFilter,
    out_json_path, v_fov_deg=28.555, rwy_width_m=45.0, rwy_length_m=3345.0, gs_deg=3.0
):
    intr = compute_camera_intrinsics(width_px, height_px, v_fov_deg)
    fx, fy = intr["fx_px"], intr["fy_px"]
    cx, cy = intr["cx_px"], intr["cy_px"]

    if coords_dict.get("runway_detected") and coords_dict.get("corners_ordered_list"):
        opt_pitch_deg, opt_roll_deg, est_agl_m = state_filter.update_from_corners(
            coords_dict["corners_ordered_list"], fx, fy, cx, cy, rwy_width_m, gs_deg
        )
    else:
        opt_pitch_deg = state_filter.pitch_deg
        opt_roll_deg = state_filter.roll_deg
        est_agl_m = state_filter.agl_m

    wow_flag = bool(est_agl_m <= 1.2)

    telemetry_payload = {
        "schema_version": "canonical-autoland-telemetry/v1.0",
        "meta": {
            "source_type": "VIDEO_STREAM_STRIDE_4",
            "scenario_id": f"VideoFrame_{frame_idx:05d}",
            "weather_condition": "Video_Capture"
        },
        "aircraft_state": {
            "latitude_deg": 48.7008,
            "longitude_deg": 9.2862,
            "altitude_msl_m": round(389.0 + est_agl_m, 3),
            "altitude_agl_m": round(est_agl_m, 3),
            "altitude_agl_ft": round(est_agl_m * 3.28084, 2),
            "pitch_deg": round(opt_pitch_deg, 4),
            "roll_deg": round(opt_roll_deg, 4),
            "heading_true_deg": 252.1,
            "tas_mps": 72.0,
            "ias_mps": 69.5,
            "beta_deg": 0.0,
            "weight_on_wheels": wow_flag,
            "hardware_ils_available": False,
            "hardware_mls_available": False
        },
        "camera_state": {
            "pitch_deg": round(opt_pitch_deg, 4),
            "raw_pitch_deg": round(opt_pitch_deg, 4),
            "roll_deg": round(opt_roll_deg, 4),
            "heading_true_deg": 252.1,
            "intrinsics": intr
        },
        "runway_reference": {
            "airport_icao": "EDDS",
            "runway_id": "25",
            "threshold_elevation_m": 389.0,
            "runway_width_m": rwy_width_m,
            "runway_length_m": rwy_length_m
        },
        "ground_truth_relative": {}
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_payload, f, indent=2)
    return out_json_path


def render_hud_on_frame(frame, coords_data, guidance_output, core_exec_ms, video_fps, frame_idx, stride):
    h, w = frame.shape[:2]
    cam_center = (int(w / 2), int(h / 2))

    # 1. Nose Bore-Sight Crosshair
    reticle_col = (220, 220, 220)
    cv2.circle(frame, cam_center, 7, reticle_col, 1, cv2.LINE_AA)
    cv2.line(frame, (cam_center[0] - 16, cam_center[1]), (cam_center[0] + 16, cam_center[1]), reticle_col, 1, cv2.LINE_AA)
    cv2.line(frame, (cam_center[0], cam_center[1] - 16), (cam_center[0], cam_center[1] + 16), reticle_col, 1, cv2.LINE_AA)

    if coords_data and coords_data.get("runway_detected") and guidance_output and guidance_output.get("vision_guidance_valid"):
        corners = np.array(coords_data["corners_ordered_list"], dtype=np.float64)
        corners_int = corners.astype(np.int32)

        # 2. Runway Polygon & Extended Centerline
        cv2.polylines(frame, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        near_mid = 0.5 * (corners[0] + corners[1])
        far_mid = 0.5 * (corners[2] + corners[3])
        rwy_dir = near_mid - far_mid
        ext_pt = near_mid + 2.2 * rwy_dir
        cv2.line(frame, (int(far_mid[0]), int(far_mid[1])), (int(ext_pt[0]), int(ext_pt[1])), (0, 140, 0), 1, cv2.LINE_AA)

        # 3. Two-Stage Ground Corridor
        screen_targets = guidance_output.get("screen_targets_px", {})
        appr_pts = screen_targets.get("appr_capture_corridor_px", [])
        align_pts = screen_targets.get("align_flare_centerline_px", [])

        if len(appr_pts) >= 2:
            appr_arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in appr_pts], dtype=np.int32)
            cv2.polylines(frame, [appr_arr], isClosed=False, color=(255, 220, 0), thickness=3, lineType=cv2.LINE_AA)

        if len(align_pts) >= 2:
            align_arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in align_pts], dtype=np.int32)
            cv2.polylines(frame, [align_arr], isClosed=False, color=(0, 180, 255), thickness=3, lineType=cv2.LINE_AA)
            gate_pt = tuple(align_arr[0])
            cv2.circle(frame, gate_pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "ALIGN GATE", (gate_pt[0] - 85, gate_pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Touchdown Bullseye
        td_raw = screen_targets.get("touchdown_point_px", near_mid.tolist())
        td_pt = (int(round(td_raw[0])), int(round(td_raw[1])))
        cv2.circle(frame, td_pt, 5, (0, 100, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, td_pt, 10, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "TDZ", (td_pt[0] + 14, td_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        # 5. Telemetry & FMA State Machine HUD
        phase = guidance_output["autoland_phase_state"]
        vis_est = guidance_output["visual_estimates"]
        gp_vec = guidance_output["glide_path_vector"]
        ils_dev = guidance_output["virtual_ils_deviations"]

        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (790, 290), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

        active_fma = phase["active_mode"]
        hud_lines = [
            (f"FMA MODE: [{active_fma}] | Calc Rate: 1/{stride} Frames (7.5 Hz) | Core: {core_exec_ms:.2f} ms | FPS: {video_fps:.1f}", (0, 255, 0)),
            (f"Phase Logic: {phase['phase_description']}", (180, 255, 255)),
            (f"Alt AGL: {vis_est['fused_agl_altitude_ft']:.0f} ft ({vis_est['fused_agl_altitude_m']:.1f} m) | WoW: {phase['wow_active']} (Poll: {phase['wow_polling_rate_ms']:.0f} ms)", (255, 255, 255)),
            (f"Dist to Threshold: {vis_est['distance_to_threshold_m']:.1f} m | Lat Offset: {vis_est['lateral_offset_m']:+.1f} m | Rel Yaw: {vis_est['relative_yaw_to_runway_deg']:+.2f} deg", (255, 255, 255)),
            (f"Virtual ILS -> LOC Err: {ils_dev['localizer_error_deg']:+.2f} deg | GS Err: {ils_dev['glideslope_error_deg']:+.2f} deg", (0, 215, 255)),
            (f"Cmds -> Pitch: {gp_vec['commanded_pitch_deg']:+.2f} deg | Bank: {gp_vec['commanded_roll_bank_deg']:+.1f} deg | Rudder: {gp_vec['commanded_rudder_decrab_deg']:+.1f} deg | Brake: {gp_vec['commanded_autobrake_pct']:.0f}%", (255, 200, 100))
        ]
        for i, (text, col) in enumerate(hud_lines):
            cv2.putText(frame, text, (35, 50 + i * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.49, col, 1, cv2.LINE_AA)
    else:
        status_msg = "RUNWAY DETECTED - COMPUTING..." if (coords_data and coords_data.get("runway_detected")) else "SEARCHING FOR RUNWAY..."
        cv2.putText(frame, f"{status_msg} | Frame: {frame_idx} | FPS: {video_fps:.1f}",
                    (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)


def run_video_glidepath_evaluation(
    video_path,
    model_path="models/runway_pose_batch2_best.pt",
    glidepath_stride=4,
    conf_threshold=0.45,
    show_live=True
):
    detector = RunwayDetectorCore(model_path=model_path, default_conf=conf_threshold)
    detector.warmup()

    exporter = RunwayJsonExporter(detector=detector, output_dir="runs/video_eval_jsons/coords")
    glide_solver = GlidePathVectorCore(target_glideslope_deg=3.0, touchdown_offset_m=300.0)
    state_filter = OpticalStateFilter(alpha_attitude=0.45, alpha_agl=0.30)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs("runs/inference", exist_ok=True)
    out_video_path = f"runs/inference/landing_video_autoland_stride{glidepath_stride}.mp4"
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    frame_idx = 0
    last_coords_data = None
    last_guidance_output = None
    last_core_exec_ms = 0.0
    prev_frame_time = time.perf_counter()

    print(f"Starting Video Autoland Evaluation (Computing GlidePath on 1 in every {glidepath_stride} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.perf_counter()
        inst_fps = 1.0 / max(now - prev_frame_time, 1e-6)
        prev_frame_time = now

        if frame_idx % glidepath_stride == 0:
            coords_json_path, last_coords_data = exporter.export_from_sim_frame(
                frame_bgr=frame,
                scenario_id="LandingVideo",
                frame_index=frame_idx,
                custom_json_path="runs/video_eval_jsons/latest_coords.json",
                conf_threshold=conf_threshold
            )

            telem_json_path = create_dynamic_frame_telemetry_json(
                frame_idx=frame_idx,
                width_px=width,
                height_px=height,
                coords_dict=last_coords_data,
                state_filter=state_filter,
                out_json_path="runs/video_eval_jsons/latest_telemetry_std.json"
            )

            t0 = time.perf_counter()
            last_guidance_output = glide_solver.compute_glide_path(
                telemetry_json_path=telem_json_path,
                coords_json_path=coords_json_path
            )
            last_core_exec_ms = (time.perf_counter() - t0) * 1000.0

        render_hud_on_frame(
            frame=frame,
            coords_data=last_coords_data,
            guidance_output=last_guidance_output,
            core_exec_ms=last_core_exec_ms,
            video_fps=inst_fps,
            frame_idx=frame_idx,
            stride=glidepath_stride
        )

        writer.write(frame)
        if show_live:
            cv2.imshow(f"Autoland 1-in-{glidepath_stride} Frame Video Verification", cv2.resize(frame, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Finished! Saved full annotated Autoland video to: {out_video_path}")


if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\User\Desktop\ILSData-main\runway_video_raw_2026-09-07 11-08-00.mkv"
    MODEL_PATH = r"C:\Users\User\Desktop\ILSData-main\models\runway_pose_batch2_best.pt"
    run_video_glidepath_evaluation(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        glidepath_stride=4,
        conf_threshold=0.45,
        show_live=True
    )