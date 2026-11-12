import os
import time
import cv2
import mss
import numpy as np
import scipy.io as sio

from matlab_array_standardizer import MatlabArrayStandardizer
from runway_detector_core import RunwayDetectorCore
from runway_json_exporter import RunwayJsonExporter
from glide_path_core import GlidePathVectorCore
from evaluate_video_glidepath import render_hud_on_frame

PHASE_MAP = {"APPR": 1, "ALIGN": 2, "FLARE": 3, "ROLLOUT": 4}


def run_flightgear_closed_loop(
    matlab_state_mat_path="runs/live_arif/matlab_state_out.mat",
    guidance_out_mat_path="runs/live_arif/autoland_guidance_cmd.mat",
    model_path="models/runway_pose_batch2_best.pt",
    monitor_index=1,
    conf_threshold=0.45,
    show_hud=True
):
    os.makedirs("runs/live_arif", exist_ok=True)

    # 1. Initialize Passive Core Engines
    standardizer = MatlabArrayStandardizer(output_dir="runs/live_arif", ref_airport_icao="KSFO")
    detector = RunwayDetectorCore(model_path=model_path, default_conf=conf_threshold)
    detector.warmup()
    exporter = RunwayJsonExporter(detector=detector, output_dir="runs/live_arif")
    glide_solver = GlidePathVectorCore(target_glideslope_deg=3.0, touchdown_offset_m=300.0)

    sct = mss.mss()
    monitor = sct.monitors[monitor_index]

    frame_idx = 0
    prev_time = time.perf_counter()
    print("[FlightGear Autoland Loop] Active. Waiting for MATLAB state array & FlightGear window...")

    while True:
        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Step A: Capture FlightGear Simulation Tab
        sct_img = sct.grab(monitor)
        frame_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        if frame_bgr.shape[1] != 1920 or frame_bgr.shape[0] != 1080:
            frame_bgr = cv2.resize(frame_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)

        # Step B: Detect Runway Corners -> Export Coords JSON
        coords_json_path, coords_data = exporter.export_from_sim_frame(
            frame_bgr=frame_bgr,
            scenario_id="FlightGearLive",
            frame_index=frame_idx,
            custom_json_path="runs/live_arif/latest_coords.json",
            conf_threshold=conf_threshold
        )

        # Step C: Pull MATLAB Array -> Standardize Telemetry JSON
        if os.path.exists(matlab_state_mat_path):
            try:
                telem_json_path, _ = standardizer.pull_and_standardize_from_mat(
                    matlab_state_mat_path, scenario_id=f"FGFrame_{frame_idx:05d}"
                )
            except Exception:
                # Fallback if MATLAB is mid-write on the .mat file
                telem_json_path = "runs/live_arif/latest_telemetry_std.json"
        else:
            # Bootstrap dummy file if MATLAB hasn't written the first step yet
            telem_json_path, _ = standardizer.standardize_from_arrays(
                {"sim_array": np.array([75.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, -0.02, 2.05, -3500.0, 80.0, 185.0])},
                scenario_id="Bootstrap_000"
            )

        # Step D: Compute Glide Path & Phase State Machine
        t0 = time.perf_counter()
        guidance = glide_solver.compute_glide_path(
            telemetry_json_path=telem_json_path,
            coords_json_path=coords_json_path
        )
        core_ms = (time.perf_counter() - t0) * 1000.0

        # Step E: Export Guidance & Target Commands to MATLAB (.mat)
        if guidance.get("vision_guidance_valid", False):
            gp_vec = guidance["glide_path_vector"]
            vis_est = guidance["visual_estimates"]
            ils_dev = guidance["virtual_ils_deviations"]
            fma_str = gp_vec["active_fma_mode"]

            matlab_cmd_struct = {
                "valid": 1.0,
                "fma_phase_id": float(PHASE_MAP.get(fma_str, 1)),
                "dist_to_threshold_m": float(vis_est["distance_to_threshold_m"]),
                "dist_to_touchdown_m": float(vis_est["distance_to_touchdown_m"]),
                "lateral_offset_m": float(vis_est["lateral_offset_m"]),
                "relative_yaw_deg": float(vis_est["relative_yaw_to_runway_deg"]),
                "agl_m": float(vis_est["fused_agl_altitude_m"]),
                "loc_error_deg": float(ils_dev["localizer_error_deg"]),
                "gs_error_deg": float(ils_dev["glideslope_error_deg"]),
                "cmd_pitch_deg": float(gp_vec["commanded_pitch_deg"]),
                "cmd_sink_rate_mps": float(gp_vec["commanded_target_sink_rate_mps"]),
                "cmd_roll_bank_deg": float(gp_vec["commanded_roll_bank_deg"]),
                "cmd_heading_change_deg": float(gp_vec["commanded_heading_change_deg"]),
                "cmd_rudder_decrab_deg": float(gp_vec["commanded_rudder_decrab_deg"]),
                "cmd_autobrake_pct": float(gp_vec["commanded_autobrake_pct"])
            }
        else:
            matlab_cmd_struct = {"valid": 0.0, "fma_phase_id": 1.0}

        sio.savemat(guidance_out_mat_path, matlab_cmd_struct)

        # Step F: Live HUD Display
        if show_hud:
            render_hud_on_frame(frame_bgr, coords_data, guidance, core_ms, fps, frame_idx, stride=1)
            cv2.imshow("FlightGear + Arif Simulink Live Autoland HUD", cv2.resize(frame_bgr, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1


if __name__ == "__main__":
    run_flightgear_closed_loop()