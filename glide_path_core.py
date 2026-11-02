import os
import json
import math
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

GRAVITY_MPS2 = 9.80665
M_TO_FT = 3.28084
FT_TO_M = 0.3048


class GlidePathVectorCore:
    """
    Passive Mother Script for Multi-Phase Autoland Guidance & Pose Estimation.
    Operates in 4 Flight Control Modes based on AGL & Weight-on-Wheels (WoW):
      1. APPR    (> 150 ft AGL): Visual Localizer & Glideslope capture toward Alignment Gate
      2. ALIGN   (150 ft -> 50 ft AGL): Pitot/Air-Data aided decrab (Relative Yaw -> 0 deg)
      3. FLARE   (50 ft -> 0 ft AGL, WoW=False): Exponential sink-rate decay & 1-10ms WoW polling
      4. ROLLOUT (WoW=True): Autobrake regulation & visual centerline rudder steering
    """
    def __init__(
        self,
        target_glideslope_deg=3.0,
        touchdown_offset_m=300.0,
        align_gate_before_threshold_m=450.0,
        default_approach_tas_mps=72.0,
        max_bank_deg=25.0
    ):
        self.telemetry_logger = logging.getLogger("GlidePathCore.TelemetryInputLogger")
        self.coords_logger = logging.getLogger("GlidePathCore.CoordsInputLogger")
        self.target_gs_deg = target_glideslope_deg
        self.tdz_offset_m = touchdown_offset_m
        self.align_gate_m = align_gate_before_threshold_m
        self.default_tas_mps = default_approach_tas_mps
        self.max_bank_deg = max_bank_deg

    def log_telemetry_input(self, telemetry_json_path):
        if not os.path.exists(telemetry_json_path):
            raise FileNotFoundError(f"Telemetry JSON not found: {telemetry_json_path}")
        self.telemetry_logger.info(f"Loaded Telemetry JSON <- {telemetry_json_path}")
        with open(telemetry_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def log_coords_input(self, coords_json_path):
        if not os.path.exists(coords_json_path):
            raise FileNotFoundError(f"Runway Coords JSON not found: {coords_json_path}")
        self.coords_logger.info(f"Loaded Runway Coords JSON <- {coords_json_path}")
        with open(coords_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _project_pixel_to_ground(u_px, v_px, fx, fy, cx, cy, pitch_deg, roll_deg, agl_m):
        """
        Projects 2D pixel (u, v) onto the horizontal runway plane at depth agl_m below aircraft.
        Returns [forward_m, right_m] in the aircraft's horizontal body-heading frame.
        """
        x_norm = (u_px - cx) / fx
        y_norm = (v_px - cy) / fy
        z_norm = 1.0

        phi = math.radians(roll_deg)
        theta = math.radians(pitch_deg)

        x_unrolled = x_norm * math.cos(phi) - y_norm * math.sin(phi)
        y_unrolled = x_norm * math.sin(phi) + y_norm * math.cos(phi)

        forward_dir = z_norm * math.cos(theta) + y_unrolled * math.sin(theta)
        down_dir = y_unrolled * math.cos(theta) - z_norm * math.sin(theta)

        if down_dir <= 1e-5:
            return None

        scale = agl_m / down_dir
        return np.array([scale * forward_dir, scale * x_unrolled], dtype=np.float64)

    @staticmethod
    def _project_ground_frame_to_pixel(forward_m, right_m, down_m, fx, fy, cx, cy, pitch_deg, roll_deg):
        """
        Exact mathematical inverse of _project_pixel_to_ground.
        Projects a 3D point [forward_m, right_m, down_m] relative to the aircraft
        back onto [u_px, v_px] with zero PnP rotational distortion.
        """
        if forward_m <= 20.0:
            return None

        phi = math.radians(roll_deg)
        theta = math.radians(pitch_deg)

        # Rotate from local horizontal (forward, down) back into pitched camera frame
        z_cam = forward_m * math.cos(theta) - down_m * math.sin(theta)
        y_unrolled_cam = forward_m * math.sin(theta) + down_m * math.cos(theta)
        x_unrolled_cam = right_m

        if z_cam <= 20.0:
            return None

        x_norm_unrolled = x_unrolled_cam / z_cam
        y_norm_unrolled = y_unrolled_cam / z_cam

        # Re-apply aircraft roll (inverse of un-roll)
        x_norm = x_norm_unrolled * math.cos(phi) + y_norm_unrolled * math.sin(phi)
        y_norm = -x_norm_unrolled * math.sin(phi) + y_norm_unrolled * math.cos(phi)

        u_px = fx * x_norm + cx
        v_px = fy * y_norm + cy
        return [round(float(u_px), 2), round(float(v_px), 2)]

    def _determine_autoland_phase(self, agl_m, wow_flag, hw_ils, hw_mls):
        """
        Determines active Autoland Flight Mode Annunciator (FMA) phase from AGL & WoW.
        """
        agl_ft = agl_m * M_TO_FT
        guidance_source = "HARDWARE_ILS_MLS" if (hw_ils or hw_mls) else "VISION_VIRTUAL_ILS"

        if wow_flag or agl_ft <= 1.5:
            return {
                "active_mode": "ROLLOUT",
                "guidance_source": guidance_source,
                "agl_ft": round(agl_ft, 1),
                "wow_active": True,
                "wow_polling_rate_ms": 2.0,
                "phase_description": "Touchdown confirmed via WoW; regulating autobrake & visual centerline rollout."
            }
        elif agl_ft <= 50.0:
            return {
                "active_mode": "FLARE",
                "guidance_source": guidance_source,
                "agl_ft": round(agl_ft, 1),
                "wow_active": False,
                "wow_polling_rate_ms": 2.0,  # High-frequency 1-10ms WoW squat-switch monitoring
                "phase_description": "Below 50 ft AGL; arresting sink rate exponentially & monitoring WoW (2 ms)."
            }
        elif agl_ft <= 150.0:
            return {
                "active_mode": "ALIGN",
                "guidance_source": guidance_source,
                "agl_ft": round(agl_ft, 1),
                "wow_active": False,
                "wow_polling_rate_ms": 10.0,
                "phase_description": "Below 150 ft AGL; decrabbing aircraft nose to runway axis (Yaw -> 0 deg) using Pitot/Air Data."
            }
        else:
            return {
                "active_mode": "APPR",
                "guidance_source": guidance_source,
                "agl_ft": round(agl_ft, 1),
                "wow_active": False,
                "wow_polling_rate_ms": 50.0,
                "phase_description": "LOC & GS capture active; intercepting extended runway centerline ahead of Alignment Gate."
            }

    def _build_segmented_ground_corridor_px(
        self, p_near_ac, p_far_ac, u_rwy, n_rwy_right, x_ac_rwy, y_ac_rwy, agl_m,
        fx, fy, cx, cy, pitch_deg, roll_deg
    ):
        """
        Generates distortion-free 2D screen waypoints for the segmented Autoland path:
          Segment 1 (APPR): Curved lateral capture from ahead of aircraft onto the Alignment Gate
                            (450m before threshold on the extended runway centerline).
          Segment 2 (ALIGN -> FLARE -> TDZ): Straight stabilized corridor along the extended
                            runway centerline through the Threshold into the Touchdown Zone (Yaw = 0 deg).
        """
        # Runway-frame X positions (Threshold = 0.0, Aircraft = x_ac_rwy < 0)
        x_align_gate = -min(self.align_gate_m, abs(x_ac_rwy) * 0.45)  # e.g., -450 m before threshold
        x_tdz = self.tdz_offset_m

        # Start visualizing the corridor on the ground 450m ahead of aircraft (or halfway to gate)
        x_vis_start = max(x_ac_rwy + 500.0, x_ac_rwy * 0.82)
        if x_vis_start >= x_align_gate:
            x_vis_start = x_ac_rwy * 0.6

        appr_curve_px = []
        align_flare_line_px = []

        # 1. APPR Capture Segment (x_vis_start -> x_align_gate)
        # Smooth cosine/Hermite S-turn that eliminates lateral offset y_ac_rwy and arrives parallel at x_align_gate
        for x in np.linspace(x_vis_start, x_align_gate, 25):
            span = max(x_align_gate - x_ac_rwy, 100.0)
            s = np.clip((x - x_ac_rwy) / span, 0.0, 1.0)
            # Cosine ease-out so dy/dx = 0 (parallel to centerline) right at the Alignment Gate
            y = y_ac_rwy * 0.5 * (1.0 + math.cos(math.pi * s))

            # Convert (x, y) in Runway Frame back to Aircraft Horizontal Frame:
            # P_ac = p_near_ac + x * u_rwy + y * n_rwy_right
            pt_ac = p_near_ac + (x * u_rwy) + (y * n_rwy_right)
            px = self._project_ground_frame_to_pixel(
                pt_ac[0], pt_ac[1], agl_m, fx, fy, cx, cy, pitch_deg, roll_deg
            )
            if px is not None:
                appr_curve_px.append(px)

        # 2. ALIGN & FLARE Stabilized Segment (x_align_gate -> Threshold (0) -> TDZ (+300m))
        # Strictly on the runway centerline (y = 0.0, relative yaw = 0.0 deg)
        for x in np.linspace(x_align_gate, x_tdz, 15):
            pt_ac = p_near_ac + (x * u_rwy)
            px = self._project_ground_frame_to_pixel(
                pt_ac[0], pt_ac[1], agl_m, fx, fy, cx, cy, pitch_deg, roll_deg
            )
            if px is not None:
                align_flare_line_px.append(px)

        # Also project the exact Velocity Vector / Flight Path Marker on the ground ahead
        fpa_pt_px = self._project_ground_frame_to_pixel(
            abs(x_ac_rwy), 0.0, agl_m, fx, fy, cx, cy, pitch_deg, roll_deg
        )

        return appr_curve_px, align_flare_line_px, fpa_pt_px

    def compute_glide_path(self, telemetry_json_path, coords_json_path):
        telemetry = self.log_telemetry_input(telemetry_json_path)
        coords = self.log_coords_input(coords_json_path)

        scenario_id = telemetry.get("meta", {}).get("scenario_id", coords.get("source_id", "unknown"))

        if not coords.get("runway_detected", False) or coords.get("corners_ordered_list") is None:
            return {
                "scenario_id": scenario_id,
                "status": "NO_VISUAL_LOCK",
                "vision_guidance_valid": False,
                "message": "Runway not detected in frame; fall back to onboard inertial/baro hold."
            }

        # 1. Camera Intrinsics & Aircraft Sensor State
        intr = telemetry["camera_state"]["intrinsics"]
        fx, fy = float(intr["fx_px"]), float(intr["fy_px"])
        cx, cy = float(intr["cx_px"]), float(intr["cy_px"])

        ac_state = telemetry["aircraft_state"]
        onboard_agl_m = float(ac_state.get("altitude_agl_m", 150.0))
        cam_pitch = float(telemetry["camera_state"].get("raw_pitch_deg", telemetry["camera_state"].get("pitch_deg", 0.0)))
        cam_roll = float(telemetry["camera_state"].get("roll_deg", 0.0))

        tas_mps = float(ac_state.get("tas_mps") or self.default_tas_mps)
        ias_mps = float(ac_state.get("ias_mps") or (tas_mps * 0.96))
        vs_mps = float(ac_state.get("vertical_speed_mps") or (-tas_mps * math.tan(math.radians(self.target_gs_deg))))
        beta_deg = float(ac_state.get("beta_deg") or 0.0)
        wow_flag = bool(ac_state.get("weight_on_wheels", False))
        hw_ils = bool(ac_state.get("hardware_ils_available", False))
        hw_mls = bool(ac_state.get("hardware_mls_available", False))

        # 2. Runway Physical Dimensions & Detected Corners [NR, NL, FL, FR]
        rwy_ref = telemetry["runway_reference"]
        L = float(rwy_ref["runway_length_m"])
        image_points_2d = np.array(coords["corners_ordered_list"], dtype=np.float64)

        near_mid_px = 0.5 * (image_points_2d[0] + image_points_2d[1])
        far_mid_px = 0.5 * (image_points_2d[2] + image_points_2d[3])

        # Interpolate Touchdown Point along perspective centerline (tdz_offset_m / L)
        # Accounting for projective perspective foreshortening using near/far threshold pixel widths
        w_near_px = max(np.linalg.norm(image_points_2d[0] - image_points_2d[1]), 1.0)
        w_far_px = max(np.linalg.norm(image_points_2d[3] - image_points_2d[2]), 1.0)
        lam = self.tdz_offset_m / max(L, 1.0)
        persp_t = (lam * w_near_px) / ((1.0 - lam) * w_far_px + lam * w_near_px)
        touchdown_px = (near_mid_px + persp_t * (far_mid_px - near_mid_px)).tolist()
        threshold_px = near_mid_px.tolist()

        # 3. Attitude-Aided 3D Ray-Ground Projection
        p_near_ac = self._project_pixel_to_ground(
            near_mid_px[0], near_mid_px[1], fx, fy, cx, cy, cam_pitch, cam_roll, onboard_agl_m
        )
        p_far_ac = self._project_pixel_to_ground(
            far_mid_px[0], far_mid_px[1], fx, fy, cx, cy, cam_pitch, cam_roll, onboard_agl_m
        )

        if p_near_ac is None or p_far_ac is None:
            return {"scenario_id": scenario_id, "status": "RAY_PROJECTION_FAILED", "vision_guidance_valid": False}

        vis_dist_to_threshold_m = float(np.linalg.norm(p_near_ac))
        rwy_vec = p_far_ac - p_near_ac
        rwy_vec_norm = max(float(np.linalg.norm(rwy_vec)), 1.0)
        u_rwy = rwy_vec / rwy_vec_norm
        n_rwy_right = np.array([-u_rwy[1], u_rwy[0]])

        # Relative yaw of aircraft nose wrt runway centerline (+ nose right of rwy, - nose left)
        rwy_bearing_in_ac_rad = math.atan2(u_rwy[1], u_rwy[0])
        rel_yaw_rad = -rwy_bearing_in_ac_rad
        rel_yaw_deg = math.degrees(rel_yaw_rad)

        vec_thresh_to_ac = -p_near_ac
        x_ac_rwy = float(np.dot(vec_thresh_to_ac, u_rwy))  # Negative before threshold
        vis_lateral_offset_m = float(np.dot(vec_thresh_to_ac, n_rwy_right))
        vis_dist_to_touchdown_m = abs(x_ac_rwy) + self.tdz_offset_m

        # 4. Determine Active Autoland Phase (APPR -> ALIGN -> FLARE -> ROLLOUT)
        phase_info = self._determine_autoland_phase(onboard_agl_m, wow_flag, hw_ils, hw_mls)
        active_mode = phase_info["active_mode"]

        # 5. Build Distortion-Free Segmented Ground Corridor
        appr_curve_px, align_flare_px, fpa_marker_px = self._build_segmented_ground_corridor_px(
            p_near_ac, p_far_ac, u_rwy, n_rwy_right, x_ac_rwy, vis_lateral_offset_m,
            onboard_agl_m, fx, fy, cx, cy, cam_pitch, cam_roll
        )

        # 6. Compute Phase-Specific Control Surface & Guidance Laws
        loc_error_deg = math.degrees(math.atan2(vis_lateral_offset_m, max(abs(x_ac_rwy), 1.0)))
        current_gs_angle_deg = math.degrees(math.atan2(onboard_agl_m, max(abs(x_ac_rwy), 1.0)))
        gs_error_deg = current_gs_angle_deg - self.target_gs_deg
        ideal_alt_m = abs(x_ac_rwy) * math.tan(math.radians(self.target_gs_deg))

        if active_mode == "APPR":
            # Localizer intercept law: command intercept track toward extended centerline
            intercept_track_deg = float(np.clip(-2.0 * loc_error_deg, -30.0, 30.0))
            hdg_cmd_deg = intercept_track_deg - rel_yaw_deg
            bank_cmd_deg = float(np.clip(1.6 * hdg_cmd_deg, -self.max_bank_deg, self.max_bank_deg))
            rudder_decrab_deg = 0.0
            pitch_cmd_deg = -current_gs_angle_deg
            target_vs_mps = -tas_mps * math.tan(math.radians(self.target_gs_deg))
            autobrake_cmd_pct = 0.0

        elif active_mode == "ALIGN":
            # Pitot-Static & Visual Decrab: kick out crab angle with rudder (Yaw -> 0) & hold centerline with wing-low aileron
            intercept_track_deg = float(np.clip(-1.2 * loc_error_deg, -10.0, 10.0))
            hdg_cmd_deg = -rel_yaw_deg  # Drive nose directly onto runway heading
            bank_cmd_deg = float(np.clip(1.2 * intercept_track_deg, -10.0, 10.0))
            rudder_decrab_deg = float(np.clip(-(rel_yaw_deg + beta_deg) * 1.5, -15.0, 15.0))
            pitch_cmd_deg = -self.target_gs_deg
            target_vs_mps = -ias_mps * math.tan(math.radians(self.target_gs_deg))
            autobrake_cmd_pct = 0.0

        elif active_mode == "FLARE":
            # Exponential sink-rate arrest from ~3.5 m/s (~700 fpm) to -0.75 m/s (~150 fpm) at touchdown
            tau_flare_s = 2.5
            target_vs_mps = -(onboard_agl_m / tau_flare_s) - 0.75
            pitch_cmd_deg = float(np.clip(2.5 + (15.24 - onboard_agl_m) * 0.18, 0.0, 5.5))
            hdg_cmd_deg = -rel_yaw_deg
            bank_cmd_deg = float(np.clip(-0.5 * vis_lateral_offset_m, -5.0, 5.0))
            rudder_decrab_deg = float(np.clip(-(rel_yaw_deg + beta_deg) * 1.8, -20.0, 20.0))
            autobrake_cmd_pct = 0.0  # Armed, waiting for 1-10ms WoW trigger

        else:  # ROLLOUT (WoW == True)
            target_vs_mps = 0.0
            pitch_cmd_deg = -1.0  # Lower nosewheel smoothly
            hdg_cmd_deg = -rel_yaw_deg
            bank_cmd_deg = 0.0
            rudder_decrab_deg = float(np.clip(-rel_yaw_deg * 2.0 - 0.4 * vis_lateral_offset_m, -25.0, 25.0))
            # Regulate autobrake proportional to kinetic energy & remaining runway distance
            stopping_dist_avail_m = max(L - self.tdz_offset_m, 500.0)
            req_decel_mps2 = (tas_mps**2) / (2.0 * stopping_dist_avail_m)
            autobrake_cmd_pct = round(float(np.clip((req_decel_mps2 / 4.0) * 100.0, 15.0, 100.0)), 1)

        # 7. Ground Truth Comparison
        gt = telemetry.get("ground_truth_relative", {})
        comparison = None
        if gt and gt.get("along_track_distance_m") is not None:
            gt_dist = float(gt["along_track_distance_m"])
            gt_lat = float(gt["lateral_offset_m"])
            comparison = {
                "gt_along_track_m": round(gt_dist, 3),
                "gt_lateral_offset_m": round(gt_lat, 3),
                "distance_error_m": round(vis_dist_to_threshold_m - gt_dist, 3),
                "distance_error_pct": round(abs(vis_dist_to_threshold_m - gt_dist) / max(gt_dist, 1.0) * 100.0, 2),
                "lateral_error_m": round(vis_lateral_offset_m - gt_lat, 3)
            }

        return {
            "scenario_id": scenario_id,
            "status": "LOCKED_AND_COMPUTED",
            "vision_guidance_valid": True,
            "autoland_phase_state": phase_info,
            "screen_targets_px": {
                "touchdown_point_px": [round(touchdown_px[0], 2), round(touchdown_px[1], 2)],
                "threshold_center_px": [round(threshold_px[0], 2), round(threshold_px[1], 2)],
                "flight_path_vector_px": fpa_marker_px,
                "appr_capture_corridor_px": appr_curve_px,
                "align_flare_centerline_px": align_flare_px
            },
            "visual_estimates": {
                "distance_to_threshold_m": round(vis_dist_to_threshold_m, 3),
                "distance_to_touchdown_m": round(vis_dist_to_touchdown_m, 3),
                "along_runway_axis_x_m": round(abs(x_ac_rwy), 3),
                "lateral_offset_m": round(vis_lateral_offset_m, 3),
                "relative_yaw_to_runway_deg": round(rel_yaw_deg, 4),
                "fused_agl_altitude_m": round(onboard_agl_m, 3),
                "fused_agl_altitude_ft": round(onboard_agl_m * M_TO_FT, 1)
            },
            "virtual_ils_deviations": {
                "localizer_error_deg": round(loc_error_deg, 4),
                "current_glideslope_deg": round(current_gs_angle_deg, 4),
                "glideslope_error_deg": round(gs_error_deg, 4),
                "ideal_altitude_on_3deg_gs_m": round(ideal_alt_m, 3)
            },
            "glide_path_vector": {
                "active_fma_mode": active_mode,
                "dx_along_runway_to_threshold_m": round(abs(x_ac_rwy), 3),
                "dx_along_runway_to_touchdown_m": round(vis_dist_to_touchdown_m, 3),
                "dy_lateral_offset_from_centerline_m": round(vis_lateral_offset_m, 3),
                "dz_descent_required_m": round(-onboard_agl_m, 3),
                "aircraft_relative_yaw_to_runway_deg": round(rel_yaw_deg, 4),
                "target_touchdown_relative_yaw_deg": 0.0,
                "commanded_pitch_deg": round(pitch_cmd_deg, 3),
                "commanded_target_sink_rate_mps": round(target_vs_mps, 3),
                "commanded_heading_change_deg": round(hdg_cmd_deg, 3),
                "commanded_roll_bank_deg": round(bank_cmd_deg, 3),
                "commanded_rudder_decrab_deg": round(rudder_decrab_deg, 3),
                "commanded_autobrake_pct": autobrake_cmd_pct
            },
            "ground_truth_comparison": comparison
        }