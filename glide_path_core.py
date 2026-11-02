import os
import json
import math
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

GRAVITY_MPS2 = 9.80665
M_TO_FT = 3.28084


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
        if forward_m <= 10.0:
            return None

        phi = math.radians(roll_deg)
        theta = math.radians(pitch_deg)

        z_cam = forward_m * math.cos(theta) - down_m * math.sin(theta)
        y_unrolled_cam = forward_m * math.sin(theta) + down_m * math.cos(theta)
        x_unrolled_cam = right_m

        if z_cam <= 10.0:
            return None

        x_norm_unrolled = x_unrolled_cam / z_cam
        y_norm_unrolled = y_unrolled_cam / z_cam

        x_norm = x_norm_unrolled * math.cos(phi) + y_norm_unrolled * math.sin(phi)
        y_norm = -x_norm_unrolled * math.sin(phi) + y_norm_unrolled * math.cos(phi)

        u_px = fx * x_norm + cx
        v_px = fy * y_norm + cy
        return [round(float(u_px), 2), round(float(v_px), 2)]

    def _determine_autoland_phase(self, agl_m, wow_flag, hw_ils, hw_mls):
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
                "wow_polling_rate_ms": 2.0,
                "phase_description": "Below 50 ft AGL; arresting sink rate exponentially & monitoring WoW (2 ms)."
            }
        elif agl_ft <= 155.0:
            return {
                "active_mode": "ALIGN",
                "guidance_source": guidance_source,
                "agl_ft": round(agl_ft, 1),
                "wow_active": False,
                "wow_polling_rate_ms": 10.0,
                "phase_description": "Below 155 ft AGL; decrabbing aircraft nose to runway axis (Yaw -> 0 deg) using Pitot/Air Data."
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
        Generates distortion-free 2D screen waypoints for any distance (from 5km down to 50m):
          - Dynamically computes where the bottom edge of the camera frustum hits the ground
            so the trajectory ALWAYS starts cleanly at the bottom of the screen.
        """
        dist_to_thresh = abs(x_ac_rwy)
        x_tdz = self.tdz_offset_m

        # Alignment gate scales smoothly with remaining distance (35% of distance to threshold, capped at 450m)
        gate_dist_before_thresh = min(self.align_gate_m, max(dist_to_thresh * 0.35, 60.0))
        x_align_gate = -gate_dist_before_thresh

        # Exact ground distance where the bottom of the 1080p screen (v ~ 1060 px) intersects the ground plane
        bottom_ray_depression_rad = math.atan2(cy * 0.96, fy) - math.radians(pitch_deg)
        min_ground_lead_m = max(agl_m / math.tan(max(bottom_ray_depression_rad, 0.08)), 25.0)

        # Start curve right at the bottom edge of the camera view
        x_vis_start = x_ac_rwy + min_ground_lead_m

        # Guarantee at least 250m of visible capture curve before the alignment gate
        if x_vis_start > x_align_gate - 150.0:
            x_align_gate = min(x_vis_start + max((x_tdz - x_vis_start) * 0.45, 80.0), 0.0)

        appr_curve_px = []
        align_flare_line_px = []

        # 1. APPR / Short-Final Capture Segment (x_vis_start -> x_align_gate)
        span = max(x_align_gate - x_ac_rwy, 50.0)
        for x in np.linspace(x_vis_start, x_align_gate, 30):
            s = np.clip((x - x_ac_rwy) / span, 0.0, 1.0)
            y = y_ac_rwy * 0.5 * (1.0 + math.cos(math.pi * s))

            pt_ac = p_near_ac + (x * u_rwy) + (y * n_rwy_right)
            px = self._project_ground_frame_to_pixel(
                pt_ac[0], pt_ac[1], agl_m, fx, fy, cx, cy, pitch_deg, roll_deg
            )
            if px is not None and -200 <= px[0] <= 2120 and 0 <= px[1] <= 1200:
                appr_curve_px.append(px)

        # 2. ALIGN & FLARE Stabilized Segment (x_align_gate -> Threshold -> TDZ)
        for x in np.linspace(x_align_gate, x_tdz, 20):
            pt_ac = p_near_ac + (x * u_rwy)
            px = self._project_ground_frame_to_pixel(
                pt_ac[0], pt_ac[1], agl_m, fx, fy, cx, cy, pitch_deg, roll_deg
            )
            if px is not None and -200 <= px[0] <= 2120 and 0 <= px[1] <= 1200:
                align_flare_line_px.append(px)

        fpa_pt_px = self._project_ground_frame_to_pixel(
             max(dist_to_thresh, 100.0), 0.0, agl_m, fx, fy, cx, cy, pitch_deg, roll_deg
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
        beta_deg = float(ac_state.get("beta_deg") or 0.0)
        wow_flag = bool(ac_state.get("weight_on_wheels", False))
        hw_ils = bool(ac_state.get("hardware_ils_available", False))
        hw_mls = bool(ac_state.get("hardware_mls_available", False))

        # 2. Runway Physical Dimensions & Detected Corners [NR, NL, FL, FR]
        rwy_ref = telemetry["runway_reference"]
        W = float(rwy_ref["runway_width_m"])
        L = float(rwy_ref["runway_length_m"])
        image_points_2d = np.array(coords["corners_ordered_list"], dtype=np.float64)

        near_mid_px = 0.5 * (image_points_2d[0] + image_points_2d[1])
        far_mid_px = 0.5 * (image_points_2d[2] + image_points_2d[3])

        w_near_px = max(float(np.linalg.norm(image_points_2d[0] - image_points_2d[1])), 1.0)
        w_far_px = max(float(np.linalg.norm(image_points_2d[3] - image_points_2d[2])), 1.0)
        lam = self.tdz_offset_m / max(L, 1.0)
        persp_t = (lam * w_near_px) / ((1.0 - lam) * w_far_px + lam * w_near_px)
        touchdown_px = (near_mid_px + persp_t * (far_mid_px - near_mid_px)).tolist()
        threshold_px = near_mid_px.tolist()

        # 3. Attitude-Aided 3D Ray-Ground Projection + Close-Range Width Fusion
        p_near_ac = self._project_pixel_to_ground(
            near_mid_px[0], near_mid_px[1], fx, fy, cx, cy, cam_pitch, cam_roll, onboard_agl_m
        )
        p_far_ac = self._project_pixel_to_ground(
            far_mid_px[0], far_mid_px[1], fx, fy, cx, cy, cam_pitch, cam_roll, onboard_agl_m
        )

        if p_near_ac is None or p_far_ac is None:
            return {"scenario_id": scenario_id, "status": "RAY_PROJECTION_FAILED", "vision_guidance_valid": False}

        ray_dist_m = float(np.linalg.norm(p_near_ac))
        # Optical width-based range: D_width = (W * fx) / w_near_px
        width_dist_m = (W * fx) / w_near_px

        # Below 150 m AGL (short final), blend optical threshold width with ray-ground range
        # to immunize against shallow grazing-angle keypoint jitter at night
        if onboard_agl_m < 80.0 and w_near_px > 65.0:
            blend_w = np.clip((w_near_px - 65.0) / 100.0, 0.0, 0.65)
            vis_dist_to_threshold_m = (1.0 - blend_w) * ray_dist_m + blend_w * width_dist_m
            p_near_ac = p_near_ac * (vis_dist_to_threshold_m / max(ray_dist_m, 1.0))
        else:
            vis_dist_to_threshold_m = ray_dist_m

        rwy_vec = p_far_ac - p_near_ac
        rwy_vec_norm = max(float(np.linalg.norm(rwy_vec)), 1.0)
        u_rwy = rwy_vec / rwy_vec_norm
        n_rwy_right = np.array([-u_rwy[1], u_rwy[0]])

        rwy_bearing_in_ac_rad = math.atan2(u_rwy[1], u_rwy[0])
        rel_yaw_rad = -rwy_bearing_in_ac_rad
        rel_yaw_deg = math.degrees(rel_yaw_rad)

        vec_thresh_to_ac = -p_near_ac
        x_ac_rwy = float(np.dot(vec_thresh_to_ac, u_rwy))
        vis_lateral_offset_m = float(np.dot(vec_thresh_to_ac, n_rwy_right))
        vis_dist_to_touchdown_m = abs(x_ac_rwy) + self.tdz_offset_m

        # 4. Determine Active Autoland Phase
        phase_info = self._determine_autoland_phase(onboard_agl_m, wow_flag, hw_ils, hw_mls)
        active_mode = phase_info["active_mode"]

        # 5. Build Segmented Ground Corridor (Always Visible Down to Bottom of Screen)
        appr_curve_px, align_flare_px, fpa_marker_px = self._build_segmented_ground_corridor_px(
            p_near_ac, p_far_ac, u_rwy, n_rwy_right, x_ac_rwy, vis_lateral_offset_m,
            onboard_agl_m, fx, fy, cx, cy, cam_pitch, cam_roll
        )

        # 6. Compute Phase-Specific Control Surface Commands
        loc_error_deg = math.degrees(math.atan2(vis_lateral_offset_m, max(abs(x_ac_rwy), 1.0)))
        current_gs_angle_deg = math.degrees(math.atan2(onboard_agl_m, max(abs(x_ac_rwy), 1.0)))
        gs_error_deg = current_gs_angle_deg - self.target_gs_deg
        ideal_alt_m = abs(x_ac_rwy) * math.tan(math.radians(self.target_gs_deg))

        if active_mode == "APPR":
            intercept_track_deg = float(np.clip(-2.0 * loc_error_deg, -30.0, 30.0))
            hdg_cmd_deg = intercept_track_deg - rel_yaw_deg
            bank_cmd_deg = float(np.clip(1.6 * hdg_cmd_deg, -self.max_bank_deg, self.max_bank_deg))
            rudder_decrab_deg = 0.0
            pitch_cmd_deg = -current_gs_angle_deg
            target_vs_mps = -tas_mps * math.tan(math.radians(self.target_gs_deg))
            autobrake_cmd_pct = 0.0
        elif active_mode == "ALIGN":
            intercept_track_deg = float(np.clip(-1.2 * loc_error_deg, -10.0, 10.0))
            hdg_cmd_deg = -rel_yaw_deg
            bank_cmd_deg = float(np.clip(1.2 * intercept_track_deg, -10.0, 10.0))
            rudder_decrab_deg = float(np.clip(-(rel_yaw_deg + beta_deg) * 1.5, -15.0, 15.0))
            pitch_cmd_deg = -self.target_gs_deg
            target_vs_mps = -ias_mps * math.tan(math.radians(self.target_gs_deg))
            autobrake_cmd_pct = 0.0
        elif active_mode == "FLARE":
            tau_flare_s = 2.5
            target_vs_mps = -(onboard_agl_m / tau_flare_s) - 0.75
            pitch_cmd_deg = float(np.clip(2.5 + (15.24 - onboard_agl_m) * 0.18, 0.0, 5.5))
            hdg_cmd_deg = -rel_yaw_deg
            bank_cmd_deg = float(np.clip(-0.5 * vis_lateral_offset_m, -5.0, 5.0))
            rudder_decrab_deg = float(np.clip(-(rel_yaw_deg + beta_deg) * 1.8, -20.0, 20.0))
            autobrake_cmd_pct = 0.0
        else:
            target_vs_mps = 0.0
            pitch_cmd_deg = -1.0
            hdg_cmd_deg = -rel_yaw_deg
            bank_cmd_deg = 0.0
            rudder_decrab_deg = float(np.clip(-rel_yaw_deg * 2.0 - 0.4 * vis_lateral_offset_m, -25.0, 25.0))
            stopping_dist_avail_m = max(L - self.tdz_offset_m, 500.0)
            req_decel_mps2 = (tas_mps**2) / (2.0 * stopping_dist_avail_m)
            autobrake_cmd_pct = round(float(np.clip((req_decel_mps2 / 4.0) * 100.0, 15.0, 100.0)), 1)

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