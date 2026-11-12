import os
import json
import math
from datetime import datetime, timezone
import numpy as np
import scipy.io as sio

from standardize_telemetry import (
    AIRPORT_DB,
    EARTH_RADIUS_M,
    identify_nearest_airport,
    compute_camera_intrinsics,
    wgs84_to_local_runway_frame
)

# Ensure Arif's default FlightGear airport (KSFO RWY 10L) is registered in AIRPORT_DB
if "KSFO" not in AIRPORT_DB:
    # Exact KSFO Runway 10L Threshold from Arif's InitialB747_400.m (Scenario == 3)
    AIRPORT_DB["KSFO"] = {
        "name": "San Francisco Intl Airport (RWY 10L)",
        "lat_deg": 37.0 + (37.0 / 60.0) + (42.7 / 3600.0),  # 37.62852778
        "lon_deg": -(122.0 + (23.0 / 60.0) + (34.4 / 3600.0)),  # -122.39288889
        "elevation_m": 16.0 / 3.28,  # 4.878 m
        "runway_width_m": 61.0,
        "runway_length_m": 3618.0,
        "rwy_heading_deg": 118.0
    }


class MatlabArrayStandardizer:
    def __init__(
        self,
        output_dir="runs/live_arif",
        ref_airport_icao="KSFO",
        # Exact Flat-Earth (X=0, Y=0) Reference from InitialB747_400.m (Scenario == 2)
        ref_lat_deg=37.0 + (40.0 / 60.0) + (17.1 / 3600.0),      # 37.67141667
        ref_lon_deg=-(122.0 + (29.0 / 60.0) + (41.7 / 3600.0)),  # -122.49491667
        img_width=1920,
        img_height=1080,
        v_fov_deg=28.555
    ):
        self.output_dir = output_dir
        self.ref_airport_icao = ref_airport_icao
        self.ref_lat_deg = ref_lat_deg
        self.ref_lon_deg = ref_lon_deg
        self.img_width = img_width
        self.img_height = img_height
        self.v_fov_deg = v_fov_deg
        os.makedirs(self.output_dir, exist_ok=True)

    def flat_earth_to_wgs84(self, x_north_m, y_east_m, h_alt_m):
        """Converts Arif's Flat-Earth (x, y, h) into WGS84 (lat, lon, alt_msl)."""
        dlat_deg = math.degrees(x_north_m / EARTH_RADIUS_M)
        dlon_deg = math.degrees(
            y_east_m / (EARTH_RADIUS_M * math.cos(math.radians(self.ref_lat_deg)))
        )
        return self.ref_lat_deg + dlat_deg, self.ref_lon_deg + dlon_deg, abs(float(h_alt_m))

    def pull_and_standardize_from_mat(self, mat_file_path, scenario_id="ArifLive_001"):
        """
        Reads Arif's exported MATLAB array file (containing 'sim_array' or 'x', 'AirData', 'Controls')
        and writes the standardized telemetry JSON.
        """
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"MATLAB telemetry file not found: {mat_file_path}")

        mat_data = sio.loadmat(mat_file_path, squeeze_me=True)
        return self.standardize_from_arrays(mat_data, scenario_id=scenario_id)

    def standardize_from_arrays(self, data_dict, scenario_id="ArifLive_001"):
        """
        Accepts either:
          - A single combined 1D numpy/MATLAB array 'sim_array' of length >= 17:
            [u, v, w, p, q, r, phi, theta, psi, x, y, h, Va, alpha, beta, Mach, LiftForce, ...]
          - Or separate arrays 'x' (12,), 'AirData' (5,), 'Controls' (3,), 'cockpit_state'
        """
        if "sim_array" in data_dict:
            arr = np.asarray(data_dict["sim_array"], dtype=np.float64).flatten()
            x_vec = arr[0:12]
            air_vec = arr[12:17] if len(arr) >= 17 else np.array([75.0, 0.04, 0.0, 0.22, 2.8e6])
            ctrl_vec = arr[17:20] if len(arr) >= 20 else np.zeros(3)
        else:
            x_vec = np.asarray(data_dict.get("x", np.zeros(12)), dtype=np.float64).flatten()
            air_vec = np.asarray(data_dict.get("AirData", [75.0, 0.04, 0.0, 0.22, 2.8e6]), dtype=np.float64).flatten()
            ctrl_vec = np.asarray(data_dict.get("Controls", np.zeros(3)), dtype=np.float64).flatten()

        # 1. Unpack x(1:12): [u, v, w, p, q, r, phi, theta, psi, X_north, Y_east, h]
        u_mps, v_mps, w_mps = float(x_vec[0]), float(x_vec[1]), float(x_vec[2])
        p_rps, q_rps, r_rps = float(x_vec[3]), float(x_vec[4]), float(x_vec[5])
        phi_rad, theta_rad, psi_rad = float(x_vec[6]), float(x_vec[7]), float(x_vec[8])
        x_north_m, y_east_m, h_raw_m = float(x_vec[9]), float(x_vec[10]), float(x_vec[11])

        roll_deg = math.degrees(phi_rad)
        pitch_deg = math.degrees(theta_rad)
        heading_deg = math.degrees(psi_rad) % 360.0

        # 2. Unpack AirData(1:5): [Va, alpha, beta, Mach, LiftForce]
        tas_mps = float(air_vec[0]) if len(air_vec) > 0 else math.sqrt(u_mps**2 + v_mps**2 + w_mps**2)
        alpha_deg = math.degrees(float(air_vec[1])) if len(air_vec) > 1 else 0.0
        beta_deg = math.degrees(float(air_vec[2])) if len(air_vec) > 2 else 0.0
        mach_num = float(air_vec[3]) if len(air_vec) > 3 else tas_mps / 340.3

        # 3. Position & Altitude AGL
        ac_lat, ac_lon, alt_msl_m = self.flat_earth_to_wgs84(x_north_m, y_east_m, h_raw_m)
        airport_icao, airport_info = identify_nearest_airport(ac_lat, ac_lon)
        if self.ref_airport_icao in AIRPORT_DB:
            airport_icao = self.ref_airport_icao
            airport_info = AIRPORT_DB[airport_icao]

        elev_m = float(airport_info.get("elevation_m", 4.0))
        agl_m = max(alt_msl_m - elev_m, 0.0)

        # Vertical speed (h_dot in m/s, positive climb / negative descent)
        vs_mps = (
            u_mps * math.sin(theta_rad)
            - v_mps * math.sin(phi_rad) * math.cos(theta_rad)
            - w_mps * math.cos(phi_rad) * math.cos(theta_rad)
        )

        # ISA equivalent IAS from TAS
        rho_ratio = math.exp(-alt_msl_m / 9500.0)
        ias_mps = tas_mps * math.sqrt(rho_ratio)

        # Weight-on-Wheels (WoW) trigger at <= 0.5m AGL (or explicit flag from MATLAB)
        wow_flag = bool(data_dict.get("wow_flag", agl_m <= 0.5))

        gt_rel = wgs84_to_local_runway_frame(
            ac_lat, ac_lon, alt_msl_m,
            airport_info["lat_deg"], airport_info["lon_deg"],
            elev_m, airport_info["rwy_heading_deg"]
        )

        std_payload = {
            "schema_version": "canonical-autoland-telemetry/v1.0",
            "meta": {
                "source_type": "ARIF_MATLAB_ARRAY_BRIDGE",
                "scenario_id": scenario_id,
                "weather_condition": "FlightGear_Live",
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            },
            "aircraft_state": {
                "latitude_deg": round(ac_lat, 8),
                "longitude_deg": round(ac_lon, 8),
                "altitude_msl_m": round(alt_msl_m, 3),
                "altitude_agl_m": round(agl_m, 3),
                "altitude_agl_ft": round(agl_m * 3.28084, 2),
                "pitch_deg": round(pitch_deg, 4),
                "roll_deg": round(roll_deg, 4),
                "heading_true_deg": round(heading_deg, 4),
                "p_roll_rate_dps": round(math.degrees(p_rps), 4),
                "q_pitch_rate_dps": round(math.degrees(q_rps), 4),
                "r_yaw_rate_dps": round(math.degrees(r_rps), 4),
                "tas_mps": round(tas_mps, 3),
                "ias_mps": round(ias_mps, 3),
                "ground_speed_mps": round(math.hypot(u_mps, v_mps), 3),
                "vertical_speed_mps": round(vs_mps, 3),
                "alpha_deg": round(alpha_deg, 4),
                "beta_deg": round(beta_deg, 4),
                "mach": round(mach_num, 4),
                "weight_on_wheels": wow_flag,
                "wow_poll_interval_ms": 2.0 if agl_m <= 15.24 else 10.0,
                "hardware_ils_available": False,
                "hardware_mls_available": False,
                "current_surfaces_deg": {
                    "elevator_deg": round(math.degrees(float(ctrl_vec[0])), 3),
                    "aileron_deg": round(math.degrees(float(ctrl_vec[1])), 3),
                    "rudder_deg": round(math.degrees(float(ctrl_vec[2])), 3)
                }
            },
            "camera_state": {
                "latitude_deg": round(ac_lat, 8),
                "longitude_deg": round(ac_lon, 8),
                "altitude_msl_m": round(alt_msl_m, 3),
                "pitch_deg": round(pitch_deg, 4),
                "raw_pitch_deg": round(pitch_deg, 4),
                "roll_deg": round(roll_deg, 4),
                "heading_true_deg": round(heading_deg, 4),
                "intrinsics": compute_camera_intrinsics(self.img_width, self.img_height, self.v_fov_deg)
            },
            "environment": {
                "wind_speed_mps": 0.0,
                "visibility_m": 10000.0,
                "precip_rate_mm": 0.0,
                "in_cloud": False
            },
            "runway_reference": {
                "airport_icao": airport_icao,
                "runway_id": "10L",
                "threshold_elevation_m": elev_m,
                "runway_width_m": airport_info["runway_width_m"],
                "runway_length_m": airport_info["runway_length_m"],
                "target_lat_deg": airport_info["lat_deg"],
                "target_lon_deg": airport_info["lon_deg"]
            },
            "ground_truth_relative": gt_rel
        }

        out_json_path = os.path.join(self.output_dir, "latest_telemetry_std.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(std_payload, f, indent=2)

        return out_json_path, std_payload