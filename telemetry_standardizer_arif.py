import os
import json
import math
from standardize_telemetry import (
    AIRPORT_DB,
    EARTH_RADIUS_M,
    identify_nearest_airport,
    compute_camera_intrinsics,
    wgs84_to_local_runway_frame
)

# Add Arif's default FlightGear airport (KSFO RWY 10L / 28R) to AIRPORT_DB if not present
AIRPORT_DB["KSFO"] = {
    "name": "San Francisco Intl Airport (RWY 10L)",
    "lat_deg": 37.628739,
    "lon_deg": -122.393392,
    "elevation_m": 4.0,
    "runway_width_m": 61.0,
    "runway_length_m": 3618.0,
    "rwy_heading_deg": 117.9
}


class ArifSimStandardizer:
    """
    Passive-Active Telemetry Adapter for Arif Emre Yaylakasi's 6-DOF Simulink Simulator (fsim.slx).
    Maps Table 4.1 workspace structures (out.states, out.AirData, out.ControlsDeg, out.gamma)
    from SI/Radians into 'canonical-autoland-telemetry/v1.0'.
    """
    def __init__(self, output_dir="runs/telemetry_std_arif", ref_lat_deg=37.7, ref_lon_deg=-122.4):
        self.output_dir = output_dir
        self.ref_lat_deg = ref_lat_deg
        self.ref_lon_deg = ref_lon_deg
        os.makedirs(self.output_dir, exist_ok=True)

    def flat_earth_to_wgs84(self, x_north_m, y_east_m, h_alt_m):
        """
        Replicates the 'Flat Earth to LLA' block in Arif's FlightGear subsystem (Fig 2.15).
        """
        dlat_deg = math.degrees(x_north_m / EARTH_RADIUS_M)
        dlon_deg = math.degrees(y_east_m / (EARTH_RADIUS_M * math.cos(math.radians(self.ref_lat_deg))))
        return self.ref_lat_deg + dlat_deg, self.ref_lon_deg + dlon_deg, abs(float(h_alt_m))

    def standardize_timestep(self, arif_step_dict, scenario_id="ArifB747_001"):
        """
        Accepts a dictionary extracted from a timestep of Arif's 'out' Simulink structure:
          - states: {u, v, w, p, q, r, phi, theta, psi, x, y, h} (SI & radians)
          - AirData: {V, alpha, beta, Mach, LiftForce} (SI & radians)
          - ControlsDeg: {Ail, El, Rud} (degrees)
          - gamma: flight path angle (radians)
        """
        states = arif_step_dict.get("states", arif_step_dict)
        air_data = arif_step_dict.get("AirData", {})
        controls_deg = arif_step_dict.get("ControlsDeg", {})

        # 1. Extract body velocities (u, v, w) in m/s
        u_mps = float(states.get("u", 75.0))
        v_mps = float(states.get("v", 0.0))
        w_mps = float(states.get("w", 0.0))

        # 2. Extract Euler angles (stored in radians in out.states -> convert to degrees)
        phi_rad = float(states.get("phi", 0.0))
        theta_rad = float(states.get("theta", 0.0))
        psi_rad = float(states.get("psi", 0.0))

        roll_deg = math.degrees(phi_rad)
        pitch_deg = math.degrees(theta_rad)
        heading_deg = math.degrees(psi_rad) % 360.0

        # 3. Extract Position (Flat-Earth x, y, h in meters or direct lat/lon)
        h_msl_m = abs(float(states.get("h", 200.0)))
        if "lat_deg" in states and "lon_deg" in states:
            ac_lat = float(states["lat_deg"])
            ac_lon = float(states["lon_deg"])
        else:
            x_north = float(states.get("x", 0.0))
            y_east = float(states.get("y", 0.0))
            ac_lat, ac_lon, h_msl_m = self.flat_earth_to_wgs84(x_north, y_east, h_msl_m)

        # 4. Extract Air Data (out.AirData: V, alpha, beta, Mach)
        tas_mps = float(air_data.get("V", math.sqrt(u_mps**2 + v_mps**2 + w_mps**2)))
        alpha_deg = math.degrees(float(air_data.get("alpha", math.atan2(w_mps, max(u_mps, 0.1)))))
        beta_deg = math.degrees(float(air_data.get("beta", 0.0)))

        # Compute ISA equivalent Indicated Airspeed (IAS) from TAS & altitude
        rho_ratio = math.exp(-h_msl_m / 9500.0)
        ias_mps = tas_mps * math.sqrt(rho_ratio)

        # Vertical speed h_dot (from Eq 262 in Appendix A)
        vs_mps = (
            u_mps * math.sin(theta_rad)
            - v_mps * math.sin(phi_rad) * math.cos(theta_rad)
            - w_mps * math.cos(phi_rad) * math.cos(theta_rad)
        )

        # 5. Match Nearest Airport & Compute Ground Truth Runway Frame
        airport_icao, airport_info = identify_nearest_airport(ac_lat, ac_lon)
        elev_m = airport_info.get("elevation_m", 4.0)
        agl_m = max(h_msl_m - elev_m, 0.0)

        # Virtual Weight-on-Wheels gate for Arif's airborne-only 6-DOF model (triggers at <= 0.5m AGL)
        wow_flag = bool(agl_m <= 0.5)

        img_w = int(arif_step_dict.get("img_width", 1920))
        img_h = int(arif_step_dict.get("img_height", 1080))
        v_fov = float(arif_step_dict.get("v_fov_deg", 28.555))

        gt_rel = wgs84_to_local_runway_frame(
            ac_lat, ac_lon, h_msl_m,
            airport_info.get("lat_deg", ac_lat),
            airport_info.get("lon_deg", ac_lon),
            elev_m,
            airport_info.get("rwy_heading_deg", heading_deg)
        )

        std_payload = {
            "schema_version": "canonical-autoland-telemetry/v1.0",
            "meta": {
                "source_type": "ARIF_B747_6DOF_SIMULINK",
                "scenario_id": scenario_id,
                "weather_condition": arif_step_dict.get("weather", "Dryden_Turbulence"),
                "timestamp_utc": arif_step_dict.get("timestamp_utc"),
                "image_path": arif_step_dict.get("image_path")
            },
            "aircraft_state": {
                "latitude_deg": round(ac_lat, 8),
                "longitude_deg": round(ac_lon, 8),
                "altitude_msl_m": round(h_msl_m, 3),
                "altitude_agl_m": round(agl_m, 3),
                "altitude_agl_ft": round(agl_m * 3.28084, 2),
                "pitch_deg": round(pitch_deg, 4),
                "roll_deg": round(roll_deg, 4),
                "heading_true_deg": round(heading_deg, 4),
                "tas_mps": round(tas_mps, 3),
                "ias_mps": round(ias_mps, 3),
                "ground_speed_mps": round(tas_mps, 3),
                "vertical_speed_mps": round(vs_mps, 3),
                "alpha_deg": round(alpha_deg, 4),
                "beta_deg": round(beta_deg, 4),
                "weight_on_wheels": wow_flag,
                "wow_poll_interval_ms": 2.0 if agl_m <= 15.24 else 10.0,
                "hardware_ils_available": False,
                "hardware_mls_available": False,
                "control_surfaces_deg": {
                    "aileron_deg": float(controls_deg.get("Ail", 0.0)),
                    "elevator_deg": float(controls_deg.get("El", 0.0)),
                    "rudder_deg": float(controls_deg.get("Rud", 0.0))
                }
            },
            "camera_state": {
                "latitude_deg": round(ac_lat, 8),
                "longitude_deg": round(ac_lon, 8),
                "altitude_msl_m": round(h_msl_m, 3),
                "pitch_deg": round(pitch_deg, 4),
                "raw_pitch_deg": round(pitch_deg, 4),
                "roll_deg": round(roll_deg, 4),
                "heading_true_deg": round(heading_deg, 4),
                "intrinsics": compute_camera_intrinsics(img_w, img_h, v_fov)
            },
            "environment": {
                "wind_speed_mps": float(arif_step_dict.get("wind_speed_mps", 0.0)),
                "visibility_m": 10000.0,
                "precip_rate_mm": 0.0,
                "in_cloud": False
            },
            "runway_reference": {
                "airport_icao": airport_icao,
                "runway_id": "10L",
                "threshold_elevation_m": elev_m,
                "runway_width_m": airport_info.get("runway_width_m", 61.0),
                "runway_length_m": airport_info.get("runway_length_m", 3618.0),
                "target_lat_deg": airport_info.get("lat_deg", ac_lat),
                "target_lon_deg": airport_info.get("lon_deg", ac_lon)
            },
            "ground_truth_relative": gt_rel
        }

        out_json = os.path.join(self.output_dir, f"{scenario_id}_telemetry_std.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(std_payload, f, indent=2)

        return out_json, std_payload