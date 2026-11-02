import os
import glob
import json
import math
import csv
from pathlib import Path

AIRPORT_DB = {
    "EDDS": {"name": "Stuttgart Airport", "lat_deg": 48.695627, "lon_deg": 9.245848, "elevation_m": 389.0, "runway_width_m": 45.0, "runway_length_m": 3345.0, "rwy_heading_deg": 252.10},
    "EDDM": {"name": "Munich Airport", "lat_deg": 48.353783, "lon_deg": 11.786086, "elevation_m": 453.2, "runway_width_m": 60.0, "runway_length_m": 4000.0, "rwy_heading_deg": 263.50},
    "LFQQ": {"name": "Lille-Lesquin Airport", "lat_deg": 50.563332, "lon_deg": 3.086886, "elevation_m": 47.9, "runway_width_m": 45.0, "runway_length_m": 2825.0, "rwy_heading_deg": 257.00},
    "LIRF": {"name": "Rome Fiumicino Airport", "lat_deg": 41.800278, "lon_deg": 12.238889, "elevation_m": 4.6, "runway_width_m": 60.0, "runway_length_m": 3900.0, "rwy_heading_deg": 161.00},
    "LSZH": {"name": "Zurich Airport", "lat_deg": 47.458056, "lon_deg": 8.548056, "elevation_m": 431.6, "runway_width_m": 60.0, "runway_length_m": 3700.0, "rwy_heading_deg": 155.00}
}

EARTH_RADIUS_M = 6378137.0
_MANIFEST_CACHE = {}


def haversine_distance_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def identify_nearest_airport(lat, lon):
    best_icao = "UNKNOWN"
    min_dist = float("inf")
    for icao, info in AIRPORT_DB.items():
        d = haversine_distance_m(lat, lon, info["lat_deg"], info["lon_deg"])
        if d < min_dist:
            min_dist = d
            best_icao = icao
    return best_icao, AIRPORT_DB.get(best_icao, {})


def load_manifest_for_usable_dir(usable_dir: Path):
    """Loads and caches manifest.csv from a usable directory for exact Ground Truth lookup."""
    key = str(usable_dir.resolve())
    if key not in _MANIFEST_CACHE:
        manifest_path = usable_dir / "manifest.csv"
        lookup = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lookup[row["scenario_id"]] = row
        _MANIFEST_CACHE[key] = lookup
    return _MANIFEST_CACHE[key]


def compute_camera_intrinsics(width_px, height_px, v_fov_deg):
    v_fov_rad = math.radians(v_fov_deg)
    fy = (height_px / 2.0) / math.tan(v_fov_rad / 2.0)
    fx = fy
    cx = width_px / 2.0
    cy = height_px / 2.0
    h_fov_deg = math.degrees(2.0 * math.atan((width_px / 2.0) / fx))
    return {
        "width_px": width_px,
        "height_px": height_px,
        "fov_vertical_deg": round(v_fov_deg, 5),
        "fov_horizontal_deg": round(h_fov_deg, 5),
        "fx_px": round(fx, 4),
        "fy_px": round(fy, 4),
        "cx_px": round(cx, 2),
        "cy_px": round(cy, 2)
    }


def wgs84_to_local_runway_frame(ac_lat, ac_lon, ac_alt_m, ref_lat, ref_lon, ref_alt_m, rwy_heading_deg):
    d_north_m = math.radians(ac_lat - ref_lat) * EARTH_RADIUS_M
    d_east_m = math.radians(ac_lon - ref_lon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))

    hdg_rad = math.radians(rwy_heading_deg)
    ux_east = math.sin(hdg_rad)
    ux_north = math.cos(hdg_rad)

    uy_east = math.cos(hdg_rad)
    uy_north = -math.sin(hdg_rad)

    along_track_m = -(d_east_m * ux_east + d_north_m * ux_north)
    lateral_offset_m = d_east_m * uy_east + d_north_m * uy_north
    relative_alt_m = ac_alt_m - ref_alt_m

    return {
        "along_track_distance_m": round(along_track_m, 3),
        "lateral_offset_m": round(lateral_offset_m, 3),
        "relative_altitude_m": round(relative_alt_m, 3)
    }


def parse_msfs_v3_json(json_path, weather_folder_name):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    scenario_id = raw.get("scenario_id", Path(json_path).stem)
    usable_dir = Path(json_path).parent.parent
    manifest_lookup = load_manifest_for_usable_dir(usable_dir)
    m_row = manifest_lookup.get(scenario_id, {})

    b_ac = raw["before"]["aircraft"]
    a_ac = raw["after"]["aircraft"]
    b_cam = raw["before"]["camera_world"]
    a_cam = raw["after"]["camera_world"]

    ac_lat = 0.5 * (b_ac["latitude_deg"] + a_ac["latitude_deg"])
    ac_lon = 0.5 * (b_ac["longitude_deg"] + a_ac["longitude_deg"])
    ac_alt_m = 0.5 * (b_ac["altitude_m"] + a_ac["altitude_m"])
    ac_pitch = 0.5 * (b_ac["pitch_deg"] + a_ac["pitch_deg"])
    ac_roll = 0.5 * (b_ac["bank_deg"] + a_ac["bank_deg"])
    ac_hdg = 0.5 * (b_ac["heading_true_deg"] + a_ac["heading_true_deg"])

    cam_lat = 0.5 * (b_cam["latitude_deg"] + a_cam["latitude_deg"])
    cam_lon = 0.5 * (b_cam["longitude_deg"] + a_cam["longitude_deg"])
    cam_alt_m = 0.5 * (b_cam["altitude_m"] + a_cam["altitude_m"])
    cam_pitch = 0.5 * (b_cam["pitch_deg"] + a_cam["pitch_deg"])
    cam_roll = 0.5 * (b_cam["bank_deg"] + a_cam["bank_deg"])
    cam_hdg = 0.5 * (b_cam["heading_true_deg"] + a_cam["heading_true_deg"])

    # Also capture raw native camera pitch/bank if available
    raw_pitch = 0.5 * (
        b_cam.get("raw_pbh_deg", {}).get("pitch", cam_pitch) +
        a_cam.get("raw_pbh_deg", {}).get("pitch", cam_pitch)
    )

    img_w, img_h = raw.get("image_size", [1920, 1080])
    v_fov_deg = b_cam.get("fov_deg", 28.555)

    airport_icao, airport_info = identify_nearest_airport(ac_lat, ac_lon)
    if m_row.get("icao"):
        airport_icao = m_row["icao"]
        airport_info = AIRPORT_DB.get(airport_icao, airport_info)

    rwy_width_m = float(m_row.get("runway_width_m", airport_info.get("runway_width_m", 45.0)))
    rwy_length_m = float(m_row.get("runway_length_m", airport_info.get("runway_length_m", 3345.0)))

    # Exact Ground Truth from manifest.csv if available
    if "ground_dist_m" in m_row and "alt_diff_m" in m_row:
        gt_dist_m = float(m_row["ground_dist_m"])
        gt_alt_diff_m = float(m_row["alt_diff_m"])
        gt_glide_deg = float(m_row.get("glide_angle_deg", math.degrees(math.atan2(gt_alt_diff_m, gt_dist_m))))
        elev_m = round(ac_alt_m - gt_alt_diff_m, 3)
        # Compute cross-track lateral offset using airport runway true heading
        rwy_hdg = airport_info.get("rwy_heading_deg", cam_hdg)
        fallback_gt = wgs84_to_local_runway_frame(
            cam_lat, cam_lon, cam_alt_m,
            airport_info.get("lat_deg", ac_lat),
            airport_info.get("lon_deg", ac_lon),
            elev_m, rwy_hdg
        )
        gt_relative = {
            "along_track_distance_m": round(gt_dist_m, 3),
            "lateral_offset_m": fallback_gt["lateral_offset_m"],
            "relative_altitude_m": round(gt_alt_diff_m, 3),
            "glide_angle_deg": round(gt_glide_deg, 4)
        }
    else:
        elev_m = airport_info.get("elevation_m", 0.0)
        gt_relative = wgs84_to_local_runway_frame(
            cam_lat, cam_lon, cam_alt_m,
            airport_info.get("lat_deg", ac_lat),
            airport_info.get("lon_deg", ac_lon),
            elev_m, airport_info.get("rwy_heading_deg", cam_hdg)
        )

    img_rel_path = raw.get("image_file", f"images/{scenario_id}.png")
    full_img_path = usable_dir / img_rel_path
    pose_label_path = usable_dir / "labels_pose" / f"{scenario_id}.txt"

    return {
        "schema_version": "canonical-autoland-telemetry/v1.0",
        "meta": {
            "source_type": "MSFS2024_TAYLAN_V3",
            "scenario_id": scenario_id,
            "weather_condition": weather_folder_name,
            "timestamp_utc": raw["before"].get("native_utc"),
            "image_path": str(full_img_path),
            "pose_label_path": str(pose_label_path) if pose_label_path.exists() else None,
            "image_sha256": raw.get("image_sha256")
        },
        "aircraft_state": {
            "latitude_deg": round(ac_lat, 8),
            "longitude_deg": round(ac_lon, 8),
            "altitude_msl_m": round(ac_alt_m, 3),
            "altitude_agl_m": round(ac_alt_m - elev_m, 3),
            "altitude_agl_ft": round((ac_alt_m - elev_m) * 3.28084, 2),
            "pitch_deg": round(ac_pitch, 4),
            "roll_deg": round(ac_roll, 4),
            "heading_true_deg": round(ac_hdg, 4),
            # Pitot-Static & Air Data Computer (ADC) inputs for ALIGN & FLARE
            "tas_mps": raw.get("tas_mps", None),
            "ias_mps": raw.get("ias_mps", None),
            "ground_speed_mps": raw.get("ground_speed_mps", None),
            "vertical_speed_mps": raw.get("vertical_speed_mps", None),
            "alpha_deg": raw.get("alpha_deg", None),
            "beta_deg": raw.get("beta_deg", None),
            # Landing Gear Weight-on-Wheels (WoW) & Hardware ILS Priority flags
            "weight_on_wheels": bool(raw.get("sim_on_ground", False)),
            "wow_poll_interval_ms": 5.0,
            "hardware_ils_available": bool(raw.get("nav_has_localizer", False)),
            "hardware_mls_available": bool(raw.get("nav_has_mls", False))
        },
        "camera_state": {
            "latitude_deg": round(cam_lat, 8),
            "longitude_deg": round(cam_lon, 8),
            "altitude_msl_m": round(cam_alt_m, 3),
            "pitch_deg": round(cam_pitch, 4),
            "raw_pitch_deg": round(raw_pitch, 4),
            "roll_deg": round(cam_roll, 4),
            "heading_true_deg": round(cam_hdg, 4),
            "intrinsics": compute_camera_intrinsics(img_w, img_h, v_fov_deg)
        },
        "environment": {
            "wind_speed_mps": round(raw.get("ambient_wind_velocity_mps", 0.0), 3),
            "visibility_m": round(raw.get("ambient_visibility_m", 10000.0), 1),
            "precip_rate_mm": round(raw.get("ambient_precip_rate_mm_water", 0.0), 3),
            "in_cloud": bool(raw.get("ambient_in_cloud_bool", False))
        },
        "runway_reference": {
            "airport_icao": airport_icao,
            "runway_id": m_row.get("runway", "25"),
            "threshold_elevation_m": elev_m,
            "runway_width_m": rwy_width_m,
            "runway_length_m": rwy_length_m,
            "target_lat_deg": airport_info.get("lat_deg", ac_lat),
            "target_lon_deg": airport_info.get("lon_deg", ac_lon)
        },
        "ground_truth_relative": gt_relative
    }