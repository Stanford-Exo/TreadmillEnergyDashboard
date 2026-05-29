# File: src/server/server.py

import os
import sys
import glob
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import pandas as pd

# Setup paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../../src")))

from online_analyze.energy_analyzer import EnergyAnalyzer

# Point to exported validation parquet folder
COM_VAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_csvs/com_validation"))

class NumpyEncoder(json.JSONEncoder):
    """Encodes NumPy arrays and numeric objects into JSON-compatible values."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

class DashboardServerHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/files":
            self.handle_get_files()
        elif path == "/api/load":
            self.handle_load_file(query)
        else:
            self.send_error(404, "Endpoint not found")

    def handle_get_files(self):
        """Discovers all available validation parquet files."""
        if not os.path.exists(COM_VAL_DIR):
            files = []
        else:
            pattern = os.path.join(COM_VAL_DIR, "*.parquet")
            files = [os.path.basename(f) for f in glob.glob(pattern)]
            # Filter out static/calibration trials
            files = sorted([f for f in files if "Santos" not in f])

        response_bytes = json.dumps({"files": files}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(response_bytes)

    def handle_load_file(self, query):
        """
        Loads a selected parquet file, processes frames sequentially through 
        the EnergyAnalyzer pipeline, and returns the analyzed time-series states.
        """
        file_list = query.get("file")
        if not file_list:
            self.send_error(400, "Missing 'file' query parameter")
            return

        filename = file_list[0]
        file_path = os.path.join(COM_VAL_DIR, filename)

        if not os.path.exists(file_path):
            self.send_error(404, "Requested trial file does not exist")
            return

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            self.send_error(500, f"Error opening parquet trial: {str(e)}")
            return

        # Dynamically map left and right contact bodies
        force_cols = [col for col in df.columns if col.endswith("_force_y")]
        contact_bodies = [col.replace("_force_y", "") for col in force_cols]

        left_body, right_body = None, None
        for cb in contact_bodies:
            if cb.endswith("_l") or "left" in cb.lower():
                left_body = cb
            elif cb.endswith("_r") or "right" in cb.lower():
                right_body = cb

        # Dynamic fallback assignments if strict conventions are absent
        if left_body is None and contact_bodies:
            left_body = contact_bodies[0]
        if right_body is None and len(contact_bodies) > 1:
            right_body = contact_bodies[1]

        if not left_body or not right_body:
            self.send_error(400, "Unable to resolve bilateral contact body pairings")
            return

        times = df["time"].values
        dts = np.diff(times)
        default_dt = np.median(dts) if len(dts) > 0 else 0.01

        # Pre-calculate estimated subject mass
        f_total_y = df[f"{left_body}_force_y"].values + df[f"{right_body}_force_y"].values
        active_fy = f_total_y[f_total_y > 50.0]
        calculated_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

        # Initialize the shared streaming analyzer
        analyzer = EnergyAnalyzer(initial_mass=calculated_mass)
        processed_frames = []

        # Process frames sequentially to replicate high-frequency live ingestion
        for i in range(len(df)):
            forces = {
                'left': np.array([
                    df[f"{left_body}_force_x"].values[i],
                    df[f"{left_body}_force_y"].values[i],
                    df[f"{left_body}_force_z"].values[i]
                ]),
                'right': np.array([
                    df[f"{right_body}_force_x"].values[i],
                    df[f"{right_body}_force_y"].values[i],
                    df[f"{right_body}_force_z"].values[i]
                ])
            }
            cops = {
                'left': np.array([
                    df[f"{left_body}_cop_x"].values[i],
                    df[f"{left_body}_cop_y"].values[i],
                    df[f"{left_body}_cop_z"].values[i]
                ]),
                'right': np.array([
                    df[f"{right_body}_cop_x"].values[i],
                    df[f"{right_body}_cop_y"].values[i],
                    df[f"{right_body}_cop_z"].values[i]
                ])
            }
            dt = dts[i] if i < len(dts) else default_dt
            
            # Execute Kalman Filter tracking and gait cycle segmentation
            res = analyzer.update(times[i], forces, cops, dt)
            
            # Estimate current stride cycle progress percentage (0 - 100%)
            phase = 0.0
            if analyzer.first_l_strike_seen and analyzer.gait_cycle_buffer['time']:
                start_time = analyzer.gait_cycle_buffer['time'][0]
                elapsed = times[i] - start_time
                summary = analyzer.stride_analyzer.get_metrics_summary()
                mean_dur = summary.get('stride_duration_mean', 1.0)
                if mean_dur > 0:
                    phase = min(max((elapsed / mean_dur) * 100.0, 0.0), 100.0)

            frame_info = {
                "time": times[i],
                "com_pos": res['com_pos'],
                "com_vel": res['com_vel'],
                "power_left": res['power_left'],
                "power_right": res['power_right'],
                "power_total": res['power_total'],
                "mass": res['mass'],
                "tilt": res['tilt'],
                "left_force": forces['left'],
                "right_force": forces['right'],
                "left_cop": cops['left'],
                "right_cop": cops['right'],
                "left_active": bool(analyzer.stride_analyzer.contact_states['left']),
                "right_active": bool(analyzer.stride_analyzer.contact_states['right']),
                "phase": phase
            }
            processed_frames.append(frame_info)

        # Retrieve cumulative stride averages and temporal/spatial metrics
        profiles = analyzer.get_aggregate_profiles()
        metrics = analyzer.stride_analyzer.get_metrics_summary()

        response_payload = {
            "metadata": {
                "filename": filename,
                "left_body": left_body,
                "right_body": right_body,
                "frames_count": len(processed_frames),
                "calculated_mass": calculated_mass,
                "metrics": metrics
            },
            "frames": processed_frames,
            "profiles": profiles
        }

        response_bytes = json.dumps(response_payload, cls=NumpyEncoder).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(response_bytes)

def run(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardServerHandler)
    print(f"✅ Dashboard Server running at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Dashboard Server...")
        httpd.server_close()

if __name__ == "__main__":
    run()