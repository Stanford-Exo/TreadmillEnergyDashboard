# File: src/validation/validate_kf.py

import os
import glob
import numpy as np
import pandas as pd
from com_kf import ComKalmanFilter

# Setup paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COM_VAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_csvs/com_validation"))


def find_validation_files(directory):
    """Finds all parquet validation files in the target folder."""
    pattern = os.path.join(directory, "*_com_validation.parquet")
    files = glob.glob(pattern)
    return files


def run_validation_on_file(file_path):
    """Runs the Kalman filter on a single trial and compares velocity against ground truth."""
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    
    # Load dataset
    df = pd.read_parquet(file_path)
    if df.empty:
        print("  Error: Empty dataframe.")
        return None

    # Identify unique contact body prefixes (e.g., 'calcn_l', 'calcn_r')
    force_cols = [col for col in df.columns if col.endswith("_force_x")]
    contact_bodies = [col.replace("_force_x", "") for col in force_cols]
    
    if not contact_bodies:
        print("  Error: No contact body force columns identified.")
        return None

    # Calculate total 3D force vectors across all feet per frame
    f_total_x = np.zeros(len(df))
    f_total_y = np.zeros(len(df))
    f_total_z = np.zeros(len(df))
    
    for cb in contact_bodies:
        f_total_x += df[f"{cb}_force_x"].values
        f_total_y += df[f"{cb}_force_y"].values
        f_total_z += df[f"{cb}_force_z"].values

    # Determine time steps (dt)
    times = df["time"].values
    dts = np.diff(times)
    # Default to the median dt if we have variations or single frame bounds
    default_dt = np.median(dts) if len(dts) > 0 else 0.01

    # Approximate subject mass from vertical forces to initialize the filter:
    # m = Mean(Fz) / g (only considering frames with active ground force)
    active_fz = f_total_z[f_total_z > 50.0]
    if len(active_fz) > 0:
        calculated_mass = np.mean(active_fz) / 9.81
    else:
        calculated_mass = 70.0  # Fallback guess

    # Initialize 9D Kalman Filter
    kf = ComKalmanFilter(initial_mass=calculated_mass, pos_std=0.1, vel_std=0.1)

    # Placeholders for estimations
    est_vel_x = []
    est_vel_y = []
    est_vel_z = []

    # Run filter sequentially
    for i in range(len(df)):
        F_m = np.array([f_total_x[i], f_total_y[i], f_total_z[i]])
        dt = dts[i-1] if (i > 0 and i-1 < len(dts)) else default_dt
        
        kf.update(F_m, dt)
        
        vel = kf.com_velocity
        est_vel_x.append(vel[0])
        est_vel_y.append(vel[1])
        est_vel_z.append(vel[2])

    # Extract ground truth velocities
    gt_vel_x = df["com_vel_x"].values
    gt_vel_y = df["com_vel_y"].values
    gt_vel_z = df["com_vel_z"].values

    # Convert to arrays for calculations
    est_vel_x = np.array(est_vel_x)
    est_vel_y = np.array(est_vel_y)
    est_vel_z = np.array(est_vel_z)

    # Compute RMSE
    rmse_x = np.sqrt(np.mean((est_vel_x - gt_vel_x) ** 2))
    rmse_y = np.sqrt(np.mean((est_vel_y - gt_vel_y) ** 2))
    rmse_z = np.sqrt(np.mean((est_vel_z - gt_vel_z) ** 2))

    # Compute Pearson correlation coefficient (r)
    r_x = np.corrcoef(est_vel_x, gt_vel_x)[0, 1] if (np.std(est_vel_x) > 0 and np.std(gt_vel_x) > 0) else 0.0
    r_y = np.corrcoef(est_vel_y, gt_vel_y)[0, 1] if (np.std(est_vel_y) > 0 and np.std(gt_vel_y) > 0) else 0.0
    r_z = np.corrcoef(est_vel_z, gt_vel_z)[0, 1] if (np.std(est_vel_z) > 0 and np.std(gt_vel_z) > 0) else 0.0

    print(f"  Results (Velocity Evaluation):")
    print(f"    - X            : RMSE = {rmse_x:.4f} m/s | Correlation (r) = {r_x:.4f}")
    print(f"    - Y (Vertical) : RMSE = {rmse_y:.4f} m/s | Correlation (r) = {r_y:.4f}")
    print(f"    - Z            : RMSE = {rmse_z:.4f} m/s | Correlation (r) = {r_z:.4f}")

    return {
        "file": os.path.basename(file_path),
        "rmse": [rmse_x, rmse_y, rmse_z],
        "r": [r_x, r_y, r_z]
    }


def main():
    print("========================================")
    print("   COM Velocity Estimator Validation    ")
    print("========================================")
    
    if not os.path.exists(COM_VAL_DIR):
        print(f"Error: Validation directory '{COM_VAL_DIR}' does not exist.")
        print("Please run 'python src/addbiomechanics_export/export.py' first.")
        return

    files = find_validation_files(COM_VAL_DIR)
    if not files:
        print(f"No validation files (*_com_validation.parquet) found in: {COM_VAL_DIR}")
        return

    print(f"Found {len(files)} validation files.")
    results = []

    for file_path in files:
        res = run_validation_on_file(file_path)
        if res:
            results.append(res)

    if results:
        # Calculate summary across all processed files
        avg_rmse_x = np.mean([r["rmse"][0] for r in results])
        avg_rmse_y = np.mean([r["rmse"][1] for r in results])
        avg_rmse_z = np.mean([r["rmse"][2] for r in results])

        avg_r_x = np.mean([r["r"][0] for r in results])
        avg_r_y = np.mean([r["r"][1] for r in results])
        avg_r_z = np.mean([r["r"][2] for r in results])

        print("\n========================================")
        print("         OVERALL SUMMARY METRICS        ")
        print("========================================")
        print(f"Average RMSE (m/s):")
        print(f"  X: {avg_rmse_x:.4f} | Y: {avg_rmse_y:.4f} | Z: {avg_rmse_z:.4f}")
        print(f"Average Correlation (r):")
        print(f"  X: {avg_r_x:.4f} | Y: {avg_r_y:.4f} | Z: {avg_r_z:.4f}")
        print("========================================")


if __name__ == "__main__":
    main()