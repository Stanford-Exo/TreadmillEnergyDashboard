# File: src/validation/validate_strides.py

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import json

# Resolve absolute pathing relative to execution roots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../../src")))

from online_analyze.energy_analyzer import EnergyAnalyzer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

COM_VAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_csvs/com_validation"))
OUT_ANALYSIS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_csvs/stride_analysis"))


def main():
    parser = argparse.ArgumentParser(description="Processes parquet exports to evaluate spatial and temporal stride metrics.")
    parser.add_argument("--plot", action="store_true", help="Generate average gait power curve figures")
    args = parser.parse_args()

    if args.plot and plt is None:
        print("Error: matplotlib is required to plot. Please run: pip install matplotlib")
        sys.exit(1)

    if not os.path.exists(COM_VAL_DIR):
        print(f"Error: Validation source directory '{COM_VAL_DIR}' not found. Execute exports first.")
        sys.exit(1)

    os.makedirs(OUT_ANALYSIS_DIR, exist_ok=True)
    files = glob.glob(os.path.join(COM_VAL_DIR, "*.parquet"))
    
    if not files:
        print(f"No validation files found in {COM_VAL_DIR}.")
        sys.exit(0)

    print(f"Processing {len(files)} trial datasets...")
    overall_dataset_metrics = []

    for file_path in files:
        filename = os.path.basename(file_path)
        if 'Santos' in filename:
            print(f"  Skipping static trial: {filename}")
            continue

        print(f"\nEvaluating: {filename}")
        df = pd.read_parquet(file_path)
        if df.empty:
            continue

        # Extract dynamic mapping labels
        force_cols = [col for col in df.columns if col.endswith("_force_y")]
        contact_bodies = [col.replace("_force_y", "") for col in force_cols]

        left_body, right_body = None, None
        for cb in contact_bodies:
            if cb.endswith("_l") or "left" in cb.lower():
                left_body = cb
            elif cb.endswith("_r") or "right" in cb.lower():
                right_body = cb

        if left_body is None and contact_bodies:
            left_body = contact_bodies[0]
        if right_body is None and len(contact_bodies) > 1:
            right_body = contact_bodies[1]

        if not left_body or not right_body:
            print("  Error: Could not identify bilateral contact body names.")
            continue

        print(f"  Mapped Left: '{left_body}' | Right: '{right_body}'")

        # Initialize tracking elements
        times = df["time"].values
        dts = np.diff(times)
        default_dt = np.median(dts) if len(dts) > 0 else 0.01

        f_total_y = df[f"{left_body}_force_y"].values + df[f"{right_body}_force_y"].values
        active_fy = f_total_y[f_total_y > 50.0]
        calculated_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

        analyzer = EnergyAnalyzer(initial_mass=calculated_mass)

        # Feed the step stream sequentially
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
            analyzer.update(times[i], forces, cops, dt)

        # Print spatial metrics
        stats = analyzer.stride_analyzer.get_metrics_summary()
        print(f"  Gait Parameters Evaluated:")
        print(f"    - Completed Strides: {stats.get('stride_duration_count', 0)}")
        print(f"    - Stride Duration: {stats.get('stride_duration_mean', 0.0):.3f} s (± {stats.get('stride_duration_std', 0.0):.3f})")
        print(f"    - Stride Frequency: {stats.get('stride_frequency_mean', 0.0):.3f} Hz")
        print(f"    - Stride Length: {stats.get('stride_length_mean', 0.0):.3f} m (± {stats.get('stride_length_std', 0.0):.3f})")
        print(f"    - Step Width: {stats.get('step_width_mean', 0.0):.3f} m (± {stats.get('step_width_std', 0.0):.3f})")
        print(f"    - Duty Factor: {stats.get('duty_factor_mean', 0.0):.3f}")
        print(f"    - Treadmill Speed Estimate: {stats.get('estimated_belt_speed', 0.0):.3f} m/s")

        stats['file_name'] = filename
        overall_dataset_metrics.append(stats)

        # Plot curves
        if args.plot and plt is not None:
            aggs = analyzer.get_aggregate_profiles()
            x = np.linspace(0, 100, 100)
            
            plt.figure(figsize=(8, 5))
            for key, col, label in [('left', 'red', 'Left Foot'), ('right', 'blue', 'Right Foot'), ('total', 'black', 'Total Power')]:
                mean_profile = np.array(aggs[f"{key}_mean"])
                std_profile = np.array(aggs[f"{key}_std"])
                
                plt.plot(x, mean_profile, color=col, label=label, linewidth=2)
                plt.fill_between(x, mean_profile - std_profile, mean_profile + std_profile, color=col, alpha=0.15)
                
            plt.title(f"Stride-Average Power Profile: {filename}", fontsize=11, fontweight='bold')
            plt.xlabel("Gait Cycle Percentage (%)")
            plt.ylabel("Mechanical Power (W)")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()
            
            print(f"    Displaying stride-average plot for {filename}...")
            plt.show()

    # Export compiled dataset summaries
    if overall_dataset_metrics:
        out_df = pd.DataFrame(overall_dataset_metrics)
        csv_path = os.path.join(OUT_ANALYSIS_DIR, "dataset_stride_summary.csv")
        out_df.to_csv(csv_path, index=False)
        print(f"\nDataset metrics written to: {csv_path}")


if __name__ == "__main__":
    main()