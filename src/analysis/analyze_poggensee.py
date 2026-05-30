# File: src/analysis/analyze_poggensee.py

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

# Resolve absolute pathing relative to the src/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

from online_analyze.energy_analyzer import EnergyAnalyzer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

POGGENSEE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))

def main():
    parser = argparse.ArgumentParser(description="Analyze Poggensee exoskeleton trial datasets and plot stride power.")
    parser.add_argument("--dir", type=str, default=POGGENSEE_DIR, help="Directory containing the exported Poggensee parquet files")
    args = parser.parse_args()

    if plt is None:
        print("Error: The 'matplotlib' library is required to display plots. Please install it: pip install matplotlib")
        sys.exit(1)

    input_dir = os.path.abspath(args.dir)
    if not os.path.exists(input_dir):
        print(f"Error: Target directory '{input_dir}' does not exist. Please process the exports first.")
        sys.exit(1)

    files = glob.glob(os.path.join(input_dir, "*.parquet"))
    if not files:
        print(f"No parquet files found in {input_dir}.")
        sys.exit(0)

    print(f"Found {len(files)} Poggensee trial dataset(s). Processing...")

    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        print(f"\nProcessing trial: {filename}")

        df = pd.read_parquet(file_path)
        if df.empty:
            print("  Skipping: empty dataset.")
            continue

        # Map contact bodies dynamically
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
            print("  Warning: Could not automatically resolve bilateral contact body pairings. Skipping.")
            continue

        print(f"  Mapped contact bodies: Left='{left_body}', Right='{right_body}'")

        # Extract time and temporal details
        times = df["time"].values
        dts = np.diff(times)
        default_dt = np.median(dts) if len(dts) > 0 else 0.01

        # Calculate estimated mass based on vertical forces
        f_total_y = df[f"{left_body}_force_y"].values + df[f"{right_body}_force_y"].values
        active_fy = f_total_y[f_total_y > 50.0]
        calculated_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

        analyzer = EnergyAnalyzer(initial_mass=calculated_mass)

        # Feed frame sequence to analyzer
        for i in range(min(len(df), 50000)):
            if i % 10000 == 0:
                print(f"  Processing frame {i}/{len(df)}...")
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
            # print(f"  Frame {i}: Left Force={forces['left']}, Right Force={forces['right']}")
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

        # Retrieve gait parameter metrics
        stats = analyzer.stride_analyzer.get_metrics_summary()
        print(f"  Strides analyzed: {stats.get('stride_duration_count', 0)}")
        print(f"  Treadmill Speed Estimate: {stats.get('estimated_belt_speed', 0.0):.3f} m/s")

        # Plot curves
        aggs = analyzer.get_aggregate_profiles()
        x = np.linspace(0, 100, 100)

        plt.figure(figsize=(8, 5))
        for key, col, label in [('left', 'red', 'Left Foot'), ('right', 'blue', 'Right Foot'), ('total', 'black', 'Total Power')]:
            mean_profile = np.array(aggs[f"{key}_mean"])
            std_profile = np.array(aggs[f"{key}_std"])

            plt.plot(x, mean_profile, color=col, label=label, linewidth=2)
            plt.fill_between(x, mean_profile - std_profile, mean_profile + std_profile, color=col, alpha=0.15)

        plt.title(f"Poggensee Stride-Average Power Profile: {filename}", fontsize=11, fontweight='bold')
        plt.xlabel("Gait Cycle Percentage (%)")
        plt.ylabel("Mechanical Power (W)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()

        print(f"  Displaying stride-average plot for {filename}...")
        plt.show()

if __name__ == "__main__":
    main()