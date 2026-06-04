# File: src/analysis/plot_metabolics_diagnostic.py

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))

# --- Aesthetic Styling ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "Arial", "sans-serif"]


def plot_metabolics_for_file(file_path, current_idx, total_files):
    filename = os.path.basename(file_path)
    print(f"\n[{current_idx}/{total_files}] Loading: {filename}")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  -> Error reading file: {e}")
        return

    time = df["time"].values / 60.0  # Convert to minutes

    vo2_col = next((c for c in df.columns if c.lower() == "vo2"), None)
    vco2_col = next((c for c in df.columns if c.lower() == "vco2"), None)

    if not vo2_col or not vco2_col:
        print("  -> Warning: Could not find VO2 or VCO2 columns. Skipping.")
        return

    vo2_raw = df[vo2_col].values
    vco2_raw = df[vco2_col].values

    # 1. Base validity mask (no zeroes, no NaNs)
    valid_mask = (vo2_raw > 0) & (vco2_raw > 0) & (~np.isnan(vo2_raw))

    # 2. Down-sampling mask: Find frames where the metabolic values actually change
    # We use > 1e-3 to safely ignore tiny floating-point artifact noise from saving/loading
    vo2_changed = np.concatenate(([True], np.abs(vo2_raw[1:] - vo2_raw[:-1]) > 1e-3))
    vco2_changed = np.concatenate(([True], np.abs(vco2_raw[1:] - vco2_raw[:-1]) > 1e-3))

    is_update_frame = vo2_changed | vco2_changed

    # Combine masks to extract only valid, distinct breath updates
    final_mask = valid_mask & is_update_frame

    if not np.any(final_mask):
        print(
            "  -> Warning: No valid metabolic breath data found in this file. Skipping."
        )
        return

    t_valid = time[final_mask]
    vo2 = vo2_raw[final_mask]
    vco2 = vco2_raw[final_mask]

    print(
        f"  -> Down-sampled from {len(df):,} raw frames to {len(vo2):,} distinct breath updates."
    )

    # Calculate Physiological Metrics
    rer = vco2 / vo2
    # Brockway Equation -> Watts
    watts = ((3.941 * vo2) + (1.106 * vco2)) * 4.184 / 60.0

    # Create Plot
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True, facecolor="#FFFFFF")

    title = f"Metabolic Diagnostics: {filename}"
    subtitle = f"File {current_idx} of {total_files} | Close this window to automatically load the next trial."
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)
    fig.text(0.5, 0.92, subtitle, ha="center", fontsize=11, color="#4B5563")

    # Styling function for subplots
    def style_ax(ax):
        ax.set_facecolor("#FFFFFF")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(True, linestyle="-", color="#EDEDED", alpha=1.0)
        ax.set_axisbelow(True)

    # 1. Plot VO2 & VCO2
    style_ax(axs[0])
    axs[0].plot(
        t_valid,
        vo2,
        "o-",
        color="#10B981",
        markersize=4,
        alpha=0.8,
        linewidth=1.0,
        label="VO2 (O2 Consumption)",
    )
    axs[0].plot(
        t_valid,
        vco2,
        "s-",
        color="#8B5CF6",
        markersize=4,
        alpha=0.8,
        linewidth=1.0,
        label="VCO2 (CO2 Production)",
    )
    axs[0].set_ylabel("mL/min", fontweight="bold")
    axs[0].set_title(
        "1. Raw Gas Exchange (Look for sudden drops = Mask Leaks)",
        fontweight="bold",
        color="#374151",
    )
    axs[0].legend(loc="upper left")

    # 2. Plot Respiratory Exchange Ratio (RER)
    style_ax(axs[1])
    axs[1].plot(
        t_valid,
        rer,
        "o-",
        color="#F59E0B",
        markersize=4,
        alpha=0.8,
        linewidth=1.0,
        label="RER (VCO2 / VO2)",
    )
    axs[1].axhspan(
        0.7,
        1.0,
        color="#10B981",
        alpha=0.1,
        label="Normal Physiological Range (0.7 - 1.0)",
    )
    axs[1].axhline(1.0, color="#EF4444", linestyle="--", linewidth=1.5)
    axs[1].axhline(0.7, color="#EF4444", linestyle="--", linewidth=1.5)
    axs[1].set_ylabel("RER Ratio", fontweight="bold")
    axs[1].set_title(
        "2. Respiratory Exchange Ratio (Values > 1.0 = Hyperventilation/Talking)",
        fontweight="bold",
        color="#374151",
    )
    axs[1].legend(loc="upper left")

    # 3. Plot Estimated Gross Biological Power
    style_ax(axs[2])
    axs[2].plot(
        t_valid,
        watts,
        "o-",
        color="#EF4444",
        markersize=4,
        alpha=0.6,
        linewidth=0.8,
        label="Gross Biological Watts (Breath-by-Breath)",
    )

    # Overlay a rolling median (robust to spikes)
    # 50 breaths is approximately 1 minute of rolling data (assuming ~1.2s per breath)
    window_size = min(50, len(watts) // 10)
    if window_size > 0:
        rolling_watts = (
            pd.Series(watts).rolling(window=window_size, center=True).median()
        )
        axs[2].plot(
            t_valid,
            rolling_watts,
            "-",
            color="#111827",
            linewidth=2.5,
            label=f"Rolling Median (~1 min)",
        )

    axs[2].set_ylabel("Watts (W)", fontweight="bold")
    axs[2].set_xlabel("Trial Time (Minutes)", fontweight="bold", fontsize=11)
    axs[2].set_title(
        "3. Calculated Biological Power (Look for high-variance noise dragging down averages)",
        fontweight="bold",
        color="#374151",
    )
    axs[2].legend(loc="upper left")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for subtitles

    print("  -> Displaying plot. Close the window when done to load the next one.")
    plt.show()  # This blocks script execution until the user closes the window


def main():
    parser = argparse.ArgumentParser(
        description="Automatically cycle through and diagnose down-sampled ADAPT metabolic data."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_DIR,
        help="Directory containing exported parquet files",
    )
    args = parser.parse_args()

    target_dir = os.path.abspath(args.dir)
    if not os.path.exists(target_dir):
        print(f"Error: Directory not found at {target_dir}")
        sys.exit(1)

    # Find all parquet files containing 'ADAPT'
    search_pattern = os.path.join(target_dir, "*ADAPT*.parquet")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"No files containing 'ADAPT' found in {target_dir}")
        sys.exit(0)

    print(
        f"Found {len(files)} ADAPT trial files. Starting interactive visualization loop..."
    )

    for idx, file_path in enumerate(files):
        plot_metabolics_for_file(file_path, idx + 1, len(files))

    print("\nAll ADAPT files have been reviewed.")


if __name__ == "__main__":
    main()
