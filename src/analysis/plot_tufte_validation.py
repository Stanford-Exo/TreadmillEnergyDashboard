# File: src/analysis/plot_tufte_validation.py

import glob
import os
import sys

import numpy as np
import pandas as pd

# Resolve paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

from online_analyze.com_kf import ComKalmanFilter

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required to generate the validation plot.")
    print("Please install them: pip install matplotlib")
    sys.exit(1)

# Dynamic directories matching project structure
COM_VAL_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../exported_csvs/com_validation")
)
OUTPUT_PLOT_PATH = os.path.abspath(
    os.path.join(
        SCRIPT_DIR, "../../exported_csvs/com_validation/power_validation_tufte.pdf"
    )
)

# --- Clean, Modern Palette (Matching standard project colors) ---
COLOR_LEFT = "#EF4444"  # Soft Crimson (Left Leg)
COLOR_RIGHT = "#3B82F6"  # Readable Blue (Right Leg)
COLOR_TEXT = "#37352F"  # High-contrast dark charcoal
COLOR_GRID = "#EDEDED"  # Light background grid lines
COLOR_MUTED = "#787774"  # Muted subtext grey

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Inter",
    "-apple-system",
    "Helvetica",
    "Arial",
    "sans-serif",
]
plt.rcParams["text.color"] = COLOR_TEXT
plt.rcParams["axes.labelcolor"] = COLOR_TEXT
plt.rcParams["xtick.color"] = COLOR_MUTED
plt.rcParams["ytick.color"] = COLOR_MUTED


def load_and_calculate_powers():
    files = glob.glob(os.path.join(COM_VAL_DIR, "*.parquet"))
    files = [f for f in files if "Santos" not in f]  # Skip static calibration trials

    gt_powers_all = []
    est_powers_all = []
    limb_labels_all = []

    for file_path in files:
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Skipping {os.path.basename(file_path)} due to read error: {e}")
            continue

        if df.empty:
            continue

        # Dynamically map left and right contact bodies
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
            continue

        times = df["time"].values
        dts = np.diff(times)
        default_dt = np.median(dts) if len(dts) > 0 else 0.01

        # Calculate total 3D force vectors to drive the Kalman filter
        f_total_x = (
            df[f"{left_body}_force_x"].values + df[f"{right_body}_force_x"].values
        )
        f_total_y = (
            df[f"{left_body}_force_y"].values + df[f"{right_body}_force_y"].values
        )
        f_total_z = (
            df[f"{left_body}_force_z"].values + df[f"{right_body}_force_z"].values
        )

        active_fy = f_total_y[f_total_y > 50.0]
        calculated_mass = np.mean(f_total_y) / 9.81 if len(active_fy) > 0 else 70.0

        kf = ComKalmanFilter(initial_mass=calculated_mass)
        est_vel_x, est_vel_y, est_vel_z = [], [], []

        # Run filter pass
        for i in range(len(df)):
            vel = kf.com_velocity
            est_vel_x.append(vel[0])
            est_vel_y.append(vel[1])
            est_vel_z.append(vel[2])
            F_m = np.array([f_total_x[i], f_total_y[i], f_total_z[i]])
            dt = dts[i] if (i < len(dts)) else default_dt
            kf.update(F_m, dt)

        est_vel_x = np.array(est_vel_x)
        est_vel_y = np.array(est_vel_y)
        est_vel_z = np.array(est_vel_z)

        gt_vel_x = df["com_vel_x"].values
        gt_vel_y = df["com_vel_y"].values
        gt_vel_z = df["com_vel_z"].values

        for cb in [left_body, right_body]:
            fx = df[f"{cb}_force_x"].values
            fy = df[f"{cb}_force_y"].values
            fz = df[f"{cb}_force_z"].values

            # Mechanical Power = F . v
            gt_p = fx * gt_vel_x + fy * gt_vel_y + fz * gt_vel_z
            est_p = fx * est_vel_x + fy * est_vel_y + fz * est_vel_z

            gt_powers_all.extend(gt_p)
            est_powers_all.extend(est_p)
            limb_labels_all.extend([cb] * len(gt_p))

    return np.array(gt_powers_all), np.array(est_powers_all), np.array(limb_labels_all)


def style_subplot(ax, title_text):
    """Applies clean structural styling to each individual panel."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(COLOR_TEXT)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["left"].set_position(("outward", 12))

    ax.spines["bottom"].set_color(COLOR_TEXT)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["bottom"].set_position(("outward", 12))

    ax.tick_params(
        axis="both", which="both", colors=COLOR_TEXT, labelsize=9.5, width=1.0, length=4
    )
    ax.set_title(title_text, fontsize=12, fontweight="bold", pad=12, color=COLOR_TEXT)
    ax.grid(axis="both", color=COLOR_GRID, linestyle="-", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)


def main():
    if not os.path.exists(COM_VAL_DIR):
        print(f"Error: Validation directory '{COM_VAL_DIR}' not found.")
        print("Please run 'make export' first to generate validation files.")
        sys.exit(1)

    print("Ingesting validation trials and computing power transfers...")
    gt_powers, est_powers, limbs = load_and_calculate_powers()

    if len(gt_powers) == 0:
        print("Error: No valid validation datasets detected.")
        sys.exit(1)

    print(f"Plotting {len(gt_powers)} total coordinates across dual subplots...")

    # Establish side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6.2), sharex=True, sharey=True, facecolor="white"
    )

    # Global Limits
    min_val = min(gt_powers.min(), est_powers.min()) - 20
    max_val = max(gt_powers.max(), est_powers.max()) + 20
    unity_line = np.linspace(min_val, max_val, 100)

    # Clean list-comprehension string matching to prevent platform-dependent object dtype issues
    left_mask_all = np.array(
        ["_l" in str(x).lower() or "left" in str(x).lower() for x in limbs]
    )
    right_mask_all = ~left_mask_all

    # Subsample indices independently per subplot to keep plots readable
    np.random.seed(42)
    left_indices = np.where(left_mask_all)[0]
    right_indices = np.where(right_mask_all)[0]

    sample_size = 5000
    left_sample_idx = np.random.choice(
        left_indices, size=min(len(left_indices), sample_size), replace=False
    )
    right_sample_idx = np.random.choice(
        right_indices, size=min(len(right_indices), sample_size), replace=False
    )

    # -------------------------------------------------------------------------
    # Panel A: Left Leg (Red)
    # -------------------------------------------------------------------------
    style_subplot(ax1, "Left Leg Power Validation")

    # Unity line
    ax1.plot(
        unity_line,
        unity_line,
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        zorder=2,
    )

    # Scatter points
    ax1.scatter(
        gt_powers[left_sample_idx],
        est_powers[left_sample_idx],
        color=COLOR_LEFT,
        alpha=0.35,
        s=12,
        rasterized=True,
        zorder=3,
    )

    # Statistics Calculation (using all points for statistical accuracy)
    r_left = np.corrcoef(gt_powers[left_mask_all], est_powers[left_mask_all])[0, 1]
    stats_left = (
        f"Pearson $r$: {r_left:.4f}\nSample: $N$ = {len(left_indices):,} points"
    )
    ax1.text(
        0.05,
        0.95,
        stats_left,
        transform=ax1.transAxes,
        fontsize=10,
        color=COLOR_TEXT,
        va="top",
        ha="left",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor=COLOR_GRID,
            alpha=0.9,
            linewidth=1.0,
        ),
    )

    # -------------------------------------------------------------------------
    # Panel B: Right Leg (Blue)
    # -------------------------------------------------------------------------
    style_subplot(ax2, "Right Leg Power Validation")

    # Unity line
    ax2.plot(
        unity_line,
        unity_line,
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        zorder=2,
    )

    # Scatter points
    ax2.scatter(
        gt_powers[right_sample_idx],
        est_powers[right_sample_idx],
        color=COLOR_RIGHT,
        alpha=0.35,
        s=12,
        rasterized=True,
        zorder=3,
    )

    # Statistics Calculation (using all points for statistical accuracy)
    r_right = np.corrcoef(gt_powers[right_mask_all], est_powers[right_mask_all])[0, 1]
    stats_right = (
        f"Pearson $r$: {r_right:.4f}\nSample: $N$ = {len(right_indices):,} points"
    )
    ax2.text(
        0.05,
        0.95,
        stats_right,
        transform=ax2.transAxes,
        fontsize=10,
        color=COLOR_TEXT,
        va="top",
        ha="left",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor=COLOR_GRID,
            alpha=0.9,
            linewidth=1.0,
        ),
    )

    # -------------------------------------------------------------------------
    # Global Layout Formatting
    # -------------------------------------------------------------------------
    ax1.set_ylabel(
        "Estimated Real-Time Power (Our Method) (W)",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )

    # Position X-labels centrally on both subplots
    ax1.set_xlabel(
        "Musculoskeletal Reference Power (W)",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )
    ax2.set_xlabel(
        "Musculoskeletal Reference Power (W)",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )

    # Equal ranges across both panels
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, transparent=True)
    print(
        f"✅ Success! Saved updated dual-panel validation plot to: {OUTPUT_PLOT_PATH}"
    )


if __name__ == "__main__":
    main()
