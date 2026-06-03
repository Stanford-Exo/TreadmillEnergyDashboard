# File: src/analysis/correlate_metabolics.py

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

# Setup optional plotting module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except ImportError:
    plt = None

# Pointing to the precomputed Wide-Format Parquet file
DEFAULT_PARQUET_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../exported_pogensee/precomputed_poggensee.parquet")
)

# --- Tufte/Notion Aesthetics ---
if plt is not None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "Arial", "sans-serif"]

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"


def process_within_trial_changes(df, use_no_achilles=False):
    """
    Processes the raw windowed dataset to compute within-trial changes (deltas)
    after applying filters to remove metabolic outliers and trial cooldowns.
    Changes are computed down from the maximum metabolic and mechanical rates.
    """
    delta_data = []

    # Analyze on a per-trial basis
    for trial_name, group in df.groupby("trial_name"):
        # 1. Sort sequentially by chronological time
        group = group.sort_values(by="window_start_s").reset_index(drop=True)

        # 2. Drop the last two windows (last 10 minutes) to eliminate cooldown/slowdown artifacts
        if len(group) > 2:
            group = group.iloc[:-2].reset_index(drop=True)
        else:
            continue

        # 3. Apply absolute bounds to catch mask-off outliers (< 50W or > 600W)
        group = group[
            (group["net_bio_cost_w"] >= 50.0) & (group["net_bio_cost_w"] <= 600.0)
        ].reset_index(drop=True)
        if group.empty:
            continue

        # 4. Filter out extreme transients (deviating from the trial median by 100W or more)
        trial_median_bio = group["net_bio_cost_w"].median()
        group = group[
            np.abs(group["net_bio_cost_w"] - trial_median_bio) < 100.0
        ].reset_index(drop=True)

        if len(group) < 2:
            # We need at least two windows to evaluate changes within a trial
            continue

        # Define target column based on the Achilles extraction model
        mech_col = (
            "mechanical_power_no_achilles" if use_no_achilles else "mechanical_power"
        )
        mech_std_col = (
            "mechanical_power_no_achilles_std"
            if use_no_achilles
            else "mechanical_power_std"
        )
        bio_std_col = "net_bio_cost_std_w"

        # 5. Determine the maximum metabolic and mechanical rates within this trial
        max_bio_idx = group["net_bio_cost_w"].idxmax()
        max_mech_idx = group[mech_col].idxmax()

        max_bio = group.loc[max_bio_idx, "net_bio_cost_w"]
        max_mech = group.loc[max_mech_idx, mech_col]

        max_bio_std = (
            group.loc[max_bio_idx, bio_std_col] if bio_std_col in group.columns else 0.0
        )
        max_mech_std = (
            group.loc[max_mech_idx, mech_std_col]
            if mech_std_col in group.columns
            else 0.0
        )

        for idx, row in group.iterrows():
            # Changes calculated down from the maximum values (guaranteeing delta <= 0)
            delta_bio = row["net_bio_cost_w"] - max_bio
            delta_mech = row[mech_col] - max_mech

            curr_bio_std = row[bio_std_col] if bio_std_col in row else 0.0
            curr_mech_std = row[mech_std_col] if mech_std_col in row else 0.0

            # Propagate error: sqrt(std_current^2 + std_max^2)
            delta_bio_std = np.sqrt(curr_bio_std**2 + max_bio_std**2)
            delta_mech_std = np.sqrt(curr_mech_std**2 + max_mech_std**2)

            delta_data.append(
                {
                    "trial_name": trial_name,
                    "window_start_s": row["window_start_s"],
                    "delta_bio_w": delta_bio,
                    "delta_mech_w": delta_mech,
                    "delta_bio_std_w": delta_bio_std,
                    "delta_mech_std_w": delta_mech_std,
                    "num_valid_strides": row["num_valid_strides"]
                    if "num_valid_strides" in row
                    else 250,
                    "raw_bio_w": row["net_bio_cost_w"],
                    "raw_mech_w": row[mech_col],
                }
            )

    return pd.DataFrame(delta_data)


def plot_delta_correlation(df_deltas, use_no_achilles=False):
    """
    Plots the scatter correlation between the change in Estimated Mechanical Power
    and the change in Measured Net Biological Cost relative to trial maximums.
    Visualizes bivariate standard deviations using semi-transparent 1-sigma uncertainty ellipses.
    Forces the regression line through the origin and frames (0,0) in the upper-right corner.
    """
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)

    # Title & Subtitle based on the Achilles tendon configuration
    if use_no_achilles:
        title = "Within-Trial Changes: Delta Mechanics (NO Achilles Model) vs. Delta Metabolics"
        subtitle = "Correlating within-subject changes with standard error ellipses (cooldowns & outliers excluded)."
        x_label = "Change in Est. Mechanical Cost ($\Delta$ Mechanical Power) (W)"
    else:
        title = "Within-Trial Changes: Delta Mechanics (Achilles Extracted) vs. Delta Metabolics"
        subtitle = "Correlating within-subject changes with standard error ellipses (cooldowns & outliers excluded)."
        x_label = "Change in Est. Mechanical Muscle Cost ($\Delta$ Muscle Power) (W)"

    fig.text(0.05, 0.93, title, fontsize=13, fontweight="bold", color=NOTION_TEXT)
    fig.text(0.05, 0.89, subtitle, fontsize=9.5, color=NOTION_SUBTEXT)

    # Shifted 'right' from 0.95 to 0.72 to reserve 28% of the canvas on the right for the legend
    plt.subplots_adjust(top=0.82, right=0.72, bottom=0.15)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(NOTION_TEXT)

    ax.tick_params(axis="both", colors=NOTION_TEXT, length=4)
    ax.grid(color="#F1F5F9", linestyle="-", linewidth=0.5)  # Muted, delicate gridlines
    ax.set_axisbelow(True)

    x_vals = df_deltas["delta_mech_w"].values
    y_vals = df_deltas["delta_bio_w"].values

    # Color coordinate different trials to highlight within-subject trajectory groupings
    unique_trials = df_deltas["trial_name"].unique()
    colormap = cm.get_cmap("tab10", max(len(unique_trials), 10))

    for i, trial_name in enumerate(unique_trials):
        sub_df = df_deltas[df_deltas["trial_name"] == trial_name]

        # Plot centers (slightly smaller, white borders)
        ax.scatter(
            sub_df["delta_mech_w"],
            sub_df["delta_bio_w"],
            color=colormap(i),
            s=30,
            edgecolors="white",
            linewidths=0.5,
            label=trial_name,
            zorder=4,
        )

        # Plot uncertainty ellipses scaled down to Standard Error of the Mean (SEM)
        # Using hollow ellipses (facecolor='none') to eliminate color overlaps/mud
        for _, row in sub_df.iterrows():
            x_val = row["delta_mech_w"]
            y_val = row["delta_bio_w"]
            x_std = row["delta_mech_std_w"] if "delta_mech_std_w" in row else 0.0
            y_std = row["delta_bio_std_w"] if "delta_bio_std_w" in row else 0.0

            n_strides = row["num_valid_strides"] if "num_valid_strides" in row else 250
            if pd.isna(n_strides) or n_strides <= 0:
                n_strides = 250
            n_breaths = 15

            x_sem = x_std / np.sqrt(n_strides)
            y_sem = y_std / np.sqrt(n_breaths)

            ellipse = Ellipse(
                xy=(x_val, y_val),
                width=2 * x_sem,
                height=2 * y_sem,
                angle=0,
                facecolor="none",  # Hollow faces prevent color mud
                edgecolor=colormap(i),
                alpha=0.35,  # Legible outline alpha
                linewidth=0.6,  # Delicate Tufte line weight
                zorder=3,
            )
            ax.add_patch(ellipse)

    # Reference cross-hairs at delta origin (0, 0)
    ax.axhline(0, color=NOTION_SUBTEXT, linestyle=":", linewidth=0.8, zorder=1)
    ax.axvline(0, color=NOTION_SUBTEXT, linestyle=":", linewidth=0.8, zorder=1)

    # Perform regression forced through the origin: y = m * x
    # Slope coefficient m = sum(x*y) / sum(x^2)
    sum_xx = np.sum(x_vals**2)
    slope = np.sum(x_vals * y_vals) / sum_xx if sum_xx > 0 else 0.0

    # Calculate R^2 for fit through the origin: R^2 = 1 - sum((y - y_fit)^2) / sum(y^2)
    y_fit = slope * x_vals
    ss_res = np.sum((y_vals - y_fit) ** 2)
    ss_tot = np.sum(y_vals**2)
    r2_origin = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Standard Pearson r correlation coefficient
    r_corr = np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 1 else 0.0

    # Define plot bounds based on the minimum values to position (0,0) in the upper-right corner
    min_x = np.min(x_vals)
    min_y = np.min(y_vals)

    range_x = abs(min_x) if min_x != 0 else 1.0
    range_y = abs(min_y) if min_y != 0 else 1.0

    pad_left = range_x * 0.1
    pad_bottom = range_y * 0.1
    pad_right = range_x * 0.05
    pad_top = range_y * 0.05

    # Define regression line from the minimum coordinate up to the origin
    x_line = np.linspace(min_x - pad_left, 0.0, 100)
    y_line = slope * x_line

    ax.plot(
        x_line,
        y_line,
        color=NOTION_TEXT,
        linewidth=2.0,
        zorder=2,
        label=f"Origin Fit (y = {slope:.2f}x)",
    )

    # Calculate p-value for regression through the origin
    n = len(x_vals)
    if n > 1 and sum_xx > 0:
        s2 = ss_res / (n - 1)
        se_slope = np.sqrt(s2 / sum_xx)
        t_stat = slope / se_slope if se_slope > 0 else 0.0
        p_val = 2 * (1 - t.cdf(abs(t_stat), df=n - 1))
    else:
        se_slope = 0.0
        p_val = 1.0

    stats_text = (
        f"$R^2$ (through origin): {r2_origin:.3f}\n"
        f"Pearson $r$: {r_corr:.3f}\n"
        f"Slope: {slope:.3f} ± {se_slope:.3f}\n"
        f"p-value: {p_val:.2e}\n"
        f"N: {n} segments"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        ha="left",
        color=NOTION_TEXT,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=NOTION_BG,
            edgecolor="#EDEDED",
            alpha=0.9,
        ),
    )

    ax.set_xlabel(x_label, fontsize=10, fontweight="bold", color=NOTION_TEXT)
    ax.set_ylabel(
        "Change in Net Measured Biological Cost ($\Delta$ Metabolics) (W)",
        fontsize=10,
        fontweight="bold",
        color=NOTION_TEXT,
    )

    # Restrict axes so the plot sits entirely in the negative quadrant with the origin framed upper-right
    ax.set_xlim(min_x - pad_left, pad_right)
    ax.set_ylim(min_y - pad_bottom, pad_top)

    # Set the legend anchor slightly to the right of the axes within the new whitespace area
    ax.legend(
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Correlate within-trial adaptation changes in estimated mechanics vs metabolics."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_PARQUET_PATH,
        help="Path to the precomputed_poggensee.parquet file.",
    )
    parser.add_argument(
        "--no-achilles",
        action="store_true",
        help="Correlate muscle cost changes assuming NO Achilles tendon storage.",
    )
    args = parser.parse_args()

    if plt is None:
        print("Error: matplotlib required. pip install matplotlib")
        sys.exit(1)

    parquet_path = os.path.abspath(args.file)
    if not os.path.exists(parquet_path):
        print(f"Error: Could not find Parquet file at {parquet_path}")
        print(
            "Please run 'python src/analysis/precompute_poggensee.py' first to generate the data."
        )
        sys.exit(1)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading Parquet: {e}")
        sys.exit(1)

    if df.empty:
        print(f"The file at {parquet_path} is empty. No data to plot.")
        sys.exit(1)

    # Process and extract within-trial changes relative to maximums
    df_deltas = process_within_trial_changes(df, use_no_achilles=args.no_achilles)

    if df_deltas.empty:
        print(
            "Error: No valid within-trial change points remain after filtering. Verify your dataset values."
        )
        sys.exit(1)

    print(
        f"Successfully compiled {len(df_deltas)} change segments across {len(df_deltas['trial_name'].unique())} unique trials."
    )
    plot_delta_correlation(df_deltas, use_no_achilles=args.no_achilles)


if __name__ == "__main__":
    main()
