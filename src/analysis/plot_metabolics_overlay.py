# File: src/analysis/plot_metabolics_subject_day.py

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))

# --- Tufte/Notion Aesthetics ---
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Please install it: pip install matplotlib")
    sys.exit(1)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "Arial", "sans-serif"]

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"

# Standardized color map for each type of trial condition
COLOR_MAP = {
    "QS": "#6366F1",  # Indigo Blue (Quiet Standing baseline)
    "NW": "#10B981",  # Emerald Green (Normal Walking)
    "ZT": "#4B5563",  # Charcoal/Slate (Zero Torque - ZT1 & ZT2)
    "GA": "#2563EB",  # Royal Blue (Generic Assistance - GA1 & GA2)
    "OP": "#EF4444",  # Crimson Red (Optimized Assistance - OP1 & OP2)
}


def compute_metabolics_time_series(df):
    """
    Strips invalid frames, downsamples metabolic updates,
    and applies the Brockway equation to yield a time-series of Watts.
    """
    time = df["time"].values / 60.0  # Convert to minutes

    vo2_col = next((c for c in df.columns if c.lower() == "vo2"), None)
    vco2_col = next((c for c in df.columns if c.lower() == "vco2"), None)

    if not vo2_col or not vco2_col:
        return None, None

    vo2_raw = df[vo2_col].values
    vco2_raw = df[vco2_col].values

    # Base validity mask (no zeroes, no NaNs)
    valid_mask = (vo2_raw > 0) & (vco2_raw > 0) & (~np.isnan(vo2_raw))

    # Downsampling mask: isolate when breath values update (floating-point safe > 1e-3)
    vo2_changed = np.concatenate(([True], np.abs(vo2_raw[1:] - vo2_raw[:-1]) > 1e-3))
    vco2_changed = np.concatenate(([True], np.abs(vco2_raw[1:] - vco2_raw[:-1]) > 1e-3))
    is_update_frame = vo2_changed | vco2_changed

    final_mask = valid_mask & is_update_frame

    if not np.any(final_mask):
        return None, None

    t_valid = time[final_mask]
    vo2 = vo2_raw[final_mask]
    vco2 = vco2_raw[final_mask]

    # Brockway Equation -> Watts
    watts = ((3.941 * vo2) + (1.106 * vco2)) * 4.184 / 60.0

    return t_valid, watts


def parse_file_group_and_condition(filename):
    """
    Parses a filename to extract the Subject_Day group prefix and the condition.
    Example: 'Continued_Optimization_1_Day2_GA1.parquet'
             -> Group: 'Continued_Optimization_1_Day2'
             -> Condition: 'GA'
             -> Label: 'GA1'
    """
    base = os.path.splitext(filename)[0]

    # Matches condition strings like _GA1, _GA2, _QS, etc.
    match = re.search(r"_(QS|GA\d|OP\d|ZT\d|NW\d)$", base, re.IGNORECASE)
    if not match:
        return None, None, None

    condition_raw = match.group(1).upper()
    group_prefix = base[: match.start()]

    # Classify clean category (e.g. ZT1 -> ZT, QS -> QS)
    category = re.sub(r"\d", "", condition_raw)

    return group_prefix, category, condition_raw


def get_steady_state_mean(t, w):
    """Calculates steady-state median power from the final 3 minutes."""
    if len(t) == 0:
        return 0.0
    t_end = t[-1]
    ss_mask = t >= (t_end - 3.0)
    return float(np.median(w[ss_mask])) if np.any(ss_mask) else float(np.median(w))


def plot_subject_day_group(group_key, file_infos, current_idx, total_groups):
    print(f"\n[{current_idx}/{total_groups}] Plotting Subject/Day: {group_key}")

    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)

    title = f"Metabolic Profiles: {group_key.replace('_', ' ')}"
    subtitle = f"Group {current_idx} of {total_groups} | All conditions overlaid starting from 0 minutes"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96, color=NOTION_TEXT)
    fig.text(0.5, 0.91, subtitle, ha="center", fontsize=10, color=NOTION_SUBTEXT)

    plt.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.74)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(NOTION_TEXT)

    ax.tick_params(axis="both", colors=NOTION_TEXT, length=4, labelsize=9)
    ax.grid(color=NOTION_GRID, linestyle="-", linewidth=1.0)
    ax.set_axisbelow(True)

    legend_tracker = set()
    summary_data = []
    max_duration = 0.0

    # Sort files so QS and NW are plotted underneath, with active exoskeleton conditions on top
    sort_order = {"QS": 0, "NW": 1, "ZT": 2, "GA": 3, "OP": 4}
    sorted_files = sorted(file_infos, key=lambda x: sort_order.get(x["category"], 5))

    for finfo in sorted_files:
        path = finfo["path"]
        category = finfo["category"]
        raw_label = finfo["raw_label"]
        color = COLOR_MAP.get(category, "#7C3AED")

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  -> Error reading {raw_label}: {e}")
            continue

        t, w = compute_metabolics_time_series(df)
        if t is None or len(t) == 0:
            continue

        # --- Shift timeline to start at exactly 0.0 minutes ---
        t_norm = t - t[0]
        max_duration = max(max_duration, t_norm[-1])

        # Calculate steady state metric
        ss_mean = get_steady_state_mean(t_norm, w)
        summary_data.append((category, raw_label, ss_mean))

        # Check if we have already listed this base category in the legend
        label_in_legend = category if category not in legend_tracker else None
        legend_tracker.add(category)

        # Plot raw breath updates
        ax.plot(
            t_norm,
            w,
            "o-",
            color=color,
            markersize=3.5,
            alpha=0.18,
            linewidth=0.6,
            label=None,
        )

        # Plot robust rolling average (approx 1-minute window)
        window_size = min(35, len(w) // 5)
        if window_size > 2:
            rolling_w = pd.Series(w).rolling(window=window_size, center=True).median()
            ax.plot(
                t_norm,
                rolling_w,
                "-",
                color=color,
                linewidth=2.2,
                label=label_in_legend,
            )
        else:
            # Fallback if trial is too short to construct rolling window
            ax.plot(
                t_norm,
                w,
                "-",
                color=color,
                linewidth=2.2,
                label=label_in_legend,
            )

    ax.set_ylabel(
        "Biological Power (W)", fontsize=11, fontweight="bold", color=NOTION_TEXT
    )
    ax.set_xlabel(
        "Trial Time (Minutes from Start)",
        fontsize=11,
        fontweight="bold",
        color=NOTION_TEXT,
    )
    ax.set_xlim(-0.5, max_duration + 0.5)

    # Compile the floating summary panel on the right sidebar space
    summary_str = "Steady-State Averages:\n(Final 3 minutes)\n"
    sorted_summary = sorted(summary_data, key=lambda x: (sort_order.get(x[0], 5), x[1]))

    for cat, lbl, val in sorted_summary:
        summary_str += f"\n  {lbl:<4} : {val:.1f} W"

    # Place summary text box off the main plotting area on the right
    ax.text(
        1.03,
        0.5,
        summary_str,
        transform=ax.transAxes,
        fontsize=9.5,
        va="center",
        ha="left",
        color=NOTION_TEXT,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor=NOTION_BG,
            edgecolor=NOTION_GRID,
            alpha=0.95,
        ),
    )

    ax.legend(
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.03, 0.95),
    )

    print("  -> Displaying plot. Close the window to load the next subject/day.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay and compare all metabolic conditions for each unique Subject & Day."
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

    all_files = sorted(glob.glob(os.path.join(target_dir, "*.parquet")))

    # Group trials by Subject/Day prefix
    groups = {}

    for f in all_files:
        base = os.path.basename(f)
        if "ADAPT" in base:
            continue

        group_key, category, raw_label = parse_file_group_and_condition(base)
        if group_key and category:
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(
                {"path": f, "category": category, "raw_label": raw_label}
            )

    if not groups:
        print(f"No valid non-ADAPT trials to group under: {target_dir}")
        return

    print(f"Discovered {len(groups)} unique Subject/Day experimental groups.")

    for idx, (group_key, file_infos) in enumerate(sorted(groups.items())):
        plot_subject_day_group(group_key, file_infos, idx + 1, len(groups))

    print("\nAll overlay visualizations complete.")


if __name__ == "__main__":
    main()
