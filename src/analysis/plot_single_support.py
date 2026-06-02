# File: src/analysis/plot_single_support.py

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Setup paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DEFAULT_DIR = os.path.join(REPO_ROOT, "exported_pogensee")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required to plot histograms.")
    print("Please install it by running: pip install matplotlib")
    sys.exit(1)

# --- Aesthetic Styling (Notion/Tufte Muted Style) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "Arial", "sans-serif"]

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"

COLOR_LEFT = "#F87171"  # Soft Red for Left
COLOR_RIGHT = "#60A5FA"  # Soft Blue for Right


def calculate_single_support_durations(df, contact_threshold=30.0):
    """
    Identifies contiguous sequences of single-support phases for left and right feet.
    Returns durations in seconds.
    """
    times = df["time"].values

    # Identify active contact states based on vertical force
    l_contact = (df["calcn_l_force_y"] > contact_threshold).values
    r_contact = (df["calcn_r_force_y"] > contact_threshold).values

    # Single Support Definitions
    l_ss = l_contact & ~r_contact
    r_ss = r_contact & ~l_contact

    def get_runs(boolean_mask):
        if not np.any(boolean_mask):
            return np.array([])

        # Detect state changes
        diff = np.diff(boolean_mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Adjust for boundary conditions
        if boolean_mask[0]:
            starts = np.insert(starts, 0, 0)
        if boolean_mask[-1]:
            ends = np.append(ends, len(boolean_mask))

        # Calculate chronological durations
        durations = []
        for s, e in zip(starts, ends):
            if e > s:
                durations.append(times[e - 1] - times[s])
        return np.array(durations)

    l_durations = get_runs(l_ss)
    r_durations = get_runs(r_ss)

    return l_durations, r_durations


def plot_trial_histogram(filename, l_durs, r_durs):
    """Generates an aesthetic dual histogram plot for a single trial."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)

    title = f"Single-Support Phase Durations: {filename}"
    subtitle = "Calculated using raw vertical ground reaction forces (zero-filtering, no state estimation)."

    fig.text(0.06, 0.94, title, fontsize=13, fontweight="bold", color=NOTION_TEXT)
    fig.text(0.06, 0.90, subtitle, fontsize=9.5, color=NOTION_SUBTEXT)
    plt.subplots_adjust(top=0.84, bottom=0.15, left=0.08, right=0.92)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(NOTION_TEXT)

    ax.tick_params(axis="both", colors=NOTION_TEXT, length=4, labelsize=9)
    ax.grid(color=NOTION_GRID, linestyle="-", linewidth=1.0)
    ax.set_axisbelow(True)

    # Determine reasonable bin ranges
    all_durs = np.concatenate([l_durs, r_durs])
    if len(all_durs) > 0:
        bin_min = max(0.0, np.percentile(all_durs, 0.5))
        bin_max = np.percentile(all_durs, 99.5)
        # Avoid zero-width limits if data is extremely uniform
        if bin_max <= bin_min:
            bin_max = bin_min + 1.0
        bins = np.linspace(bin_min, bin_max, 40)
    else:
        bins = 40

    # Draw Histograms
    if len(l_durs) > 0:
        ax.hist(
            l_durs,
            bins=bins,
            color=COLOR_LEFT,
            edgecolor="#DC2626",
            alpha=0.6,
            label=f"Left Foot Single Support (N={len(l_durs)})",
            zorder=3,
        )
    if len(r_durs) > 0:
        ax.hist(
            r_durs,
            bins=bins,
            color=COLOR_RIGHT,
            edgecolor="#2563EB",
            alpha=0.6,
            label=f"Right Foot Single Support (N={len(r_durs)})",
            zorder=2,
        )

    # Descriptive statistics annotation
    stats_text = ""
    if len(l_durs) > 0:
        stats_text += f"Left  Mean: {np.mean(l_durs):.3f}s ± {np.std(l_durs):.3f}s\n"
    if len(r_durs) > 0:
        stats_text += f"Right Mean: {np.mean(r_durs):.3f}s ± {np.std(r_durs):.3f}s"

    if stats_text:
        ax.text(
            0.95,
            0.95,
            stats_text.strip(),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            color=NOTION_TEXT,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=NOTION_BG,
                edgecolor=NOTION_GRID,
                alpha=0.9,
            ),
        )

    ax.set_xlabel(
        "Single-Support Duration (seconds)",
        fontsize=10,
        fontweight="bold",
        color=NOTION_TEXT,
    )
    ax.set_ylabel(
        "Occurrences (Count)", fontsize=10, fontweight="bold", color=NOTION_TEXT
    )
    ax.legend(
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=8.5,
        loc="upper left",
    )

    print("  -> Showing plot. Close the window to proceed to the next trial.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect single-support phase distributions on raw Poggensee force-plate data."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_DIR,
        help="Path to directory containing processed parquet files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Vertical contact threshold force in Newtons",
    )
    args = parser.parse_args()

    target_dir = os.path.abspath(args.dir)
    if not os.path.exists(target_dir):
        print(f"Error: Target directory '{target_dir}' does not exist.")
        sys.exit(1)

    files = glob.glob(os.path.join(target_dir, "*.parquet"))
    # Filter out the wide-format precomputed summary file
    files = [f for f in files if "precomputed_poggensee" not in f]

    if not files:
        print(f"No trial parquet files found in: {target_dir}")
        sys.exit(0)

    print(f"Discovered {len(files)} files to evaluate.")

    for idx, file_path in enumerate(sorted(files)):
        filename = os.path.basename(file_path)
        print(f"\n[{idx + 1}/{len(files)}] Loading {filename}...")

        try:
            # Load only the timestamp and vertical ground forces to minimize memory overhead
            df = pd.read_parquet(
                file_path, columns=["time", "calcn_l_force_y", "calcn_r_force_y"]
            )
        except Exception as e:
            print(f"  Warning: Skipping {filename} because of read failure: {e}")
            continue

        l_durs, r_durs = calculate_single_support_durations(df, args.threshold)

        if len(l_durs) == 0 and len(r_durs) == 0:
            print(
                "  Warning: No active single-support intervals detected. Confirm force plate ranges."
            )
            continue

        plot_trial_histogram(filename, l_durs, r_durs)


if __name__ == "__main__":
    main()
