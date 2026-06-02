# File: src/analysis/plot_gait_clean_heuristics.py

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Setup paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DEFAULT_DIR = os.path.join(REPO_ROOT, "exported_pogensee")
DEFAULT_FILE = "Continued_Optimization_1_adaptation_Day2_ADAPT1.parquet"

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required for time-domain plotting.")
    sys.exit(1)

# --- Aesthetic Styling (Muted Notion Style) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "Arial", "sans-serif"]

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"

COLOR_LEFT = "#EF4444"  # Soft Red for Left Foot
COLOR_RIGHT = "#3B82F6"  # Soft Blue for Right Foot
COLOR_CLEAN_BG = "#D1FAE5"  # Pastel green for allowed gait patches


class CleanGaitFilter:
    def __init__(
        self, contact_threshold=30.0, lss_bounds=(0.15, 0.60), ds_bounds=(0.03, 0.35)
    ):
        self.contact_threshold = contact_threshold
        self.lss_bounds = lss_bounds
        self.ds_bounds = ds_bounds

    def extract_state_blocks(self, df, time, l_force, r_force):
        l_contact = l_force > self.contact_threshold
        r_contact = r_force > self.contact_threshold

        # Map binary combinations to state integers
        states = np.zeros(len(df), dtype=int)
        states[l_contact & ~r_contact] = 1  # LSS
        states[~l_contact & r_contact] = 2  # RSS
        states[l_contact & r_contact] = 3  # DS
        states[~l_contact & ~r_contact] = 0  # None

        # Find state transitions
        diff = np.diff(states)
        change_indices = np.where(diff != 0)[0] + 1
        starts = np.insert(change_indices, 0, 0)
        ends = np.append(change_indices, len(df))

        blocks = []
        for s, e in zip(starts, ends):
            blocks.append(
                {
                    "state": states[s],
                    "start_idx": s,
                    "end_idx": e,
                    "start_time": time[s],
                    "end_time": time[e - 1],
                    "duration": time[e - 1] - time[s],
                    "valid": True,
                }
            )
        return blocks

    def filter_blocks(self, blocks):
        num_blocks = len(blocks)

        # 1. Individual Duration Checks
        for b in blocks:
            state = b["state"]
            dur = b["duration"]
            if state == 0:  # Flight phase is invalid for walking
                b["valid"] = False
            elif state in [1, 2]:  # Single Support
                if not (self.lss_bounds[0] <= dur <= self.lss_bounds[1]):
                    b["valid"] = False
            elif state == 3:  # Double Support
                if not (self.ds_bounds[0] <= dur <= self.ds_bounds[1]):
                    b["valid"] = False

        # 2. Sequence Checker (Strict Loop Pattern)
        # Expected loop: 1 -> 3 -> 2 -> 3 -> 1 ...
        for i in range(2, num_blocks - 2):
            if not blocks[i]["valid"]:
                continue

            current_state = blocks[i]["state"]
            next_state = blocks[i + 1]["state"]
            prev_state = blocks[i - 1]["state"]

            if current_state == 1:
                if prev_state != 3 or next_state != 3:
                    blocks[i]["valid"] = False
            elif current_state == 2:
                if prev_state != 3 or next_state != 3:
                    blocks[i]["valid"] = False
            elif current_state == 3:
                connected_states = {prev_state, next_state}
                if connected_states != {1, 2}:
                    blocks[i]["valid"] = False

    def identify_clean_frames(self, df, time, l_force, r_force, neighbor_consensus=2):
        blocks = self.extract_state_blocks(df, time, l_force, r_force)
        self.filter_blocks(blocks)

        clean_mask = np.zeros(len(df), dtype=bool)
        num_blocks = len(blocks)

        # Apply neighbor consensus
        for i in range(num_blocks):
            start_check = max(0, i - neighbor_consensus)
            end_check = min(num_blocks, i + neighbor_consensus + 1)

            is_neighborhood_clean = all(
                blocks[j]["valid"] for j in range(start_check, end_check)
            )

            if is_neighborhood_clean:
                # Track steady state rhythmicity (duration standard deviation)
                ds_durs = [
                    blocks[j]["duration"]
                    for j in range(start_check, end_check)
                    if blocks[j]["state"] == 3
                ]
                ss_durs = [
                    blocks[j]["duration"]
                    for j in range(start_check, end_check)
                    if blocks[j]["state"] in [1, 2]
                ]

                if len(ds_durs) > 1 and np.std(ds_durs) > 0.03:
                    continue
                if len(ss_durs) > 1 and np.std(ss_durs) > 0.04:
                    continue

                b = blocks[i]
                clean_mask[b["start_idx"] : b["end_idx"]] = True

        return clean_mask


def main():
    parser = argparse.ArgumentParser(
        description="Plot Raw GRF with background highlight indicators based on the clean gait heuristic."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=os.path.join(DEFAULT_DIR, DEFAULT_FILE),
        help="Path to the specific parquet trial file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Vertical force contact threshold in Newtons (default: 30.0 N)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=1300.0,
        help="Start time in seconds for the plotted window",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration of the plotted window in seconds (default: 30s)",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        print(f"Error: Could not find parquet file at {file_path}")
        sys.exit(1)

    print(f"Loading {os.path.basename(file_path)}...")
    df = pd.read_parquet(file_path)

    # We evaluate the heuristic on the whole file to prevent window boundary artifacts
    time_all = df["time"].values
    l_force_all = df["calcn_l_force_y"].values
    r_force_all = df["calcn_r_force_y"].values

    # Run heuristic with widened thresholds
    gait_filter = CleanGaitFilter(
        contact_threshold=args.threshold,
        lss_bounds=(0.15, 0.60),  # Widen single-support
        ds_bounds=(0.03, 0.35),  # Widen double-support
    )
    clean_mask_all = gait_filter.identify_clean_frames(
        df, time_all, l_force_all, r_force_all, neighbor_consensus=2
    )

    # Filter data to the selected zoom time window for plotting
    df_win = df[(df["time"] >= args.start) & (df["time"] <= args.start + args.duration)]
    mask_win = clean_mask_all[
        (df["time"] >= args.start) & (df["time"] <= args.start + args.duration)
    ]

    if df_win.empty:
        print(
            f"Error: No data found in time window {args.start}s to {args.start + args.duration}s."
        )
        sys.exit(1)

    time = df_win["time"].values
    l_force = df_win["calcn_l_force_y"].values
    r_force = df_win["calcn_r_force_y"].values

    l_contact = l_force > args.threshold
    r_contact = r_force > args.threshold

    l_ss = l_contact & ~r_contact
    r_ss = r_contact & ~l_contact

    # Create diagnostic subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 8.5), sharex=True, facecolor=NOTION_BG
    )
    fig.suptitle(
        f"Gait Signal Heuristic Verification: {os.path.basename(file_path)}",
        fontsize=14,
        fontweight="bold",
        color=NOTION_TEXT,
        y=0.97,
    )

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(NOTION_BG)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(NOTION_TEXT)
        ax.tick_params(axis="both", colors=NOTION_TEXT, length=4, labelsize=9)
        ax.grid(color=NOTION_GRID, linestyle="-", linewidth=0.7)
        ax.set_axisbelow(True)

    # --- Plot 1: Raw Vertical Forces ---
    ax1.plot(
        time,
        l_force,
        color=COLOR_LEFT,
        alpha=0.8,
        linewidth=1.5,
        label="Left Force (Fy)",
    )
    ax1.plot(
        time,
        r_force,
        color=COLOR_RIGHT,
        alpha=0.8,
        linewidth=1.5,
        label="Right Force (Fy)",
    )
    ax1.axhline(
        args.threshold,
        color=NOTION_TEXT,
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold ({args.threshold} N)",
    )
    ax1.set_ylabel(
        "Vertical Force (N)", fontsize=9, fontweight="bold", color=NOTION_TEXT
    )
    ax1.set_title(
        "1. Raw Vertical Ground Reaction Forces",
        fontsize=10,
        fontweight="bold",
        color=NOTION_SUBTEXT,
        loc="left",
    )
    ax1.legend(
        loc="upper right",
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=8,
    )

    # --- Plot 2: Binary Contact States ---
    ax2.step(
        time,
        l_contact.astype(int) + 0.05,
        where="post",
        color=COLOR_LEFT,
        linewidth=1.5,
        label="Left Contact",
    )
    ax2.step(
        time,
        r_contact.astype(int) - 0.05,
        where="post",
        color=COLOR_RIGHT,
        linewidth=1.5,
        label="Right Contact",
    )
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Off", "On"])
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_ylabel("Contact State", fontsize=9, fontweight="bold", color=NOTION_TEXT)
    ax2.set_title(
        "2. Binary Contact States (Force > Threshold)",
        fontsize=10,
        fontweight="bold",
        color=NOTION_SUBTEXT,
        loc="left",
    )
    ax2.legend(
        loc="upper right",
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=8,
    )

    # --- Plot 3: Resulting Single-Support Windows ---
    ax3.step(
        time,
        l_ss.astype(int) + 0.05,
        where="post",
        color=COLOR_LEFT,
        linewidth=1.5,
        label="Left Single Support",
    )
    ax3.step(
        time,
        r_ss.astype(int) - 0.05,
        where="post",
        color=COLOR_RIGHT,
        linewidth=1.5,
        label="Right Single Support",
    )
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Off", "On"])
    ax3.set_ylim(-0.2, 1.2)
    ax3.set_ylabel(
        "Single-Support Active", fontsize=9, fontweight="bold", color=NOTION_TEXT
    )
    ax3.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold", color=NOTION_TEXT)
    ax3.set_title(
        "3. Resulting Single-Support Windows (Contact L & Not Contact R)",
        fontsize=10,
        fontweight="bold",
        color=NOTION_SUBTEXT,
        loc="left",
    )
    ax3.legend(
        loc="upper right",
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=8,
    )

    # --- Highlight Heuristic Decisions in the Background of All Subplots ---
    for ax in [ax1, ax2, ax3]:
        # Shade allowed clean patches in soft green
        ax.fill_between(
            time,
            0,
            1,
            where=mask_win,
            color=COLOR_CLEAN_BG,
            alpha=0.35,
            transform=ax.get_xaxis_transform(),
            zorder=1,
            label="Allowed (Clean Patch)",
        )

    plt.xlim(args.start, args.start + args.duration)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
