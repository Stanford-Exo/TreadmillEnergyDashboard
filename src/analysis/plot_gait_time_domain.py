# File: src/analysis/plot_gait_time_domain_metabolics.py

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

# --- Aesthetic Styling (Muted Notion/Tufte Style) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "Arial", "sans-serif"]

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"

COLOR_LEFT = "#EF4444"  # Soft Red for Left Foot
COLOR_RIGHT = "#3B82F6"  # Soft Blue for Right Foot
COLOR_VO2 = "#10B981"  # Emerald for VO2
COLOR_VCO2 = "#8B5CF6"  # Purple for VCO2
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

        states = np.zeros(len(df), dtype=int)
        states[l_contact & ~r_contact] = 1  # Left Single Support (LSS)
        states[~l_contact & r_contact] = 2  # Right Single Support (RSS)
        states[l_contact & r_contact] = 3  # Double Support (DS)
        states[~l_contact & ~r_contact] = 0  # None/Flight

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
        for b in blocks:
            state = b["state"]
            dur = b["duration"]
            if state == 0:
                b["valid"] = False
            elif state in [1, 2]:
                if not (self.lss_bounds[0] <= dur <= self.lss_bounds[1]):
                    b["valid"] = False
            elif state == 3:
                if not (self.ds_bounds[0] <= dur <= self.ds_bounds[1]):
                    b["valid"] = False

        for i in range(2, num_blocks - 2):
            if not blocks[i]["valid"]:
                continue

            current_state = blocks[i]["state"]
            next_state = blocks[i + 1]["state"]
            prev_state = blocks[i - 1]["state"]

            if current_state in [1, 2]:
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

        for i in range(num_blocks):
            start_check = max(0, i - neighbor_consensus)
            end_check = min(num_blocks, i + neighbor_consensus + 1)

            is_neighborhood_clean = all(
                blocks[j]["valid"] for j in range(start_check, end_check)
            )

            if is_neighborhood_clean:
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
        description="Plot Ground Reaction Forces along with temporal VO2/VCO2 breath dynamics."
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
        default=None,
        help="Start time in seconds (defaults to full trial start)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (defaults to full trial duration)",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        print(f"Error: Could not find parquet file at {file_path}")
        sys.exit(1)

    print(f"Loading {os.path.basename(file_path)}...")
    df = pd.read_parquet(file_path)

    time_all = df["time"].values
    l_force_all = df["calcn_l_force_y"].values
    r_force_all = df["calcn_r_force_y"].values

    # Detect respiratory columns
    vo2_col = next((c for c in df.columns if c.lower() == "vo2"), None)
    vco2_col = next((c for c in df.columns if c.lower() == "vco2"), None)

    # Set temporal limits
    t_min = time_all.min()
    t_max = time_all.max()

    start_time = args.start if args.start is not None else t_min
    duration = args.duration if args.duration is not None else (t_max - start_time)
    end_time = min(start_time + duration, t_max)

    print(
        f"Window configured from {start_time:.1f}s to {end_time:.1f}s (Total duration: {end_time - start_time:.1f}s)"
    )

    # Run clean gait segmentation
    gait_filter = CleanGaitFilter(contact_threshold=args.threshold)
    clean_mask_all = gait_filter.identify_clean_frames(
        df, time_all, l_force_all, r_force_all, neighbor_consensus=2
    )

    # Filter arrays to the targeted window
    window_mask = (df["time"] >= start_time) & (df["time"] <= end_time)
    df_win = df[window_mask]
    mask_win = clean_mask_all[window_mask]

    if df_win.empty:
        print(
            f"Error: No data remains in the time window {start_time}s to {end_time}s."
        )
        sys.exit(1)

    time = df_win["time"].values
    l_force = df_win["calcn_l_force_y"].values
    r_force = df_win["calcn_r_force_y"].values

    l_contact = l_force > args.threshold
    r_contact = r_force > args.threshold
    l_ss = l_contact & ~r_contact
    r_ss = r_contact & ~l_contact

    # Setup subplot rows based on metabolic column availability
    has_metabolics = (vo2_col is not None) and (df_win[vo2_col].dropna().sum() > 0)
    num_subplots = 4 if has_metabolics else 3

    fig, axs = plt.subplots(
        num_subplots,
        1,
        figsize=(12, 2.5 * num_subplots),
        sharex=True,
        facecolor=NOTION_BG,
    )

    if num_subplots == 1:
        axs = [axs]

    fig.suptitle(
        f"Gait Signals & Gas Exchange Dynamics: {os.path.basename(file_path)}",
        fontsize=13,
        fontweight="bold",
        color=NOTION_TEXT,
        y=0.97,
    )

    for ax in axs:
        ax.set_facecolor(NOTION_BG)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(NOTION_TEXT)
        ax.tick_params(axis="both", colors=NOTION_TEXT, length=4, labelsize=9)
        ax.grid(color=NOTION_GRID, linestyle="-", linewidth=0.7)
        ax.set_axisbelow(True)

    # --- Plot 1: Vertical Forces ---
    axs[0].plot(
        time,
        l_force,
        color=COLOR_LEFT,
        alpha=0.7,
        linewidth=1.2,
        label="Left Force (Fy)",
    )
    axs[0].plot(
        time,
        r_force,
        color=COLOR_RIGHT,
        alpha=0.7,
        linewidth=1.2,
        label="Right Force (Fy)",
    )
    axs[0].axhline(
        args.threshold,
        color=NOTION_TEXT,
        linestyle="--",
        linewidth=1.0,
        label=f"Threshold ({args.threshold} N)",
    )
    axs[0].set_ylabel("Force (N)", fontsize=9, fontweight="bold", color=NOTION_TEXT)
    axs[0].set_title(
        "1. Vertical Ground Reaction Forces",
        fontsize=10,
        fontweight="bold",
        color=NOTION_SUBTEXT,
        loc="left",
    )
    axs[0].legend(
        loc="upper right",
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=8,
    )

    # --- Plot 2: Binary Contact States ---
    axs[1].step(
        time,
        l_contact.astype(int) + 0.05,
        where="post",
        color=COLOR_LEFT,
        linewidth=1.2,
        label="Left Contact",
    )
    axs[1].step(
        time,
        r_contact.astype(int) - 0.05,
        where="post",
        color=COLOR_RIGHT,
        linewidth=1.2,
        label="Right Contact",
    )
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(["Off", "On"])
    axs[1].set_ylim(-0.2, 1.2)
    axs[1].set_ylabel("Contact", fontsize=9, fontweight="bold", color=NOTION_TEXT)
    axs[1].set_title(
        "2. Binary Contact States (Force > Threshold)",
        fontsize=10,
        fontweight="bold",
        color=NOTION_SUBTEXT,
        loc="left",
    )

    # --- Plot 3: Single-Support Active ---
    axs[2].step(
        time,
        l_ss.astype(int) + 0.05,
        where="post",
        color=COLOR_LEFT,
        linewidth=1.2,
        label="Left Single Support",
    )
    axs[2].step(
        time,
        r_ss.astype(int) - 0.05,
        where="post",
        color=COLOR_RIGHT,
        linewidth=1.2,
        label="Right Single Support",
    )
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(["Off", "On"])
    axs[2].set_ylim(-0.2, 1.2)
    axs[2].set_ylabel(
        "Single Support", fontsize=9, fontweight="bold", color=NOTION_TEXT
    )
    axs[2].set_title(
        "3. Single-Support Windows",
        fontsize=10,
        fontweight="bold",
        color=NOTION_SUBTEXT,
        loc="left",
    )

    # --- Plot 4: VO2 & VCO2 Profiles (If Present) ---
    if has_metabolics:
        vo2_vals = df_win[vo2_col].values
        vco2_vals = df_win[vco2_col].values if vco2_col else None

        # Mask zero and missing values typical of gas exchange transitions
        valid_indices = (vo2_vals > 0) & (~np.isnan(vo2_vals))
        t_met = time[valid_indices]
        vo2_clean = vo2_vals[valid_indices]

        axs[3].plot(
            t_met,
            vo2_clean,
            color=COLOR_VO2,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="VO2 (O2 Consumption)",
        )

        if vco2_vals is not None:
            vco2_clean = vco2_vals[valid_indices]
            axs[3].plot(
                t_met,
                vco2_clean,
                color=COLOR_VCO2,
                marker="s",
                markersize=3,
                linewidth=1.2,
                label="VCO2 (CO2 Production)",
            )

        axs[3].set_ylabel(
            "Rate (mL/min)", fontsize=9, fontweight="bold", color=NOTION_TEXT
        )
        axs[3].set_title(
            "4. Respiratory Gas Exchange (Indirect Calorimetry)",
            fontsize=10,
            fontweight="bold",
            color=NOTION_SUBTEXT,
            loc="left",
        )
        axs[3].legend(
            loc="upper left",
            frameon=True,
            facecolor=NOTION_BG,
            edgecolor=NOTION_GRID,
            fontsize=8,
        )

        # Print baseline averages inside terminal
        mean_vo2 = np.mean(vo2_clean)
        mean_vco2 = np.mean(vco2_clean) if vco2_vals is not None else 0.85 * mean_vo2
        cal_per_min = 3.941 * mean_vo2 + 1.106 * mean_vco2
        watts = cal_per_min * 4.184 / 60.0

        print("\nTrial Metabolic Summary Statistics:")
        print(f"  - Mean VO2  : {mean_vo2:.1f} mL/min")
        if vco2_col:
            print(f"  - Mean VCO2 : {mean_vco2:.1f} mL/min")
        print(f"  - Calculated Gross Biological Cost: {watts:.1f} W")

    axs[-1].set_xlabel(
        "Time (seconds)", fontsize=10, fontweight="bold", color=NOTION_TEXT
    )

    # Shading the background with the Clean Segment Mask
    for ax in axs:
        ax.fill_between(
            time,
            0,
            1,
            where=mask_win,
            color=COLOR_CLEAN_BG,
            alpha=0.3,
            transform=ax.get_xaxis_transform(),
            zorder=1,
        )

    plt.xlim(start_time, end_time)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
