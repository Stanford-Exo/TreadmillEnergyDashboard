# File: src/analysis/plot_gait_time_domain.py

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DEFAULT_DIR = os.path.join(REPO_ROOT, "exported_pogensee")

# --- Aesthetic Styling (Muted Notion/Tufte Style) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "Arial", "sans-serif"]

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"

COLOR_LEFT = "#EF4444"  # Soft Red
COLOR_RIGHT = "#3B82F6"  # Soft Blue
COLOR_VO2 = "#10B981"  # Emerald
COLOR_VCO2 = "#8B5CF6"  # Purple
COLOR_RER = "#F59E0B"  # Amber
COLOR_WATTS = "#EF4444"  # Red for Power
COLOR_CLEAN_BG = "#D1FAE5"  # Pastel green for allowed gait patches
COLOR_INVALID_BG = "#FEE2E2"  # Pastel red for rejected chunks


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
        states[l_contact & ~r_contact] = 1  # LSS
        states[~l_contact & r_contact] = 2  # RSS
        states[l_contact & r_contact] = 3  # DS
        states[~l_contact & ~r_contact] = 0  # Flight

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
            if all(blocks[j]["valid"] for j in range(start_check, end_check)):
                b = blocks[i]
                clean_mask[b["start_idx"] : b["end_idx"]] = True
        return clean_mask


def main():
    parser = argparse.ArgumentParser(
        description="Plot Time-Domain Gait & Metabolic Heuristics."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to the parquet trial file"
    )
    parser.add_argument(
        "--threshold", type=float, default=30.0, help="Vertical force threshold (N)"
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    print(f"Loading {os.path.basename(file_path)}...")
    df = pd.read_parquet(file_path)

    time = df["time"].values
    l_force = df["calcn_l_force_y"].values
    r_force = df["calcn_r_force_y"].values

    # Extract Baseline
    qs_baseline = df["qs_baseline_w"].iloc[0] if "qs_baseline_w" in df.columns else 70.0

    # 1. Base GRF clean mask
    gait_filter = CleanGaitFilter(contact_threshold=args.threshold)
    grf_clean_mask = gait_filter.identify_clean_frames(df, time, l_force, r_force)

    vo2_col = next((c for c in df.columns if c.lower() == "vo2"), None)
    vco2_col = next((c for c in df.columns if c.lower() == "vco2"), None)
    has_metabolics = bool(vo2_col and vco2_col)

    frame_met_valid = np.ones(len(df), dtype=bool)
    vo2_raw, vco2_raw, rer, watts = None, None, None, None

    if has_metabolics:
        vo2_raw = df[vo2_col].values
        vco2_raw = df[vco2_col].values

        # Safe RER calculation
        with np.errstate(divide="ignore", invalid="ignore"):
            rer = np.where(vo2_raw > 0, vco2_raw / vo2_raw, 0)

        watts = ((3.941 * vo2_raw) + (1.106 * vco2_raw)) * 4.184 / 60.0

        # Heuristic 1 & 3: Frame-level metabolic validity
        presence_mask = (vo2_raw > 0) & (vco2_raw > 0) & (~np.isnan(vo2_raw))
        physio_mask = (rer >= 0.72) & (rer <= 1.05) & (vo2_raw > 200.0)
        frame_met_valid = presence_mask & physio_mask

    # AND the GRF and Metabolic masks
    combined_frame_mask = grf_clean_mask & frame_met_valid

    # --- CHUNKING & HEURISTICS 2, 4, 5 ---
    chunk_duration = 300.0  # 5 minutes
    final_valid_mask = np.zeros(len(df), dtype=bool)

    t_min, t_max = time.min(), time.max()
    chunk_edges = np.arange(t_min, t_max, chunk_duration)
    if chunk_edges[-1] < t_max:
        chunk_edges = np.append(chunk_edges, t_max)

    chunk_status = []

    for i in range(len(chunk_edges) - 1):
        c_start = chunk_edges[i]
        c_end = chunk_edges[i + 1]

        # Allow trailing chunks >= 3 mins (180s)
        if (c_end - c_start) < 180.0:
            chunk_status.append((c_start, c_end, False, "Too Short"))
            continue

        idx_mask = (time >= c_start) & (time < c_end)
        if not np.any(idx_mask):
            continue

        if has_metabolics:
            # Heuristic 4: Yield check (must have > 60% valid metabolic frames)
            chunk_presence = presence_mask[idx_mask]
            chunk_valid = frame_met_valid[idx_mask]

            # Yield = Valid Breaths / Total Breath frames in chunk
            yield_pct = (
                chunk_valid.sum() / len(chunk_valid) if len(chunk_valid) > 0 else 0
            )

            # Heuristic 5: Energy Floor
            chunk_watts = watts[idx_mask]
            valid_watts = chunk_watts[chunk_valid]

            net_watts = 0
            if len(valid_watts) > 0:
                net_watts = np.median(valid_watts) - qs_baseline

            is_valid_chunk = (yield_pct >= 0.60) and (net_watts >= 30.0)

            if is_valid_chunk:
                final_valid_mask[idx_mask] = combined_frame_mask[idx_mask]
                chunk_status.append(
                    (
                        c_start,
                        c_end,
                        True,
                        f"Yield: {yield_pct:.0%} | Net: {net_watts:.0f}W",
                    )
                )
            else:
                reason = (
                    f"Low Yield: {yield_pct:.0%}"
                    if yield_pct < 0.60
                    else f"Low Pwr: {net_watts:.0f}W"
                )
                chunk_status.append((c_start, c_end, False, reason))
        else:
            # If no metabolics, just rely on GRF mask
            final_valid_mask[idx_mask] = combined_frame_mask[idx_mask]
            chunk_status.append((c_start, c_end, True, "GRF Only"))

    # --- PLOTTING ---
    num_subplots = 5 if has_metabolics else 2
    fig, axs = plt.subplots(
        num_subplots,
        1,
        figsize=(15, 3 * num_subplots),
        sharex=True,
        facecolor=NOTION_BG,
    )
    if num_subplots == 1:
        axs = [axs]

    fig.suptitle(
        f"Heuristic Time-Domain Verification: {os.path.basename(file_path)}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    for ax in axs:
        ax.set_facecolor(NOTION_BG)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(color=NOTION_GRID, linestyle="-", linewidth=0.7)
        ax.set_axisbelow(True)

    # 1. Forces
    axs[0].plot(
        time, l_force, color=COLOR_LEFT, alpha=0.8, linewidth=1.2, label="Left Force"
    )
    axs[0].plot(
        time, r_force, color=COLOR_RIGHT, alpha=0.8, linewidth=1.2, label="Right Force"
    )
    axs[0].set_ylabel("Force (N)", fontweight="bold")
    axs[0].set_title(
        "1. Vertical Ground Reaction Forces",
        fontweight="bold",
        color=NOTION_SUBTEXT,
        loc="left",
    )
    axs[0].legend(loc="upper right")

    # 2. Contact States
    l_contact = l_force > args.threshold
    r_contact = r_force > args.threshold
    axs[1].step(
        time,
        l_contact.astype(int) + 0.05,
        where="post",
        color=COLOR_LEFT,
        label="Left Contact",
    )
    axs[1].step(
        time,
        r_contact.astype(int) - 0.05,
        where="post",
        color=COLOR_RIGHT,
        label="Right Contact",
    )
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(["Off", "On"])
    axs[1].set_ylabel("Contact", fontweight="bold")
    axs[1].set_title(
        "2. Binary Contact States", fontweight="bold", color=NOTION_SUBTEXT, loc="left"
    )

    # Metabolics Plots
    if has_metabolics:
        # Extract downsampled updates for visual clarity (like the diagnostic script)
        updates = np.concatenate(([True], np.abs(vo2_raw[1:] - vo2_raw[:-1]) > 1e-3))
        t_met = time[updates & presence_mask]
        v_vo2 = vo2_raw[updates & presence_mask]
        v_vco2 = vco2_raw[updates & presence_mask]
        v_rer = rer[updates & presence_mask]
        v_watts = watts[updates & presence_mask]

        # 3. Gas Exchange
        axs[2].plot(
            t_met, v_vo2, "o-", color=COLOR_VO2, markersize=3, alpha=0.7, label="VO2"
        )
        axs[2].plot(
            t_met, v_vco2, "s-", color=COLOR_VCO2, markersize=3, alpha=0.7, label="VCO2"
        )
        axs[2].set_ylabel("mL/min", fontweight="bold")
        axs[2].set_title(
            "3. Gas Exchange (Zeroes & NaNs dropped)",
            fontweight="bold",
            color=NOTION_SUBTEXT,
            loc="left",
        )
        axs[2].legend(loc="upper right")

        # 4. RER
        axs[3].plot(
            t_met, v_rer, "o-", color=COLOR_RER, markersize=3, alpha=0.8, label="RER"
        )
        axs[3].axhspan(
            0.72, 1.05, color="#10B981", alpha=0.1, label="Valid Range (0.72 - 1.05)"
        )
        axs[3].axhline(1.05, color="red", linestyle="--")
        axs[3].axhline(0.72, color="red", linestyle="--")
        axs[3].set_ylabel("RER Ratio", fontweight="bold")
        axs[3].set_title(
            "4. Respiratory Exchange Ratio (Spikes invalidate frames)",
            fontweight="bold",
            color=NOTION_SUBTEXT,
            loc="left",
        )
        axs[3].legend(loc="upper right")

        # 5. Watts
        axs[4].plot(
            t_met,
            v_watts,
            "o-",
            color=COLOR_WATTS,
            markersize=3,
            alpha=0.6,
            label="Gross Watts",
        )
        axs[4].axhline(
            qs_baseline,
            color="black",
            linestyle="--",
            label=f"QS Baseline ({qs_baseline:.1f}W)",
        )
        axs[4].axhline(
            qs_baseline + 30.0, color="red", linestyle=":", label="30W Floor Threshold"
        )
        axs[4].set_ylabel("Watts (W)", fontweight="bold")
        axs[4].set_title(
            "5. Biological Power (Sub-baseline chunks invalidated)",
            fontweight="bold",
            color=NOTION_SUBTEXT,
            loc="left",
        )
        axs[4].legend(loc="upper right")

    axs[-1].set_xlabel("Time (seconds)", fontweight="bold")

    # --- Shade and Segment Backgrounds ---
    for ax in axs:
        # Green shade for final valid frames
        ax.fill_between(
            time,
            0,
            1,
            where=final_valid_mask,
            color=COLOR_CLEAN_BG,
            alpha=0.5,
            transform=ax.get_xaxis_transform(),
        )

        # Vertical segment lines and labels
        for c_start, c_end, is_valid, label in chunk_status:
            ax.axvline(c_start, color="black", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.axvline(c_end, color="black", linestyle="--", alpha=0.7, linewidth=1.5)

            if not is_valid:
                # Shade invalid chunks light red
                ax.fill_betweenx(
                    ax.get_ylim(), c_start, c_end, color=COLOR_INVALID_BG, alpha=0.3
                )

            # Draw text on the top subplot only
            if ax == axs[0]:
                color = "green" if is_valid else "red"
                status_text = "VALID CHUNK" if is_valid else "INVALID CHUNK"
                ax.text(
                    c_start + (c_end - c_start) / 2,
                    ax.get_ylim()[1] * 0.95,
                    f"{status_text}\n{label}",
                    color=color,
                    ha="center",
                    va="top",
                    fontweight="bold",
                    fontsize=9,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.8,
                        edgecolor=color,
                        boxstyle="round,pad=0.2",
                    ),
                )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


if __name__ == "__main__":
    main()
