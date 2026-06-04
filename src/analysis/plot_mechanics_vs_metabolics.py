# File: src/analysis/plot_mechanics_vs_metabolics.py

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

from online_analyze.energy_analyzer import EnergyAnalyzer

# Path configuration
DEFAULT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))

# --- Tufte/Notion Aesthetics ---
try:
    import matplotlib.gridspec as gridspec
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

# Color map for each condition category
COLOR_MAP = {
    "NW": "#10B981",  # Emerald Green (Normal Walking)
    "ZT": "#4B5563",  # Charcoal/Slate (Zero Torque)
    "GA": "#2563EB",  # Royal Blue (Generic Assistance)
    "OP": "#EF4444",  # Crimson Red (Optimized Assistance)
}


# --- Biological Component Extraction & Gait Heuristics ---


def find_zero_crossings(y):
    return np.where(np.diff(np.sign(y)))[0]


def extract_biological_components(human_power, dt):
    """
    Splits human power into 'Likely Achilles' and 'Muscle Power' components.
    Uses double-array wrapping to handle phase boundaries cleanly.
    """
    N = len(human_power)
    doubled_power = np.concatenate([human_power, human_power])
    achilles_doubled = np.zeros_like(doubled_power)

    search_area = doubled_power[N // 2 : N + N // 2]
    if len(search_area) == 0:
        return human_power, np.zeros_like(human_power)

    peak_local = np.argmax(search_area)
    peak_idx = N // 2 + peak_local

    if doubled_power[peak_idx] <= 0:
        return human_power, np.zeros_like(human_power)

    crossings = find_zero_crossings(doubled_power)
    boundaries = [0] + [c + 1 for c in crossings] + [2 * N]

    pos_start, pos_end = -1, -1
    neg_start, neg_end = -1, -1

    for i in range(len(boundaries) - 1):
        if boundaries[i] <= peak_idx < boundaries[i + 1]:
            pos_start = boundaries[i]
            pos_end = boundaries[i + 1]
            if i > 0:
                neg_start = boundaries[i - 1]
                neg_end = boundaries[i]
            break

    if pos_start != -1 and neg_start != -1:
        pos_chunk = doubled_power[pos_start:pos_end]
        neg_chunk = doubled_power[neg_start:neg_end]

        e_pos = np.trapz(pos_chunk, dx=dt)
        e_neg = np.trapz(neg_chunk, dx=dt)

        if e_pos > 0 and e_neg < 0:
            achilles_energy = min(e_pos, abs(e_neg))
            scale_pos = achilles_energy / e_pos if e_pos != 0 else 0
            scale_neg = achilles_energy / abs(e_neg) if e_neg != 0 else 0

            achilles_doubled[pos_start:pos_end] = pos_chunk * scale_pos
            achilles_doubled[neg_start:neg_end] = neg_chunk * scale_neg

    achilles_power = achilles_doubled[:N] + achilles_doubled[N:]
    muscle_power = human_power - achilles_power
    return muscle_power, achilles_power


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
        states[~l_contact & ~r_contact] = 0  # Flight/None

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

    def identify_clean_frames_and_blocks(
        self, df, time, l_force, r_force, neighbor_consensus=2
    ):
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
                b = blocks[i]
                clean_mask[b["start_idx"] : b["end_idx"]] = True

        return clean_mask, blocks


def find_clean_cycles(blocks, times):
    cycles = []
    num_blocks = len(blocks)
    for i in range(num_blocks - 4):
        if blocks[i]["state"] == 1 and blocks[i]["valid"]:
            if (
                blocks[i + 1]["state"] == 3
                and blocks[i + 1]["valid"]
                and blocks[i + 2]["state"] == 2
                and blocks[i + 2]["valid"]
                and blocks[i + 3]["state"] == 3
                and blocks[i + 3]["valid"]
                and blocks[i + 4]["state"] == 1
                and blocks[i + 4]["valid"]
            ):
                start_idx = blocks[i]["start_idx"]
                end_idx = blocks[i + 4]["start_idx"]
                cycles.append(
                    {
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "start_time": times[start_idx],
                        "end_time": times[end_idx],
                        "duration": times[end_idx] - times[start_idx],
                    }
                )
    return cycles


def resample_gait_cycle(times, values, num_points=100):
    t_start, t_end = times[0], times[-1]
    if t_end == t_start:
        return np.zeros(num_points)
    norm_times = (times - t_start) / (t_end - t_start)
    target_times = np.linspace(0.0, 1.0, num_points)
    return np.interp(target_times, norm_times, values)


def parse_file_group_and_condition(filename):
    base = os.path.splitext(filename)[0]
    match = re.search(r"_(QS|GA\d|OP\d|ZT\d|NW\d)$", base, re.IGNORECASE)
    if not match:
        return None, None, None
    condition_raw = match.group(1).upper()
    group_prefix = base[: match.start()]
    category = re.sub(r"\d", "", condition_raw)
    return group_prefix, category, condition_raw


def compute_metabolics_time_series(df):
    """Calculates chronological metabolic rate (Watts) using the Brockway equation."""
    time = df["time"].values / 60.0  # Convert to minutes
    vo2_col = next((c for c in df.columns if c.lower() == "vo2"), None)
    vco2_col = next((c for c in df.columns if c.lower() == "vco2"), None)

    if not vo2_col or not vco2_col:
        return None, None

    vo2_raw = df[vo2_col].values
    vco2_raw = df[vco2_col].values

    valid_mask = (vo2_raw > 0) & (vco2_raw > 0) & (~np.isnan(vo2_raw))
    vo2_changed = np.concatenate(([True], np.abs(vo2_raw[1:] - vo2_raw[:-1]) > 1e-3))
    vco2_changed = np.concatenate(([True], np.abs(vco2_raw[1:] - vco2_raw[:-1]) > 1e-3))
    is_update_frame = vo2_changed | vco2_changed

    final_mask = valid_mask & is_update_frame
    if not np.any(final_mask):
        return None, None

    t_valid = time[final_mask]
    vo2 = vo2_raw[final_mask]
    vco2 = vco2_raw[final_mask]

    watts = ((3.941 * vo2) + (1.106 * vco2)) * 4.184 / 60.0
    return t_valid, watts


# --- Core Pipeline Execution per Subject/Day ---


def plot_subject_day_overlay(group_key, file_infos, qs_path, current_idx, total_groups):
    print(f"\n[{current_idx}/{total_groups}] Running analysis for group: {group_key}")

    # 1. Compute matched day's Quiet Standing (QS) baseline
    try:
        df_qs = pd.read_parquet(qs_path)
    except Exception as e:
        print(f"  -> Error reading QS baseline parquet: {e}. Skipping group.")
        return

    # Find absolute minimum time of baseline file to align clocks
    qs_absolute_start_min = df_qs["time"].min() / 60.0

    t_qs, w_qs = compute_metabolics_time_series(df_qs)
    if w_qs is None or len(w_qs) == 0:
        print(
            "  -> Error: Matched QS baseline file has no valid gas exchange data. Skipping."
        )
        return

    # Align baseline timeline to start at its own absolute zero
    t_qs_norm = t_qs - qs_absolute_start_min
    ss_qs_mask = t_qs_norm >= (t_qs_norm[-1] - 3.0)
    qs_baseline_w = (
        float(np.median(w_qs[ss_qs_mask]))
        if np.any(ss_qs_mask)
        else float(np.median(w_qs))
    )
    print(f"  -> Quiet Standing baseline: {qs_baseline_w:.1f} W")

    # 2. Setup Multi-Panel Figure Layout via a clean plt.figure call to avoid dual scales
    fig = plt.figure(figsize=(16, 7.5), facecolor=NOTION_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.3, 1.0])

    ax_time = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharey=ax_time)

    for ax in [ax_time, ax_bar]:
        ax.set_facecolor(NOTION_BG)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(NOTION_TEXT)
        ax.tick_params(axis="both", colors=NOTION_TEXT, length=4, labelsize=9)
        ax.grid(color=NOTION_GRID, linestyle="-", linewidth=1.0)
        ax.set_axisbelow(True)

    title = f"Net Energetics: {group_key.replace('_', ' ')}"
    subtitle = f"Group {current_idx} of {total_groups} | Left: Solid curve = Net Metabolics (Observed) | Dashed curve/Dots = Est. Muscle Cost (Mechanics)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96, color=NOTION_TEXT)
    fig.text(0.5, 0.91, subtitle, ha="center", fontsize=9.5, color=NOTION_SUBTEXT)

    legend_tracker = set()
    summary_data = []
    max_duration = 0.0

    # Sort conditions for stable stacking order
    sort_order = {"NW": 0, "ZT": 1, "GA": 2, "OP": 3}
    sorted_files = sorted(file_infos, key=lambda x: sort_order.get(x["category"], 4))

    for finfo in sorted_files:
        path = finfo["path"]
        category = finfo["category"]
        raw_label = finfo["raw_label"]
        color = COLOR_MAP.get(category, "#7C3AED")

        print(f"\n  -> Processing Condition: {raw_label}")

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"    Error reading active trial Parquet ({raw_label}): {e}")
            continue

        # Extract temporal references (Aligning metabolics and biomechanics to the SAME 0 scale)
        times = df["time"].values
        dts = np.diff(times)
        default_dt = np.median(dts) if len(dts) > 0 else 0.01

        # Absolute shared zero reference in minutes
        trial_absolute_start_min = df["time"].min() / 60.0

        # A. Process Net Metabolic Time-Series
        t_met, w_met = compute_metabolics_time_series(df)
        has_metabolics = w_met is not None and len(w_met) > 0
        net_met_steady_state = 0.0

        if has_metabolics:
            # Align metabolics to the absolute start time
            t_met_norm = t_met - trial_absolute_start_min
            max_duration = max(max_duration, t_met_norm[-1])

            # Net Metabolics = Gross - Quiet Standing baseline
            w_net_met = w_met - qs_baseline_w

            # Determine steady state value for summary table (final 3 min of aligned timeline)
            ss_met_mask = t_met_norm >= (t_met_norm[-1] - 3.0)
            net_met_steady_state = (
                float(np.median(w_net_met[ss_met_mask]))
                if np.any(ss_met_mask)
                else float(np.median(w_net_met))
            )

            # Plot rolling average
            window_size = min(35, len(w_net_met) // 5)
            if window_size > 2:
                rolling_net = (
                    pd.Series(w_net_met)
                    .rolling(window=window_size, center=True)
                    .median()
                )
                ax_time.plot(
                    t_met_norm,
                    rolling_net,
                    "-",
                    color=color,
                    linewidth=2.2,
                    label=category if category not in legend_tracker else None,
                )
                legend_tracker.add(category)

        # B. Process Chronological Mechanical Power (Stride-by-Stride)
        force_cols = [col for col in df.columns if col.endswith("_force_y")]
        contact_bodies = [col.replace("_force_y", "") for col in force_cols]
        left_body = next(
            (cb for cb in contact_bodies if cb.endswith("_l") or "left" in cb.lower()),
            contact_bodies[0],
        )
        right_body = next(
            (cb for cb in contact_bodies if cb.endswith("_r") or "right" in cb.lower()),
            contact_bodies[1],
        )

        l_force = df[f"{left_body}_force_y"].values
        r_force = df[f"{right_body}_force_y"].values

        # Run state machine filter to identify clean step intervals
        gait_filter = CleanGaitFilter(contact_threshold=30.0)
        clean_mask, blocks = gait_filter.identify_clean_frames_and_blocks(
            df, times, l_force, r_force, neighbor_consensus=2
        )
        clean_cycles = find_clean_cycles(blocks, times)

        if len(clean_cycles) > 0:
            # Reconstruct COM velocities using Kalman Filter
            active_fy = (l_force + r_force)[(l_force + r_force) > 50.0]
            calc_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

            # Exporter placeholders for exoskeleton inputs
            tauL = df["tauL"].values if "tauL" in df.columns else np.zeros(len(df))
            velL = df["velaL"].values if "velaL" in df.columns else np.zeros(len(df))
            tauR = df["tauR"].values if "tauR" in df.columns else np.zeros(len(df))
            velR = df["velaR"].values if "velaR" in df.columns else np.zeros(len(df))

            analyzer = EnergyAnalyzer(
                initial_mass=calc_mass, foot_roll_length=0.254, override_belt_speed=1.25
            )
            inst_human_l, inst_human_r = [], []

            # Populate filter sequentially (with incremental console progress logging)
            total_frames = len(df)
            print(f"    - Processing {total_frames:,} frames through EnergyAnalyzer...")
            for i in range(total_frames):
                if i % 10000 == 0:
                    print(
                        f"      [Progress] Frame {i:,} / {total_frames:,} ({i / total_frames:.0%})"
                    )

                forces = {
                    "left": np.array(
                        [
                            df[f"{left_body}_force_x"].values[i],
                            df[f"{left_body}_force_y"].values[i],
                            df[f"{left_body}_force_z"].values[i],
                        ]
                    ),
                    "right": np.array(
                        [
                            df[f"{right_body}_force_x"].values[i],
                            df[f"{right_body}_force_y"].values[i],
                            df[f"{right_body}_force_z"].values[i],
                        ]
                    ),
                }
                cops = {
                    "left": np.array(
                        [
                            df[f"{left_body}_cop_x"].values[i],
                            df[f"{left_body}_cop_y"].values[i],
                            df[f"{left_body}_cop_z"].values[i],
                        ]
                    ),
                    "right": np.array(
                        [
                            df[f"{right_body}_cop_x"].values[i],
                            df[f"{right_body}_cop_y"].values[i],
                            df[f"{right_body}_cop_z"].values[i],
                        ]
                    ),
                }
                dt = dts[i] if i < len(dts) else default_dt
                res = analyzer.update(
                    times[i],
                    forces,
                    cops,
                    dt,
                    exo_power_left=(tauL[i] * velL[i]),
                    exo_power_right=(tauR[i] * velR[i]),
                    is_clean=clean_mask[i],
                )
                inst_human_l.append(res["hum_left"])
                inst_human_r.append(res["hum_right"])

            # Compute power rates on a stride-by-stride basis
            stride_times = []
            stride_costs = []

            for cycle in clean_cycles:
                s, e = cycle["start_idx"], cycle["end_idx"]
                c_times = times[s:e]
                stride_dur = cycle["duration"]
                dt_stride = stride_dur / 100.0

                # Interpolate profiles
                l_hum_res = resample_gait_cycle(
                    c_times, np.array(inst_human_l[s:e]), num_points=100
                )
                r_hum_res = resample_gait_cycle(
                    c_times, np.array(inst_human_r[s:e]), num_points=100
                )

                # Segment and isolate Achilles storage
                l_mus, l_ach = extract_biological_components(l_hum_res, dt_stride)
                r_mus, r_ach = extract_biological_components(r_hum_res, dt_stride)

                # Muscle cost components
                j_pos_l = np.trapz(np.maximum(l_mus, 0), dx=dt_stride)
                j_neg_l = abs(np.trapz(np.minimum(l_mus, 0), dx=dt_stride))
                j_pos_r = np.trapz(np.maximum(r_mus, 0), dx=dt_stride)
                j_neg_r = abs(np.trapz(np.minimum(r_mus, 0), dx=dt_stride))

                # Achilles cost components
                j_pos_ach_l = np.trapz(np.maximum(l_ach, 0), dx=dt_stride)
                j_pos_ach_r = np.trapz(np.maximum(r_ach, 0), dx=dt_stride)

                # Custom metabolic prediction formula: 4.34 * Pos_mus + 1.66 * Neg_mus + 0.25 * Pos_ach
                est_stride_watts = (
                    4.34 * (j_pos_l + j_pos_r)
                    + 1.66 * (j_neg_l + j_neg_r)
                    + 0.25 * (j_pos_ach_l + j_pos_ach_r)
                ) / stride_dur

                # Map stride mid-time to the aligned absolute timescale (minutes)
                mid_time_norm = (
                    (cycle["start_time"] + cycle["end_time"]) / 2.0
                ) / 60.0 - trial_absolute_start_min
                stride_times.append(mid_time_norm)
                stride_costs.append(est_stride_watts)
                max_duration = max(max_duration, mid_time_norm)

            # Plot stride-by-stride mechanical power scatter dots
            ax_time.scatter(
                stride_times,
                stride_costs,
                color=color,
                s=15,
                alpha=0.45,
                edgecolors="none",
                zorder=3,
            )

            # Plot mechanical rolling average trendline (Dashed Line)
            if len(stride_times) > 5:
                sort_idx = np.argsort(stride_times)
                st_sorted = np.array(stride_times)[sort_idx]
                sc_sorted = np.array(stride_costs)[sort_idx]
                rolling_mech = (
                    pd.Series(sc_sorted)
                    .rolling(window=15, min_periods=1, center=True)
                    .median()
                )
                ax_time.plot(
                    st_sorted,
                    rolling_mech,
                    "--",
                    color=color,
                    linewidth=2.0,
                    label=None,
                    zorder=4,
                )

            # Compile steady-state summary averages for comparison panel (final 3 minutes of strides)
            if stride_times:
                stride_times_arr = np.array(stride_times)
                ss_mech_mask = stride_times_arr >= (stride_times_arr[-1] - 3.0)
                mech_ss_mean = (
                    float(np.median(np.array(stride_costs)[ss_mech_mask]))
                    if np.any(ss_mech_mask)
                    else float(np.median(stride_costs))
                )

                summary_data.append(
                    (
                        category,
                        raw_label,
                        net_met_steady_state if has_metabolics else None,
                        mech_ss_mean,
                    )
                )

    ax_time.set_ylabel(
        "Net Power (W)\n[Net metabolics & Est. biological muscle cost]",
        fontsize=11,
        fontweight="bold",
        color=NOTION_TEXT,
    )
    ax_time.set_xlabel(
        "Trial Time (Minutes from Start)",
        fontsize=11,
        fontweight="bold",
        color=NOTION_TEXT,
    )
    ax_time.set_xlim(-0.5, max_duration + 0.5)
    ax_time.legend(
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=9,
        loc="upper right",
    )

    # 3. Render sidebar bar chart
    sorted_summary = sorted(summary_data, key=lambda x: (sort_order.get(x[0], 5), x[1]))
    labels = [x[1] for x in sorted_summary]
    met_values = [x[2] if x[2] is not None else 0.0 for x in sorted_summary]
    mech_values = [x[3] for x in sorted_summary]

    x_positions = np.arange(len(labels))
    width = 0.35

    for idx, (cat, lbl, met_val, mech_val) in enumerate(sorted_summary):
        color = COLOR_MAP.get(cat, "#7C3AED")

        # Left Bar: Measured Net Metabolic Cost (Solid fill)
        if met_val is not None:
            ax_bar.bar(
                x_positions[idx] - width / 2,
                met_val,
                width,
                color=color,
                edgecolor=color,
                alpha=0.85,
                zorder=3,
            )

        # Right Bar: Estimated Mechanical Muscle Cost (Hatched fill)
        ax_bar.bar(
            x_positions[idx] + width / 2,
            mech_val,
            width,
            color="none",
            edgecolor=color,
            hatch="///",
            linewidth=1.5,
            zorder=3,
        )

    # Sidebar styling
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(
        labels, rotation=45, ha="right", fontsize=9, color=NOTION_TEXT
    )
    ax_bar.set_xlabel(
        "Trial Conditions", fontsize=11, fontweight="bold", color=NOTION_TEXT
    )
    ax_bar.set_title(
        "Steady-State Comparison\n(Final 3 minutes)",
        fontsize=11,
        fontweight="bold",
        color=NOTION_TEXT,
    )

    # Render mini-legend for bar types
    met_proxy = plt.Rectangle(
        (0, 0), 1, 1, color="#9CA3AF", alpha=0.85, label="Net Metabolics"
    )
    mech_proxy = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="none",
        edgecolor="#9CA3AF",
        hatch="///",
        linewidth=1.5,
        label="Mech Muscle Cost",
    )
    ax_bar.legend(
        handles=[met_proxy, mech_proxy],
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=8,
        loc="upper right",
    )

    # Match limits cleanly
    ax_time.set_ylim(-10.0, 310.0)

    print(
        "  -> Displaying overlay plot with sidebar bar chart. Close to load next group."
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Net Metabolic Cost vs. Estimated Stride-by-Stride Muscle Power."
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

    # Organize trials by Subject & Day, separating active validation trials from matched QS baselines
    groups = {}
    qs_baselines = {}

    for f in all_files:
        base = os.path.basename(f)
        if "ADAPT" in base:
            # Explicitly bypass any chronological adaptation trials
            continue

        group_key, category, raw_label = parse_file_group_and_condition(base)
        if group_key and category:
            if category == "QS":
                qs_baselines[group_key] = f
            else:
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(
                    {"path": f, "category": category, "raw_label": raw_label}
                )

    # Filter out any groups that lack a matching day-specific Quiet Standing baseline parquet file
    valid_groups = {}
    for gkey, file_infos in groups.items():
        if gkey in qs_baselines:
            valid_groups[gkey] = (file_infos, qs_baselines[gkey])

    if not valid_groups:
        print(
            f"No valid steady-state experimental groups matched with QS baseline trials in: {target_dir}"
        )
        return

    print(
        f"Discovered {len(valid_groups)} unique Subject/Day validation groups with matched baselines."
    )

    for idx, (group_key, (file_infos, qs_path)) in enumerate(
        sorted(valid_groups.items())
    ):
        plot_subject_day_overlay(
            group_key, file_infos, qs_path, idx + 1, len(valid_groups)
        )

    print("\nOverlay processing complete.")


if __name__ == "__main__":
    main()
