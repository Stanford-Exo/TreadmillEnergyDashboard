# File: src/analysis/precompute_poggensee.py

import argparse
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

from online_analyze.energy_analyzer import EnergyAnalyzer

POGGENSEE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))

# --- Numba JIT Optimization ---
try:
    import numba as nb

    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    print(
        "Warning: 'numba' is not installed. Math operations will fall back to standard Python."
    )


def njit_opt(func):
    if USE_NUMBA:
        return nb.njit(cache=True)(func)
    return func


@njit_opt
def trapz_1d(y, dx):
    """JIT-compiled trapezoidal integration."""
    n = len(y)
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(n - 1):
        s += y[i] + y[i + 1]
    return s * dx * 0.5


@njit_opt
def find_zero_crossings(y):
    """JIT-compiled fast zero-crossing detector."""
    n = len(y)
    crossings = np.zeros(n, dtype=np.int64)
    count = 0
    for i in range(n - 1):
        if (y[i + 1] > 0 and y[i] <= 0) or (y[i + 1] < 0 and y[i] >= 0):
            crossings[count] = i
            count += 1
    return crossings[:count]


@njit_opt
def extract_biological_components(human_power, dt):
    """Splits human power into 'Likely Achilles' and 'Muscle Power' components."""
    human_power = np.asarray(human_power, dtype=np.float64)
    N = len(human_power)

    doubled_power = np.zeros(2 * N, dtype=np.float64)
    doubled_power[:N] = human_power
    doubled_power[N:] = human_power

    achilles_doubled = np.zeros_like(doubled_power)

    search_area = doubled_power[N // 2 : N + N // 2]
    if len(search_area) == 0:
        return human_power, np.zeros_like(human_power)

    peak_local = np.argmax(search_area)
    peak_idx = N // 2 + peak_local

    if doubled_power[peak_idx] <= 0:
        return human_power, np.zeros_like(human_power)

    crossings = find_zero_crossings(doubled_power)

    boundaries = np.zeros(len(crossings) + 2, dtype=np.int64)
    boundaries[0] = 0
    for i in range(len(crossings)):
        boundaries[i + 1] = crossings[i] + 1
    boundaries[-1] = 2 * N

    pos_start = -1
    pos_end = -1
    neg_start = -1
    neg_end = -1

    for i in range(len(boundaries) - 1):
        if boundaries[i] <= peak_idx and peak_idx < boundaries[i + 1]:
            pos_start = boundaries[i]
            pos_end = boundaries[i + 1]
            if i > 0:
                neg_start = boundaries[i - 1]
                neg_end = boundaries[i]
            break

    if pos_start != -1 and neg_start != -1:
        pos_chunk = doubled_power[pos_start:pos_end]
        neg_chunk = doubled_power[neg_start:neg_end]

        e_pos = trapz_1d(pos_chunk, dt)
        e_neg = trapz_1d(neg_chunk, dt)

        if e_pos > 0 and e_neg < 0:
            achilles_energy = min(e_pos, abs(e_neg))
            scale_pos = achilles_energy / e_pos if e_pos != 0 else 0.0
            scale_neg = achilles_energy / abs(e_neg) if e_neg != 0 else 0.0

            for i in range(pos_start, pos_end):
                achilles_doubled[i] = doubled_power[i] * scale_pos
            for i in range(neg_start, neg_end):
                achilles_doubled[i] = doubled_power[i] * scale_neg

    achilles_power = achilles_doubled[:N] + achilles_doubled[N:]
    muscle_power = human_power - achilles_power
    return muscle_power, achilles_power


# --- State-Machine Clean Gait Heuristics ---


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
        states[~l_contact & ~r_contact] = 0  # None

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

                stride_dur = times[end_idx] - times[start_idx]
                stance_dur = (
                    blocks[i]["duration"]
                    + blocks[i + 1]["duration"]
                    + blocks[i + 3]["duration"]
                )
                duty_factor = stance_dur / stride_dur if stride_dur > 0 else 0.60

                cycles.append(
                    {
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "start_time": times[start_idx],
                        "end_time": times[end_idx],
                        "stride_dur": stride_dur,
                        "duty_factor": duty_factor,
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


def compile_clean_window_row(
    df_win, inst_records, window_cycles, times, trial_name, window_start_s
):
    vo2_col = next((c for c in df_win.columns if c.lower() == "vo2"), None)
    vco2_col = next((c for c in df_win.columns if c.lower() == "vco2"), None)

    if not vo2_col:
        return None

    vo2_mean = df_win[vo2_col].replace(0, np.nan).dropna().mean()
    if pd.isna(vo2_mean) or vo2_mean <= 0:
        return None

    vco2_mean = (
        df_win[vco2_col].replace(0, np.nan).dropna().mean()
        if vco2_col
        else 0.85 * vo2_mean
    )
    cal_per_min = 3.941 * vo2_mean + 1.106 * vco2_mean
    bio_watts = cal_per_min * 4.184 / 60.0

    standing_baseline = (
        df_win["qs_baseline_w"].iloc[0] if "qs_baseline_w" in df_win.columns else 70.0
    )
    net_bio_watts = bio_watts - standing_baseline

    cycles_durs = [c["stride_dur"] for c in window_cycles]
    cycles_dfs = [c["duty_factor"] for c in window_cycles]

    mean_stride_dur = float(np.mean(cycles_durs))
    mean_duty_factor = float(np.mean(cycles_dfs))
    duty_factor_std = float(np.std(cycles_dfs)) if len(cycles_dfs) > 1 else 0.0

    dt_stride = mean_stride_dur / 100.0

    profiles = {
        k: []
        for k in [
            "ref_exo",
            "contra_exo",
            "ref_hum",
            "contra_hum",
            "ref_sys",
            "contra_sys",
            "com_x",
            "com_y",
            "com_z",
        ]
    }

    for c in window_cycles:
        s, e = c["start_idx"], c["end_idx"]
        c_times = times[s:e]

        for k in profiles.keys():
            c_vals = np.array([inst_records[idx][k] for idx in range(s, e)])
            resampled = resample_gait_cycle(c_times, c_vals, num_points=100)
            profiles[k].append(resampled)

    profiles_arr = {k: np.array(v) for k, v in profiles.items()}

    ref_exo_mean = np.mean(profiles_arr["ref_exo"], axis=0)
    ref_exo_std = np.std(profiles_arr["ref_exo"], axis=0)
    con_exo_mean = np.mean(profiles_arr["contra_exo"], axis=0)
    con_exo_std = np.std(profiles_arr["contra_exo"], axis=0)

    com_x_mean = np.mean(profiles_arr["com_x"], axis=0)
    com_y_mean = np.mean(profiles_arr["com_y"], axis=0)
    com_z_mean = np.mean(profiles_arr["com_z"], axis=0)
    com_x_std = np.std(profiles_arr["com_x"], axis=0)
    com_y_std = np.std(profiles_arr["com_y"], axis=0)
    com_z_std = np.std(profiles_arr["com_z"], axis=0)

    com_x_centered = com_x_mean - np.mean(com_x_mean)
    com_y_centered = com_y_mean - np.mean(com_y_mean)
    com_z_centered = com_z_mean - np.mean(com_z_mean)

    ref_mus_raw, ref_ach_raw = [], []
    for rh in profiles_arr["ref_hum"]:
        rm, ra = extract_biological_components(rh, dt_stride)
        ref_mus_raw.append(rm)
        ref_ach_raw.append(ra)

    con_mus_raw, con_ach_raw = [], []
    for ch in profiles_arr["contra_hum"]:
        cm, ca = extract_biological_components(ch, dt_stride)
        con_mus_raw.append(cm)
        con_ach_raw.append(ca)

    ref_mus_raw, ref_ach_raw = np.array(ref_mus_raw), np.array(ref_ach_raw)
    con_mus_raw, con_ach_raw = np.array(con_mus_raw), np.array(con_ach_raw)

    ref_mus_mean, ref_mus_std = (
        np.mean(ref_mus_raw, axis=0),
        np.std(ref_mus_raw, axis=0),
    )
    ref_ach_mean, ref_ach_std = (
        np.mean(ref_ach_raw, axis=0),
        np.std(ref_ach_raw, axis=0),
    )
    con_mus_mean, con_mus_std = (
        np.mean(con_mus_raw, axis=0),
        np.std(con_mus_raw, axis=0),
    )
    con_ach_mean, con_ach_std = (
        np.mean(con_ach_raw, axis=0),
        np.std(con_ach_raw, axis=0),
    )

    j_pos_ref = np.trapz(np.maximum(ref_mus_mean, 0), dx=dt_stride)
    j_neg_ref = abs(np.trapz(np.minimum(ref_mus_mean, 0), dx=dt_stride))
    j_pos_con = np.trapz(np.maximum(con_mus_mean, 0), dx=dt_stride)
    j_neg_con = abs(np.trapz(np.minimum(con_mus_mean, 0), dx=dt_stride))
    est_mech_watts = (
        (4 * j_pos_ref + 1 * j_neg_ref) + (4 * j_pos_con + 1 * j_neg_con)
    ) / mean_stride_dur

    ref_hum_mean = np.mean(profiles_arr["ref_hum"], axis=0)
    con_hum_mean = np.mean(profiles_arr["contra_hum"], axis=0)
    j_pos_ref_raw = np.trapz(np.maximum(ref_hum_mean, 0), dx=dt_stride)
    j_neg_ref_raw = abs(np.trapz(np.minimum(ref_hum_mean, 0), dx=dt_stride))
    j_pos_con_raw = np.trapz(np.maximum(con_hum_mean, 0), dx=dt_stride)
    j_neg_con_raw = abs(np.trapz(np.minimum(con_hum_mean, 0), dx=dt_stride))
    est_mech_watts_no_achilles = (
        (4 * j_pos_ref_raw + 1 * j_neg_ref_raw)
        + (4 * j_pos_con_raw + 1 * j_neg_con_raw)
    ) / mean_stride_dur

    # Integrate Achilles components
    j_pos_ref_ach = np.trapz(np.maximum(ref_ach_mean, 0), dx=dt_stride)
    j_neg_ref_ach = abs(np.trapz(np.minimum(ref_ach_mean, 0), dx=dt_stride))
    j_pos_con_ach = np.trapz(np.maximum(con_ach_mean, 0), dx=dt_stride)
    j_neg_con_ach = abs(np.trapz(np.minimum(con_ach_mean, 0), dx=dt_stride))

    exo_power_net = (
        np.trapz(ref_exo_mean, dx=dt_stride) + np.trapz(con_exo_mean, dx=dt_stride)
    ) / mean_stride_dur

    stride_mech_powers = []
    stride_mech_powers_no_achilles = []
    stride_exo_powers = []

    for s_idx in range(len(window_cycles)):
        r_mus = ref_mus_raw[s_idx]
        c_mus = con_mus_raw[s_idx]
        j_pos_r = np.trapz(np.maximum(r_mus, 0), dx=dt_stride)
        j_neg_r = abs(np.trapz(np.minimum(r_mus, 0), dx=dt_stride))
        j_pos_c = np.trapz(np.maximum(c_mus, 0), dx=dt_stride)
        j_neg_c = abs(np.trapz(np.minimum(c_mus, 0), dx=dt_stride))
        p_mech = (
            (4 * j_pos_r + 1 * j_neg_r) + (4 * j_pos_c + 1 * j_neg_c)
        ) / mean_stride_dur
        stride_mech_powers.append(p_mech)

        r_hum = profiles_arr["ref_hum"][s_idx]
        c_hum = profiles_arr["contra_hum"][s_idx]
        j_pos_r_raw = np.trapz(np.maximum(r_hum, 0), dx=dt_stride)
        j_neg_r_raw = abs(np.trapz(np.minimum(r_hum, 0), dx=dt_stride))
        j_pos_c_raw = np.trapz(np.maximum(c_hum, 0), dx=dt_stride)
        j_neg_c_raw = abs(np.trapz(np.minimum(c_hum, 0), dx=dt_stride))
        p_mech_no_ach = (
            (4 * j_pos_r_raw + 1 * j_neg_r_raw) + (4 * j_pos_c_raw + 1 * j_neg_c_raw)
        ) / mean_stride_dur
        stride_mech_powers_no_achilles.append(p_mech_no_ach)

        r_exo = profiles_arr["ref_exo"][s_idx]
        c_exo = profiles_arr["contra_exo"][s_idx]
        p_exo = (
            np.trapz(r_exo, dx=dt_stride) + np.trapz(c_exo, dx=dt_stride)
        ) / mean_stride_dur
        stride_exo_powers.append(p_exo)

    mech_power_std = float(np.std(stride_mech_powers)) if stride_mech_powers else 0.0
    mech_power_no_ach_std = (
        float(np.std(stride_mech_powers_no_achilles))
        if stride_mech_powers_no_achilles
        else 0.0
    )
    text_exo_power_std = float(np.std(stride_exo_powers)) if stride_exo_powers else 0.0

    row = {
        "trial_name": trial_name,
        "window_start_s": float(window_start_s),
        "mean_stride_duration_s": mean_stride_dur,
        "mean_duty_factor": mean_duty_factor,
        "duty_factor_std": duty_factor_std,
        "num_valid_strides": len(window_cycles),
        "bio_watts": bio_watts,
        "standing_baseline_w": standing_baseline,
        "net_bio_cost_w": net_bio_watts,
        "mechanical_power": est_mech_watts,
        "mechanical_power_std": mech_power_std,
        "mechanical_power_no_achilles": est_mech_watts_no_achilles,
        "mechanical_power_no_achilles_std": mech_power_no_ach_std,
        "exo_power": exo_power_net,
        "exo_power_std": text_exo_power_std,
        "ref_mus_pos_power_w": j_pos_ref / mean_stride_dur,
        "ref_mus_neg_power_w": j_neg_ref / mean_stride_dur,
        "con_mus_pos_power_w": j_pos_con / mean_stride_dur,
        "con_mus_neg_power_w": j_neg_con / mean_stride_dur,
        "ref_ach_pos_power_w": j_pos_ref_ach / mean_stride_dur,
        "ref_ach_neg_power_w": j_neg_ref_ach / mean_stride_dur,
        "con_ach_pos_power_w": j_pos_con_ach / mean_stride_dur,
        "con_ach_neg_power_w": j_neg_con_ach / mean_stride_dur,
    }

    for i in range(100):
        row[f"ref_exo_w_{i:02d}"] = ref_exo_mean[i]
        row[f"ref_ach_w_{i:02d}"] = ref_ach_mean[i]
        row[f"ref_mus_w_{i:02d}"] = ref_mus_mean[i]
        row[f"con_exo_w_{i:02d}"] = con_exo_mean[i]
        row[f"con_ach_w_{i:02d}"] = con_ach_mean[i]
        row[f"con_mus_w_{i:02d}"] = con_mus_mean[i]

        row[f"ref_exo_std_{i:02d}"] = ref_exo_std[i]
        row[f"ref_ach_std_{i:02d}"] = ref_ach_std[i]
        row[f"ref_mus_std_{i:02d}"] = ref_mus_std[i]
        row[f"con_exo_std_{i:02d}"] = con_exo_std[i]
        row[f"con_ach_std_{i:02d}"] = con_ach_std[i]
        row[f"con_mus_std_{i:02d}"] = con_mus_std[i]

        row[f"com_x_w_{i:02d}"] = com_x_centered[i]
        row[f"com_y_w_{i:02d}"] = com_y_centered[i]
        row[f"com_z_w_{i:02d}"] = com_z_centered[i]

        row[f"com_x_std_{i:02d}"] = com_x_std[i]
        row[f"com_y_std_{i:02d}"] = com_y_std[i]
        row[f"com_z_std_{i:02d}"] = com_z_std[i]

    return row


def process_trial(df, left_body, right_body, trial_name, window_s, min_window_s):
    times = df["time"].values
    dts = np.diff(times)
    default_dt = np.median(dts) if len(dts) > 0 else 0.01

    left_forces = df[
        [f"{left_body}_force_x", f"{left_body}_force_y", f"{left_body}_force_z"]
    ].values
    right_forces = df[
        [f"{right_body}_force_x", f"{right_body}_force_y", f"{right_body}_force_z"]
    ].values
    left_cops = df[
        [f"{left_body}_cop_x", f"{left_body}_cop_y", f"{left_body}_cop_z"]
    ].values
    right_cops = df[
        [f"{right_body}_cop_x", f"{right_body}_cop_y", f"{right_body}_cop_z"]
    ].values

    tauL_vals = df["tauL"].values if "tauL" in df.columns else np.zeros(len(df))
    velL_vals = df["velaL"].values if "velaL" in df.columns else np.zeros(len(df))
    tauR_vals = df["tauR"].values if "tauR" in df.columns else np.zeros(len(df))
    velR_vals = df["velaR"].values if "velaR" in df.columns else np.zeros(len(df))

    # --- 1. Compute State-Machine Clean Gait Heuristics ---
    gait_filter = CleanGaitFilter(contact_threshold=30.0)
    clean_mask, blocks = gait_filter.identify_clean_frames_and_blocks(
        df, times, left_forces[:, 1], right_forces[:, 1], neighbor_consensus=2
    )

    clean_indices = np.where(clean_mask)[0]
    if len(clean_indices) == 0:
        print(f"  -> Skipped {trial_name}: No clean gait patches detected.")
        return []

    first_clean_idx = clean_indices[0]
    last_clean_idx = clean_indices[-1]

    t_first = times[first_clean_idx]
    t_last = times[last_clean_idx]
    true_duration = t_last - t_first

    print(
        f"  -> True Clean Duration: {true_duration:.1f}s (From {t_first:.1f}s to {t_last:.1f}s)"
    )

    if true_duration < min_window_s:
        print(
            f"  -> Skipped {trial_name}: True duration ({true_duration:.1f}s) is shorter than min required ({min_window_s}s)."
        )
        return []

    # --- 2. Extract Whole Clean Gait Cycles ---
    clean_cycles = find_clean_cycles(blocks, times)
    if not clean_cycles:
        print(
            f"  -> Skipped {trial_name}: No complete clean gait cycles could be extracted."
        )
        return []

    # --- 3. Run COM filter from frame 0 for convergence ---
    f_total_y = left_forces[:, 1] + right_forces[:, 1]
    active_fy = f_total_y[f_total_y > 50.0]
    calc_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

    analyzer = EnergyAnalyzer(
        initial_mass=calc_mass, foot_roll_length=0.254, override_belt_speed=1.25
    )

    inst_records = []

    for i in range(last_clean_idx + 1):
        t = times[i]
        dt = dts[i] if i < len(dts) else default_dt
        forces = {"left": left_forces[i], "right": right_forces[i]}
        cops = {"left": left_cops[i], "right": right_cops[i]}

        res = analyzer.update(
            t,
            forces,
            cops,
            dt,
            exo_power_left=(tauL_vals[i] * velL_vals[i]),
            exo_power_right=(tauR_vals[i] * velR_vals[i]),
            is_clean=clean_mask[i],
        )

        inst_records.append(
            {
                "com_x": res["com_pos"][0],
                "com_y": res["com_pos"][1],
                "com_z": res["com_pos"][2],
                "ref_exo": tauL_vals[i] * velL_vals[i],
                "contra_exo": tauR_vals[i] * velR_vals[i],
                "ref_hum": res["hum_left"],
                "contra_hum": res["hum_right"],
                "ref_sys": res["sys_left"],
                "contra_sys": res["sys_right"],
            }
        )

    # --- 4. Process Sequential 5-Minute Windows ---
    rows = []
    current_window_start = t_first

    while current_window_start + window_s <= t_last:
        current_window_end = current_window_start + window_s

        # Rigidly enforce strict containment: cycles must start and end within the window
        window_cycles = [
            c
            for c in clean_cycles
            if c["start_time"] >= current_window_start
            and c["end_time"] <= current_window_end
        ]

        print(
            f"    - Window [{current_window_start:.1f}s - {current_window_end:.1f}s]: Included {len(window_cycles)} clean cycles."
        )

        if window_cycles:
            df_win = df[
                (df["time"] >= current_window_start) & (df["time"] < current_window_end)
            ]
            row = compile_clean_window_row(
                df_win,
                inst_records,
                window_cycles,
                times,
                trial_name,
                current_window_start,
            )
            if row:
                rows.append(row)

        current_window_start += window_s

    # Final trailing window check
    residual_duration = t_last - current_window_start
    if residual_duration >= min_window_s:
        current_window_end = t_last
        window_cycles = [
            c
            for c in clean_cycles
            if c["start_time"] >= current_window_start
            and c["end_time"] <= current_window_end
        ]

        print(
            f"    - Window-Tail [{current_window_start:.1f}s - {current_window_end:.1f}s]: Included {len(window_cycles)} clean cycles."
        )

        if window_cycles:
            df_win = df[
                (df["time"] >= current_window_start)
                & (df["time"] <= current_window_end)
            ]
            row = compile_clean_window_row(
                df_win,
                inst_records,
                window_cycles,
                times,
                trial_name,
                current_window_start,
            )
            if row:
                rows.append(row)

    return rows


def process_file_sequential(file_path, window, min_window):
    """Processes a single trial sequentially."""
    filename = os.path.basename(file_path)
    trial_name = os.path.splitext(filename)[0]

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  -> Error reading {filename}: {e}")
        return None

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

    try:
        trial_rows = process_trial(
            df, left_body, right_body, trial_name, window, min_window
        )
    except Exception as e:
        print(f"  -> Calculation error on {filename}: {e}")
        traceback.print_exc()
        return None

    return trial_rows


def main():
    parser = argparse.ArgumentParser(
        description="Precompute continuous mechanics/metabolics sequentially."
    )
    parser.add_argument("--dir", type=str, default=POGGENSEE_DIR)
    parser.add_argument(
        "--window",
        type=float,
        default=300.0,
        help="Window size in seconds (default: 300s)",
    )
    parser.add_argument(
        "--min-window",
        type=float,
        default=180.0,
        help="Minimum valid tail window length in seconds",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(POGGENSEE_DIR, "precomputed_poggensee.parquet"),
    )
    args = parser.parse_args()

    parquet_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    processed_trials = set()
    skipped_log_path = os.path.join(os.path.dirname(parquet_path), "skipped_trials.txt")
    skipped_trials = set()

    if os.path.exists(skipped_log_path):
        try:
            with open(skipped_log_path, "r", encoding="utf-8") as f:
                skipped_trials = {line.strip() for line in f if line.strip()}
            print(f"Loaded {len(skipped_trials)} skipped/invalid trials from cache.")
        except Exception as e:
            print(f"Warning: Could not read skipped log: {e}")

    existing_df = pd.DataFrame()
    if os.path.exists(parquet_path):
        try:
            existing_df = pd.read_parquet(parquet_path)
            if not existing_df.empty and "trial_name" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    processed_trials.add(row["trial_name"])
            print(
                f"Found existing Parquet with {len(processed_trials)} completed trials. Resuming..."
            )
        except Exception as e:
            print(f"Failed to read existing Parquet {parquet_path}: {e}")
            sys.exit(1)
    else:
        print(f"Creating new precomputed Parquet at: {parquet_path}")

    files = glob.glob(os.path.join(os.path.abspath(args.dir), "*.parquet"))
    files = [f for f in files if "precomputed_poggensee" not in f]

    files_to_process = []
    for f in sorted(files):
        t_name = os.path.splitext(os.path.basename(f))[0]
        if t_name not in processed_trials and t_name not in skipped_trials:
            files_to_process.append(f)

    if not files_to_process:
        print("No new valid .parquet trials found to process.")
        return

    if USE_NUMBA:
        print("Warming up Numba JIT compiler...")
        dummy_power = np.random.randn(100).astype(np.float64)
        _ = extract_biological_components(dummy_power, 0.01)

    print(f"\nProcessing {len(files_to_process)} trial(s) sequentially...")

    for f_idx, file_path in enumerate(files_to_process):
        t_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n[{f_idx + 1}/{len(files_to_process)}] Processing: {t_name}")

        trial_rows = process_file_sequential(file_path, args.window, args.min_window)

        if trial_rows:
            new_rows_df = pd.DataFrame(trial_rows)
            existing_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
            existing_df.to_parquet(parquet_path, index=False)
            print(f"  -> Successfully precomputed and saved {len(trial_rows)} windows.")
        else:
            print(f"  -> Skipping trial: {t_name} (Failed to yield valid segments).")
            try:
                with open(skipped_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{t_name}\n")
                skipped_trials.add(t_name)
            except Exception:
                pass

    print("\nPrecompute process completed.")


if __name__ == "__main__":
    main()
