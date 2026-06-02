# File: src/analysis/precompute_poggensee.py

import argparse
import glob
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

from online_analyze.energy_analyzer import EnergyAnalyzer

POGGENSEE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))

# --- Numba JIT Optimization ---
# We push the heaviest per-stride mathematical extraction into LLVM compiled code
try:
    import numba as nb

    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    print(
        "Warning: 'numba' is not installed. Math operations will fall back to standard Python."
    )
    print("For maximum cluster speed, please run: pip install numba")


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
    """
    Splits human power into 'Likely Achilles' (balanced zero-net energy) and 'Muscle Power'.
    Heavily optimized for Numba JIT execution.
    """
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

    # Build boundaries array safely within nopython mode
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


# --- Main Analytics Engine ---


def clear_aggregations(analyzer):
    """Empties the stride profile buffers without resetting the Kalman Filter."""
    for k in analyzer.stride_profiles:
        analyzer.stride_profiles[k] = []
    for k in analyzer.stride_analyzer.metrics:
        analyzer.stride_analyzer.metrics[k] = []


def compile_window_row(df_win, analyzer, trial_name, window_start_s):
    profiles = analyzer.stride_profiles
    if len(profiles["ref_sys"]) == 0:
        return None

    # Calculate Mask Metabolics for this specific window
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

    # Extract mechanical metrics
    stats = analyzer.stride_analyzer.get_metrics_summary()
    mean_stride_dur = stats.get("stride_duration_mean", 1.0)
    dt_stride = mean_stride_dur / 100.0

    ref_exo_raw = np.array(profiles["ref_exo"])
    con_exo_raw = np.array(profiles["contra_exo"])

    com_x_raw = np.array(profiles["com_x"])
    com_y_raw = np.array(profiles["com_y"])
    com_z_raw = np.array(profiles["com_z"])

    com_x_mean_raw = np.mean(com_x_raw, axis=0)
    com_y_mean_raw = np.mean(com_y_raw, axis=0)
    com_z_mean_raw = np.mean(com_z_raw, axis=0)

    # Center the averaged trajectories around 0
    com_x_centered = com_x_mean_raw - np.mean(com_x_mean_raw)
    com_y_centered = com_y_mean_raw - np.mean(com_y_mean_raw)
    com_z_centered = com_z_mean_raw - np.mean(com_z_mean_raw)

    com_x_std = np.std(com_x_raw, axis=0)
    com_y_std = np.std(com_y_raw, axis=0)
    com_z_std = np.std(com_z_raw, axis=0)

    ref_mus_raw, ref_ach_raw = [], []
    for rh in profiles["ref_hum"]:
        rm, ra = extract_biological_components(rh, dt_stride)
        ref_mus_raw.append(rm)
        ref_ach_raw.append(ra)

    con_mus_raw, con_ach_raw = [], []
    for ch in profiles["contra_hum"]:
        cm, ca = extract_biological_components(ch, dt_stride)
        con_mus_raw.append(cm)
        con_ach_raw.append(ca)

    ref_mus_raw, ref_ach_raw = np.array(ref_mus_raw), np.array(ref_ach_raw)
    con_mus_raw, con_ach_raw = np.array(con_mus_raw), np.array(con_ach_raw)

    ref_exo_mean, ref_exo_std = np.mean(ref_exo_raw, axis=0), np.std(
        ref_exo_raw, axis=0
    )
    con_exo_mean, con_exo_std = np.mean(con_exo_raw, axis=0), np.std(
        con_exo_raw, axis=0
    )

    ref_mus_mean, ref_mus_std = np.mean(ref_mus_raw, axis=0), np.std(
        ref_mus_raw, axis=0
    )
    ref_ach_mean, ref_ach_std = np.mean(ref_ach_raw, axis=0), np.std(
        ref_ach_raw, axis=0
    )

    con_mus_mean, con_mus_std = np.mean(con_mus_raw, axis=0), np.std(
        con_mus_raw, axis=0
    )
    con_ach_mean, con_ach_std = np.mean(con_ach_raw, axis=0), np.std(
        con_ach_raw, axis=0
    )

    # Calculate summary metabolic scalars
    j_pos_ref = np.trapezoid(np.maximum(ref_mus_mean, 0), dx=dt_stride)
    j_neg_ref = abs(np.trapezoid(np.minimum(ref_mus_mean, 0), dx=dt_stride))
    j_pos_con = np.trapezoid(np.maximum(con_mus_mean, 0), dx=dt_stride)
    j_neg_con = abs(np.trapezoid(np.minimum(con_mus_mean, 0), dx=dt_stride))
    est_mech_watts = (
        (4 * j_pos_ref + 1 * j_neg_ref) + (4 * j_pos_con + 1 * j_neg_con)
    ) / mean_stride_dur

    ref_hum_mean = np.mean(np.array(profiles["ref_hum"]), axis=0)
    con_hum_mean = np.mean(np.array(profiles["contra_hum"]), axis=0)
    j_pos_ref_raw = np.trapezoid(np.maximum(ref_hum_mean, 0), dx=dt_stride)
    j_neg_ref_raw = abs(np.trapezoid(np.minimum(ref_hum_mean, 0), dx=dt_stride))
    j_pos_con_raw = np.trapezoid(np.maximum(con_hum_mean, 0), dx=dt_stride)
    j_neg_con_raw = abs(np.trapezoid(np.minimum(con_hum_mean, 0), dx=dt_stride))
    est_mech_watts_no_achilles = (
        (4 * j_pos_ref_raw + 1 * j_neg_ref_raw)
        + (4 * j_pos_con_raw + 1 * j_neg_con_raw)
    ) / mean_stride_dur

    exo_power_net = (
        np.trapezoid(ref_exo_mean, dx=dt_stride) + np.trapezoid(con_exo_mean, dx=dt_stride)
    ) / mean_stride_dur

    # Calculate stride-by-stride variability on integrated power outputs
    num_strides = len(profiles["ref_sys"])
    stride_mech_powers = []
    stride_mech_powers_no_achilles = []
    stride_exo_powers = []

    for s_idx in range(num_strides):
        r_mus = ref_mus_raw[s_idx]
        c_mus = con_mus_raw[s_idx]
        j_pos_r = np.trapezoid(np.maximum(r_mus, 0), dx=dt_stride)
        j_neg_r = abs(np.trapezoid(np.minimum(r_mus, 0), dx=dt_stride))
        j_pos_c = np.trapezoid(np.maximum(c_mus, 0), dx=dt_stride)
        j_neg_c = abs(np.trapezoid(np.minimum(c_mus, 0), dx=dt_stride))
        p_mech = (
            (4 * j_pos_r + 1 * j_neg_r) + (4 * j_pos_c + 1 * j_neg_c)
        ) / mean_stride_dur
        stride_mech_powers.append(p_mech)

        r_hum = np.array(profiles["ref_hum"][s_idx])
        c_hum = np.array(profiles["contra_hum"][s_idx])
        j_pos_r_raw = np.trapezoid(np.maximum(r_hum, 0), dx=dt_stride)
        j_neg_r_raw = abs(np.trapezoid(np.minimum(r_hum, 0), dx=dt_stride))
        j_pos_c_raw = np.trapezoid(np.maximum(c_hum, 0), dx=dt_stride)
        j_neg_c_raw = abs(np.trapezoid(np.minimum(c_hum, 0), dx=dt_stride))
        p_mech_no_ach = (
            (4 * j_pos_r_raw + 1 * j_neg_r_raw) + (4 * j_pos_c_raw + 1 * j_neg_c_raw)
        ) / mean_stride_dur
        stride_mech_powers_no_achilles.append(p_mech_no_ach)

        r_exo = ref_exo_raw[s_idx]
        c_exo = con_exo_raw[s_idx]
        p_exo = (
            np.trapezoid(r_exo, dx=dt_stride) + np.trapezoid(c_exo, dx=dt_stride)
        ) / mean_stride_dur
        stride_exo_powers.append(p_exo)

    mech_power_std = float(np.std(stride_mech_powers)) if stride_mech_powers else 0.0
    mech_power_no_ach_std = (
        float(np.std(stride_mech_powers_no_achilles))
        if stride_mech_powers_no_achilles
        else 0.0
    )
    text_exo_power_std = float(np.std(stride_exo_powers)) if stride_exo_powers else 0.0

    # Build Wide Row
    row = {
        "trial_name": trial_name,
        "window_start_s": float(window_start_s),
        "mean_stride_duration_s": mean_stride_dur,
        "mean_duty_factor": stats.get("duty_factor_mean", 0.6),
        "duty_factor_std": stats.get("duty_factor_std", 0.015),
        "num_valid_strides": len(profiles["ref_sys"]),
        "bio_watts": bio_watts,
        "standing_baseline_w": standing_baseline,
        "net_bio_cost_w": net_bio_watts,
        "mechanical_power": est_mech_watts,
        "mechanical_power_std": mech_power_std,
        "mechanical_power_no_achilles": est_mech_watts_no_achilles,
        "mechanical_power_no_achilles_std": mech_power_no_ach_std,
        "exo_power": exo_power_net,
        "exo_power_std": text_exo_power_std,
    }

    # Inject all 1D bucket columns
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


def process_trial(
    df, left_body, right_body, trial_name, burn_in_s, window_s, min_window_s
):
    times = df["time"].values
    dts = np.diff(times)
    default_dt = np.median(dts) if len(dts) > 0 else 0.01

    t_start = times[0]
    burn_in_thresh = t_start + burn_in_s
    current_window_start = burn_in_thresh
    current_window_end = current_window_start + window_s

    f_total_y = df[f"{left_body}_force_y"].values + df[f"{right_body}_force_y"].values
    active_fy = f_total_y[f_total_y > 50.0]
    calc_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

    analyzer = EnergyAnalyzer(
        initial_mass=calc_mass, foot_roll_length=0.254, override_belt_speed=1.25
    )

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

    rows = []

    for i in range(len(df)):
        t = times[i]
        dt = dts[i] if i < len(dts) else default_dt
        forces = {"left": left_forces[i], "right": right_forces[i]}
        cops = {"left": left_cops[i], "right": right_cops[i]}

        analyzer.update(
            t,
            forces,
            cops,
            dt,
            exo_power_left=(tauL_vals[i] * velL_vals[i]),
            exo_power_right=(tauR_vals[i] * velR_vals[i]),
        )

        if t < burn_in_thresh:
            clear_aggregations(analyzer)
            continue

        if t >= current_window_end:
            df_win = df[
                (df["time"] >= current_window_start) & (df["time"] < current_window_end)
            ]
            row = compile_window_row(df_win, analyzer, trial_name, current_window_start)
            if row:
                rows.append(row)

            clear_aggregations(analyzer)
            current_window_start = current_window_end
            current_window_end = current_window_start + window_s

    if (times[-1] - current_window_start) >= min_window_s:
        df_win = df[(df["time"] >= current_window_start) & (df["time"] <= times[-1])]
        row = compile_window_row(df_win, analyzer, trial_name, current_window_start)
        if row:
            rows.append(row)

    return rows


def process_file_worker(file_path, burn_in, window, min_window):
    """Worker function for running trial analysis isolated in a separate process."""
    filename = os.path.basename(file_path)
    trial_name = os.path.splitext(filename)[0]

    try:
        # Fast metadata-only pass
        df_time = pd.read_parquet(file_path, columns=["time"])
        if not df_time.empty:
            times = df_time["time"].values
            total_duration = times[-1] - times[0]
            min_required = burn_in + min_window
            if total_duration < min_required:
                return {
                    "status": "skipped",
                    "trial": trial_name,
                    "reason": f"Duration ({total_duration:.1f}s) too short.",
                }
    except Exception as e:
        return {
            "status": "error",
            "trial": trial_name,
            "reason": f"Metadata read error: {e}",
        }

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        return {"status": "error", "trial": trial_name, "reason": f"Read error: {e}"}

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
            df, left_body, right_body, trial_name, burn_in, window, min_window
        )
    except Exception as e:
        return {
            "status": "error",
            "trial": trial_name,
            "reason": f"Calculation error: {e}",
        }

    if trial_rows:
        return {"status": "success", "trial": trial_name, "rows": trial_rows}
    else:
        return {
            "status": "skipped",
            "trial": trial_name,
            "reason": "No valid windows generated.",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Precompute continuous mechanics/metabolics in parallel chunks."
    )
    parser.add_argument("--dir", type=str, default=POGGENSEE_DIR)
    parser.add_argument(
        "--burn-in", type=float, default=30.0, help="Seconds to discard at trial start"
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of CPU workers to parallelize execution",
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

    # --- ADD THIS WARM-UP BLOCK ---
    if USE_NUMBA:
        print("Warming up Numba JIT compiler to prevent multi-core cache collisions...")
        # Pass a dummy array to force LLVM to compile the functions in the main process
        dummy_power = np.random.randn(100).astype(np.float64)
        _ = extract_biological_components(dummy_power, 0.01)
    # ------------------------------

    print(f"\nDispatching {len(files_to_process)} trial(s) to {args.workers} workers...")

    # Parallel Execution Pool
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_file_worker, fp, args.burn_in, args.window, args.min_window
            ): fp
            for fp in files_to_process
        }

        for future in as_completed(futures):
            res = future.result()
            t_name = res["trial"]

            if res["status"] == "success":
                new_rows_df = pd.DataFrame(res["rows"])
                existing_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
                existing_df.to_parquet(parquet_path, index=False)
                print(f"  -> Saved {len(res['rows'])} windows for {t_name}")
            else:
                reason = res["reason"]
                prefix = "Error" if res["status"] == "error" else "Skipped"
                print(f"  -> {prefix} {t_name}: {reason}")

                try:
                    with open(skipped_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{t_name}\n")
                    skipped_trials.add(t_name)
                except Exception as write_err:
                    pass

    print("\nPrecompute process completed.")


if __name__ == "__main__":
    main()
