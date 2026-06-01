# File: src/analysis/precompute_poggensee.py

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

from online_analyze.energy_analyzer import EnergyAnalyzer

POGGENSEE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))


def find_zero_crossings(y):
    return np.where(np.diff(np.sign(y)))[0]


def extract_biological_components(human_power, dt):
    """
    Splits human power into 'Likely Achilles' (balanced zero-net energy) and 'Muscle Power'.
    """
    N = len(human_power)
    doubled_power = np.concatenate([human_power, human_power])
    achilles_doubled = np.zeros_like(doubled_power)
    
    search_area = doubled_power[N//2 : N + N//2]
    if len(search_area) == 0:
        return human_power, np.zeros_like(human_power)
        
    peak_local = np.argmax(search_area)
    peak_idx = N//2 + peak_local
    
    if doubled_power[peak_idx] <= 0:
        return human_power, np.zeros_like(human_power)
        
    crossings = find_zero_crossings(doubled_power)
    boundaries = [0] + [c + 1 for c in crossings] + [2 * N]
    
    pos_start, pos_end = -1, -1
    neg_start, neg_end = -1, -1
    
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= peak_idx < boundaries[i+1]:
            pos_start = boundaries[i]
            pos_end = boundaries[i+1]
            if i > 0:
                neg_start = boundaries[i-1]
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


def clear_aggregations(analyzer):
    """Empties the stride profile buffers without resetting the Kalman Filter."""
    for k in analyzer.stride_profiles:
        analyzer.stride_profiles[k] = []
    for k in analyzer.stride_analyzer.metrics:
        analyzer.stride_analyzer.metrics[k] = []


def compile_window_row(df_win, analyzer, trial_name, window_start_s):
    """
    Computes metabolics and mechanical aggregates for the completed 5-minute window
    currently held in the analyzer's buffers, returning a Wide-Format dictionary.
    """
    profiles = analyzer.stride_profiles
    if len(profiles['ref_sys']) == 0:
        return None

    # 1. Calculate Mask Metabolics for this specific window
    vo2_col = next((c for c in df_win.columns if c.lower() == 'vo2'), None)
    vco2_col = next((c for c in df_win.columns if c.lower() == 'vco2'), None)
    
    if not vo2_col:
        return None
        
    vo2_mean = df_win[vo2_col].replace(0, np.nan).dropna().mean()
    if pd.isna(vo2_mean) or vo2_mean <= 0:
        return None
        
    vco2_mean = df_win[vco2_col].replace(0, np.nan).dropna().mean() if vco2_col else 0.85 * vo2_mean
    cal_per_min = 3.941 * vo2_mean + 1.106 * vco2_mean
    bio_watts = cal_per_min * 4.184 / 60.0
    
    standing_baseline = df_win["qs_baseline_w"].iloc[0] if "qs_baseline_w" in df_win.columns else 70.0
    net_bio_watts = bio_watts - standing_baseline

    # 2. Extract mechanical metrics
    stats = analyzer.stride_analyzer.get_metrics_summary()
    mean_stride_dur = stats.get('stride_duration_mean', 1.0)
    dt_stride = mean_stride_dur / 100.0

    ref_exo_raw = np.array(profiles['ref_exo'])
    con_exo_raw = np.array(profiles['contra_exo'])

    com_x_raw = np.array(profiles['com_x'])
    com_y_raw = np.array(profiles['com_y'])
    com_z_raw = np.array(profiles['com_z'])

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
    for rh in profiles['ref_hum']:
        rm, ra = extract_biological_components(rh, dt_stride)
        ref_mus_raw.append(rm)
        ref_ach_raw.append(ra)
        
    con_mus_raw, con_ach_raw = [], []
    for ch in profiles['contra_hum']:
        cm, ca = extract_biological_components(ch, dt_stride)
        con_mus_raw.append(cm)
        con_ach_raw.append(ca)

    ref_mus_raw, ref_ach_raw = np.array(ref_mus_raw), np.array(ref_ach_raw)
    con_mus_raw, con_ach_raw = np.array(con_mus_raw), np.array(con_ach_raw)

    ref_exo_mean, ref_exo_std = np.mean(ref_exo_raw, axis=0), np.std(ref_exo_raw, axis=0)
    con_exo_mean, con_exo_std = np.mean(con_exo_raw, axis=0), np.std(con_exo_raw, axis=0)
    
    ref_mus_mean, ref_mus_std = np.mean(ref_mus_raw, axis=0), np.std(ref_mus_raw, axis=0)
    ref_ach_mean, ref_ach_std = np.mean(ref_ach_raw, axis=0), np.std(ref_ach_raw, axis=0)
    
    con_mus_mean, con_mus_std = np.mean(con_mus_raw, axis=0), np.std(con_mus_raw, axis=0)
    con_ach_mean, con_ach_std = np.mean(con_ach_raw, axis=0), np.std(con_ach_raw, axis=0)

    # 3. Calculate summary metabolic scalars
    j_pos_ref = np.trapz(np.maximum(ref_mus_mean, 0), dx=dt_stride)
    j_neg_ref = abs(np.trapz(np.minimum(ref_mus_mean, 0), dx=dt_stride))
    j_pos_con = np.trapz(np.maximum(con_mus_mean, 0), dx=dt_stride)
    j_neg_con = abs(np.trapz(np.minimum(con_mus_mean, 0), dx=dt_stride))
    est_mech_watts = ((4 * j_pos_ref + 1 * j_neg_ref) + (4 * j_pos_con + 1 * j_neg_con)) / mean_stride_dur

    ref_hum_mean = np.mean(np.array(profiles['ref_hum']), axis=0)
    con_hum_mean = np.mean(np.array(profiles['contra_hum']), axis=0)
    j_pos_ref_raw = np.trapz(np.maximum(ref_hum_mean, 0), dx=dt_stride)
    j_neg_ref_raw = abs(np.trapz(np.minimum(ref_hum_mean, 0), dx=dt_stride))
    j_pos_con_raw = np.trapz(np.maximum(con_hum_mean, 0), dx=dt_stride)
    j_neg_con_raw = abs(np.trapz(np.minimum(con_hum_mean, 0), dx=dt_stride))
    est_mech_watts_no_achilles = ((4 * j_pos_ref_raw + 1 * j_neg_ref_raw) + (4 * j_pos_con_raw + 1 * j_neg_con_raw)) / mean_stride_dur

    exo_power_net = (np.trapz(ref_exo_mean, dx=dt_stride) + np.trapz(con_exo_mean, dx=dt_stride)) / mean_stride_dur

    # 4. Build Wide Row
    row = {
        'trial_name': trial_name,
        'window_start_s': float(window_start_s),
        'mean_stride_duration_s': mean_stride_dur,
        'mean_duty_factor': stats.get('duty_factor_mean', 0.6),
        'net_bio_cost_w': net_bio_watts,
        'mechanical_power': est_mech_watts,
        'mechanical_power_no_achilles': est_mech_watts_no_achilles,
        'exo_power': exo_power_net
    }

    # Inject all 1D bucket columns
    for i in range(100):
        row[f'ref_exo_w_{i:02d}'] = ref_exo_mean[i]
        row[f'ref_ach_w_{i:02d}'] = ref_ach_mean[i]
        row[f'ref_mus_w_{i:02d}'] = ref_mus_mean[i]
        row[f'con_exo_w_{i:02d}'] = con_exo_mean[i]
        row[f'con_ach_w_{i:02d}'] = con_ach_mean[i]
        row[f'con_mus_w_{i:02d}'] = con_mus_mean[i]

        row[f'ref_exo_std_{i:02d}'] = ref_exo_std[i]
        row[f'ref_ach_std_{i:02d}'] = ref_ach_std[i]
        row[f'ref_mus_std_{i:02d}'] = ref_mus_std[i]
        row[f'con_exo_std_{i:02d}'] = con_exo_std[i]
        row[f'con_ach_std_{i:02d}'] = con_ach_std[i]
        row[f'con_mus_std_{i:02d}'] = con_mus_std[i]

        # Center of Mass (COM) Excursion (Zero-Centered)
        row[f'com_x_w_{i:02d}'] = com_x_centered[i]
        row[f'com_y_w_{i:02d}'] = com_y_centered[i]
        row[f'com_z_w_{i:02d}'] = com_z_centered[i]

        row[f'com_x_std_{i:02d}'] = com_x_std[i]
        row[f'com_y_std_{i:02d}'] = com_y_std[i]
        row[f'com_z_std_{i:02d}'] = com_z_std[i]

    return row


def process_trial(df, left_body, right_body, trial_name, burn_in_s, window_s, min_window_s):
    """
    Runs KF and gait tracking continuously over the whole trial, 
    extracting aggregated 5-minute chunks starting after burn_in_s.
    """
    times = df["time"].values
    dts = np.diff(times)
    default_dt = np.median(dts) if len(dts) > 0 else 0.01

    t_start = times[0]
    burn_in_thresh = t_start + burn_in_s
    current_window_start = burn_in_thresh
    current_window_end = current_window_start + window_s

    # Mass guess initialization
    f_total_y = df[f"{left_body}_force_y"].values + df[f"{right_body}_force_y"].values
    active_fy = f_total_y[f_total_y > 50.0]
    calc_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

    analyzer = EnergyAnalyzer(initial_mass=calc_mass, foot_roll_length=0.254)
    
    left_forces = df[[f"{left_body}_force_x", f"{left_body}_force_y", f"{left_body}_force_z"]].values
    right_forces = df[[f"{right_body}_force_x", f"{right_body}_force_y", f"{right_body}_force_z"]].values
    left_cops = df[[f"{left_body}_cop_x", f"{left_body}_cop_y", f"{left_body}_cop_z"]].values
    right_cops = df[[f"{right_body}_cop_x", f"{right_body}_cop_y", f"{right_body}_cop_z"]].values

    tauL_vals = df['tauL'].values if 'tauL' in df.columns else np.zeros(len(df))
    velL_vals = df['velaL'].values if 'velaL' in df.columns else np.zeros(len(df))
    tauR_vals = df['tauR'].values if 'tauR' in df.columns else np.zeros(len(df))
    velR_vals = df['velaR'].values if 'velaR' in df.columns else np.zeros(len(df))

    rows = []

    for i in range(len(df)):
        t = times[i]
        dt = dts[i] if i < len(dts) else default_dt
        forces = {'left': left_forces[i], 'right': right_forces[i]}
        cops = {'left': left_cops[i], 'right': right_cops[i]}
        
        # 1. Update filter continuously (Never reset)
        analyzer.update(t, forces, cops, dt, exo_power_left=(tauL_vals[i] * velL_vals[i]), exo_power_right=(tauR_vals[i] * velR_vals[i]))
        
        # 2. Burn-In Phase: continuously clear stride buffers so they don't pollute the first window
        if t < burn_in_thresh:
            clear_aggregations(analyzer)
            continue
            
        # 3. Window Completion: Process the last 5 minutes and reset buffers for the next
        if t >= current_window_end:
            df_win = df[(df['time'] >= current_window_start) & (df['time'] < current_window_end)]
            row = compile_window_row(df_win, analyzer, trial_name, current_window_start)
            if row:
                rows.append(row)
                print(f"  -> Window {current_window_start:.0f}s-{current_window_end:.0f}s | Mech: {row['mechanical_power']:.1f}W | Bio: {row['net_bio_cost_w']:.1f}W")
            
            clear_aggregations(analyzer)
            current_window_start = current_window_end
            current_window_end = current_window_start + window_s

    # 4. Handle final remaining tail of the trial
    if (times[-1] - current_window_start) >= min_window_s:
        df_win = df[(df['time'] >= current_window_start) & (df['time'] <= times[-1])]
        row = compile_window_row(df_win, analyzer, trial_name, current_window_start)
        if row:
            rows.append(row)
            print(f"  -> Tail Window {current_window_start:.0f}s-{times[-1]:.0f}s | Mech: {row['mechanical_power']:.1f}W | Bio: {row['net_bio_cost_w']:.1f}W")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Precompute continuous mechanics/metabolics into 5-minute chunks.")
    parser.add_argument("--dir", type=str, default=POGGENSEE_DIR)
    parser.add_argument("--burn-in", type=float, default=30.0, help="Seconds to discard at trial start")
    parser.add_argument("--window", type=float, default=300.0, help="Window size in seconds (default: 300s)")
    parser.add_argument("--min-window", type=float, default=180.0, help="Minimum valid tail window length in seconds")
    parser.add_argument("--out", type=str, default=os.path.join(POGGENSEE_DIR, "precomputed_poggensee.parquet"))
    args = parser.parse_args()

    parquet_path = os.path.abspath(args.out)
    processed_trials = set()

    if os.path.exists(parquet_path):
        try:
            existing_df = pd.read_parquet(parquet_path)
            for _, row in existing_df.iterrows():
                processed_trials.add(row['trial_name'])
            print(f"Found existing Parquet with {len(processed_trials)} completed trials. Resuming...")
        except Exception as e:
            print(f"Failed to read existing Parquet {parquet_path}: {e}")
            sys.exit(1)
    else:
        existing_df = pd.DataFrame()
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        print(f"Creating new precomputed Parquet at: {parquet_path}")

    files = glob.glob(os.path.join(os.path.abspath(args.dir), "*.parquet"))
    files = [f for f in files if "precomputed_poggensee" not in f]
    
    if not files:
        print(f"No valid .parquet trials found in {args.dir}")
        return

    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        trial_name = os.path.splitext(filename)[0]
        
        if trial_name in processed_trials:
            print(f"Skipping {filename} (Already completed)")
            continue

        print(f"\nProcessing {filename}...")
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

        force_cols = [col for col in df.columns if col.endswith("_force_y")]
        contact_bodies = [col.replace("_force_y", "") for col in force_cols]
        left_body = next((cb for cb in contact_bodies if cb.endswith("_l") or "left" in cb.lower()), contact_bodies[0])
        right_body = next((cb for cb in contact_bodies if cb.endswith("_r") or "right" in cb.lower()), contact_bodies[1])
            
        try:
            trial_rows = process_trial(df, left_body, right_body, trial_name, args.burn_in, args.window, args.min_window)
        except Exception as e:
            print(f"  -> Error calculating trial metrics: {e}")
            continue
        
        if trial_rows:
            new_rows_df = pd.DataFrame(trial_rows)
            existing_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
            existing_df.to_parquet(parquet_path, index=False)
            print(f"  -> Saved {len(trial_rows)} windows for {filename}")
        else:
            print(f"  -> Skipped: No valid windows generated for {filename}.")

    print("\nPrecompute process completed.")


if __name__ == "__main__":
    main()