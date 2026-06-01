# File: src/analysis/build_metabolics_csv.py

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


def calculate_window_metrics(df_window, left_body, right_body):
    """
    Processes a specific window of dataframe frames to compute 
    Metabolic Mask Cost and Mechanical Muscle Work.
    """
    # 1. Calculate Mask Metabolics (Weir Equation)
    vo2_col = next((c for c in df_window.columns if c.lower() == 'vo2'), None)
    vco2_col = next((c for c in df_window.columns if c.lower() == 'vco2'), None)
    
    if not vo2_col:
        return None, None, None, None
        
    vo2_mean = df_window[vo2_col].replace(0, np.nan).dropna().mean()
    if pd.isna(vo2_mean) or vo2_mean <= 0:
        return None, None, None, None
        
    vco2_mean = df_window[vco2_col].replace(0, np.nan).dropna().mean() if vco2_col else 0.85 * vo2_mean
    
    # 1 cal = 4.184 J, 1 min = 60s
    cal_per_min = 3.941 * vo2_mean + 1.106 * vco2_mean
    bio_watts = cal_per_min * 4.184 / 60.0
    
    # Check for dynamically extracted QS baseline, otherwise fallback to 70W
    standing_baseline = df_window["qs_baseline_w"].iloc[0] if "qs_baseline_w" in df_window.columns else 70.0
    net_bio_watts = bio_watts - standing_baseline
    
    # 2. Run Mechanical Energy Analyzer
    times = df_window["time"].values
    dts = np.diff(times)
    default_dt = np.median(dts) if len(dts) > 0 else 0.01

    f_total_y = df_window[f"{left_body}_force_y"].values + df_window[f"{right_body}_force_y"].values
    active_fy = f_total_y[f_total_y > 50.0]
    calc_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

    analyzer = EnergyAnalyzer(initial_mass=calc_mass, foot_roll_length=0.254)

    for i in range(len(df_window)):
        forces = {
            'left': np.array([df_window[f"{left_body}_force_x"].values[i], df_window[f"{left_body}_force_y"].values[i], df_window[f"{left_body}_force_z"].values[i]]),
            'right': np.array([df_window[f"{right_body}_force_x"].values[i], df_window[f"{right_body}_force_y"].values[i], df_window[f"{right_body}_force_z"].values[i]])
        }
        cops = {
            'left': np.array([df_window[f"{left_body}_cop_x"].values[i], df_window[f"{left_body}_cop_y"].values[i], df_window[f"{left_body}_cop_z"].values[i]]),
            'right': np.array([df_window[f"{right_body}_cop_x"].values[i], df_window[f"{right_body}_cop_y"].values[i], df_window[f"{right_body}_cop_z"].values[i]])
        }
        dt = dts[i] if i < len(dts) else default_dt
        
        tauL = df_window['tauL'].values[i] if 'tauL' in df_window.columns else 0.0
        velL = df_window['velaL'].values[i] if 'velaL' in df_window.columns else 0.0
        tauR = df_window['tauR'].values[i] if 'tauR' in df_window.columns else 0.0
        velR = df_window['velaR'].values[i] if 'velaR' in df_window.columns else 0.0
        
        analyzer.update(times[i], forces, cops, dt, exo_power_left=(tauL * velL), exo_power_right=(tauR * velR))

    if len(analyzer.stride_profiles['ref_sys']) == 0:
        return None, None, None, None

    # 3. Aggregate Mechanical Costs
    aggs = analyzer.get_stride_aggregates()
    stats = analyzer.stride_analyzer.get_metrics_summary()
    
    mean_stride_dur = stats.get('stride_duration_mean', 1.0)
    dt_stride = mean_stride_dur / 100.0

    ref_hum = np.array(aggs['ref_hum_mean'])
    con_hum = np.array(aggs['contra_hum_mean'])
    ref_exo = np.array(aggs['ref_exo_mean'])
    con_exo = np.array(aggs['contra_exo_mean'])
    
    # A) Mechanics WITH Achilles Storage Excluded
    ref_mus, _ = extract_biological_components(ref_hum, dt_stride)
    con_mus, _ = extract_biological_components(con_hum, dt_stride)

    j_pos_ref = np.trapz(np.maximum(ref_mus, 0), dx=dt_stride)
    j_neg_ref = abs(np.trapz(np.minimum(ref_mus, 0), dx=dt_stride))
    j_pos_con = np.trapz(np.maximum(con_mus, 0), dx=dt_stride)
    j_neg_con = abs(np.trapz(np.minimum(con_mus, 0), dx=dt_stride))
    
    est_mech_watts = ((4 * j_pos_ref + 1 * j_neg_ref) + (4 * j_pos_con + 1 * j_neg_con)) / mean_stride_dur

    # B) Mechanics WITHOUT Achilles (All Human Power = Muscle Power)
    j_pos_ref_raw = np.trapz(np.maximum(ref_hum, 0), dx=dt_stride)
    j_neg_ref_raw = abs(np.trapz(np.minimum(ref_hum, 0), dx=dt_stride))
    j_pos_con_raw = np.trapz(np.maximum(con_hum, 0), dx=dt_stride)
    j_neg_con_raw = abs(np.trapz(np.minimum(con_hum, 0), dx=dt_stride))

    est_mech_watts_no_achilles = ((4 * j_pos_ref_raw + 1 * j_neg_ref_raw) + (4 * j_pos_con_raw + 1 * j_neg_con_raw)) / mean_stride_dur

    # C) Net Exoskeleton Power
    exo_power_net = (np.trapz(ref_exo, dx=dt_stride) + np.trapz(con_exo, dx=dt_stride)) / mean_stride_dur

    return est_mech_watts, est_mech_watts_no_achilles, exo_power_net, net_bio_watts


def main():
    parser = argparse.ArgumentParser(description="Extract mechanics/metabolics to an interruptible CSV.")
    parser.add_argument("--dir", type=str, default=POGGENSEE_DIR)
    parser.add_argument("--window", type=float, default=300.0, help="Window size in seconds (default: 300s)")
    parser.add_argument("--min-window", type=float, default=180.0, help="Minimum valid window length in seconds")
    parser.add_argument("--out", type=str, default=os.path.join(POGGENSEE_DIR, "metabolics_summary.csv"))
    args = parser.parse_args()

    csv_path = os.path.abspath(args.out)
    processed_windows = set()

    HEADER = "trial_name,segment_index,mechanical_power,mechanical_power_no_achilles,exo_power,metabolic_power\n"

    # Load existing progress if available
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            
            # Check if it's the old format missing the new columns
            if 'mechanical_power_no_achilles' not in existing_df.columns:
                print(f"Old CSV format detected. Renaming {csv_path} to metabolics_summary_old.csv")
                os.rename(csv_path, csv_path.replace(".csv", "_old.csv"))
                # Re-initialize new file
                with open(csv_path, 'w') as f:
                    f.write(HEADER)
            else:
                for _, row in existing_df.iterrows():
                    # Round to 2 decimals to avoid floating point hash missing
                    processed_windows.add((row['trial_name'], round(float(row['segment_index']), 2)))
                print(f"Found existing CSV with {len(processed_windows)} completed windows. Resuming...")
        except Exception as e:
            print(f"Failed to read existing CSV {csv_path}: {e}")
            sys.exit(1)
    else:
        # Create new CSV file with headers
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write(HEADER)
        print(f"Created new summary CSV at: {csv_path}")

    files = glob.glob(os.path.join(os.path.abspath(args.dir), "*.parquet"))
    if not files:
        print(f"No .parquet files found in {args.dir}")
        return

    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        trial_name = os.path.splitext(filename)[0]
        
        print(f"\nProcessing {filename}...")
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

        vo2_col = next((c for c in df.columns if c.lower() == 'vo2'), None)
        if not vo2_col:
            print(f"  Skipping {filename} - No 'vo2' respirometry column found.")
            continue

        force_cols = [col for col in df.columns if col.endswith("_force_y")]
        contact_bodies = [col.replace("_force_y", "") for col in force_cols]
        left_body = next((cb for cb in contact_bodies if cb.endswith("_l") or "left" in cb.lower()), contact_bodies[0])
        right_body = next((cb for cb in contact_bodies if cb.endswith("_r") or "right" in cb.lower()), contact_bodies[1])

        t_min = df['time'].min()
        t_max = df['time'].max()
        
        # Iterate over sliding windows
        for w_start in np.arange(t_min, t_max, args.window):
            # Check if this window was already processed in a previous run
            if (trial_name, round(float(w_start), 2)) in processed_windows:
                print(f"  -> Skipping Window {w_start:.0f}s (Already completed)")
                continue

            w_end = w_start + args.window
            df_win = df[(df['time'] >= w_start) & (df['time'] < w_end)]
            
            # Ensure window is long enough
            if df_win.empty or (df_win['time'].max() - df_win['time'].min()) < args.min_window:
                continue
            
            try:
                mech_w, mech_w_no_ach, exo_w, bio_w = calculate_window_metrics(df_win, left_body, right_body)
            except Exception as e:
                print(f"  -> Error calculating window {w_start:.0f}s: {e}")
                continue
            
            if mech_w is not None and bio_w is not None:
                # Open append mode, write, and immediately flush so it is cleanly interruptible
                with open(csv_path, 'a') as f:
                    f.write(f"{trial_name},{w_start:.2f},{mech_w:.4f},{mech_w_no_ach:.4f},{exo_w:.4f},{bio_w:.4f}\n")
                    f.flush()
                print(f"  -> Window {w_start:.0f}s-{w_end:.0f}s | Mech: {mech_w:.1f}W (No-Ach: {mech_w_no_ach:.1f}W) | Exo: {exo_w:+.1f}W | Bio: {bio_w:.1f}W")

    print("\nFinished processing all files.")


if __name__ == "__main__":
    main()