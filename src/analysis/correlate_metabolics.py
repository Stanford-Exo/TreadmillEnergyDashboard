# File: src/analysis/correlate_metabolics.py

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

from online_analyze.energy_analyzer import EnergyAnalyzer

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    plt = None

POGGENSEE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))

# --- Tufte Aesthetics ---
if plt is not None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', '-apple-system', 'Arial', 'sans-serif']

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"


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
        return None, None
        
    vo2_mean = df_window[vo2_col].replace(0, np.nan).dropna().mean()
    if pd.isna(vo2_mean) or vo2_mean <= 0:
        return None, None
        
    vco2_mean = df_window[vco2_col].replace(0, np.nan).dropna().mean() if vco2_col else 0.85 * vo2_mean
    
    # 1 cal = 4.184 J, 1 min = 60s
    cal_per_min = 3.941 * vo2_mean + 1.106 * vco2_mean
    bio_watts = cal_per_min * 4.184 / 60.0
    net_bio_watts = bio_watts - 70.0  # Subtract standing baseline
    
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
        return None, None

    # 3. Aggregate Mechanical Muscle Cost
    aggs = analyzer.get_stride_aggregates()
    stats = analyzer.stride_analyzer.get_metrics_summary()
    
    mean_stride_dur = stats.get('stride_duration_mean', 1.0)
    dt_stride = mean_stride_dur / 100.0

    ref_hum = np.array(aggs['ref_hum_mean'])
    con_hum = np.array(aggs['contra_hum_mean'])
    
    ref_mus, _ = extract_biological_components(ref_hum, dt_stride)
    con_mus, _ = extract_biological_components(con_hum, dt_stride)

    j_pos_ref = np.trapz(np.maximum(ref_mus, 0), dx=dt_stride)
    j_neg_ref = abs(np.trapz(np.minimum(ref_mus, 0), dx=dt_stride))
    
    j_pos_con = np.trapz(np.maximum(con_mus, 0), dx=dt_stride)
    j_neg_con = abs(np.trapz(np.minimum(con_mus, 0), dx=dt_stride))
    
    est_mech_watts = ((4 * j_pos_ref + 1 * j_neg_ref) + (4 * j_pos_con + 1 * j_neg_con)) / mean_stride_dur

    return est_mech_watts, net_bio_watts


def plot_correlation(scatter_data):
    """
    Plots the scatter correlation between Estimated Muscle Power and Biological Cost.
    """
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    
    fig.text(0.05, 0.93, "Metabolic Validation: Mechanical vs. Respirometry", fontsize=16, fontweight='bold', color=NOTION_TEXT)
    fig.text(0.05, 0.89, "Each point represents a 5-minute walking window. 4x/1x Muscle heuristic applied.", fontsize=11, color=NOTION_SUBTEXT)
    plt.subplots_adjust(top=0.82, right=0.95)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(NOTION_TEXT)
        
    ax.grid(color=NOTION_GRID, linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)

    x_vals = [d['mech_w'] for d in scatter_data]
    y_vals = [d['bio_w'] for d in scatter_data]
    files = [d['file'] for d in scatter_data]

    # Generate distinct colors for different files
    unique_files = list(set(files))
    colormap = cm.get_cmap('tab10', len(unique_files))
    
    # Plot Scatter Points by File
    for i, file_name in enumerate(unique_files):
        fx = [d['mech_w'] for d in scatter_data if d['file'] == file_name]
        fy = [d['bio_w'] for d in scatter_data if d['file'] == file_name]
        
        ax.scatter(fx, fy, color=colormap(i), label=file_name, s=80, alpha=0.8, edgecolors='white', linewidths=0.5)

    # 1:1 Identity Line
    max_val = max(max(x_vals), max(y_vals)) * 1.1
    min_val = min(min(x_vals), min(y_vals)) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], color=NOTION_SUBTEXT, linestyle='--', zorder=0, label="1:1 Unity Line (y=x)")

    # Linear Regression Fit
    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    x_line = np.linspace(min_val, max_val, 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, color=NOTION_TEXT, linewidth=2.5, zorder=5, label=f"Linear Fit (y = {slope:.2f}x {'+' if intercept>0 else '-'} {abs(intercept):.1f})")

    # Add R-squared text box
    stats_text = f"$R^2$: {r_value**2:.3f}\nPearson $r$: {r_value:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=12, va='top', ha='left', color=NOTION_TEXT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=NOTION_BG, edgecolor='#EDEDED', alpha=0.9))

    # Formatting
    ax.set_xlabel("Estimated Mechanical Muscle Cost (W)", fontsize=11, fontweight='bold', color=NOTION_TEXT)
    ax.set_ylabel("Net Measured Biological Cost (W)", fontsize=11, fontweight='bold', color=NOTION_TEXT)
    
    ax.legend(frameon=True, facecolor=NOTION_BG, edgecolor=NOTION_GRID, fontsize=9, loc='lower right')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Correlate mechanical estimates with mask metabolics over 5-min windows.")
    parser.add_argument("--dir", type=str, default=POGGENSEE_DIR)
    parser.add_argument("--window", type=float, default=300.0, help="Window size in seconds (default: 300s / 5 min)")
    parser.add_argument("--min-window", type=float, default=180.0, help="Minimum valid window length in seconds (default: 180s)")
    args = parser.parse_args()

    if plt is None:
        print("Error: matplotlib required. pip install matplotlib")
        sys.exit(1)

    files = glob.glob(os.path.join(os.path.abspath(args.dir), "*.parquet"))
    if not files:
        print(f"No .parquet files found in {args.dir}")
        return

    scatter_data = []

    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        print(f"\nProcessing {filename}...")
        df = pd.read_parquet(file_path)

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
            w_end = w_start + args.window
            df_win = df[(df['time'] >= w_start) & (df['time'] < w_end)]
            
            # Ensure window is long enough (prevents tiny fragmented windows at the end of a trial)
            if df_win.empty or (df_win['time'].max() - df_win['time'].min()) < args.min_window:
                continue
                
            mech_w, bio_w = calculate_window_metrics(df_win, left_body, right_body)
            
            if mech_w is not None and bio_w is not None:
                scatter_data.append({
                    'file': os.path.splitext(filename)[0],
                    'window_start': w_start,
                    'mech_w': mech_w,
                    'bio_w': bio_w
                })
                print(f"  -> Window {w_start:.0f}s - {w_end:.0f}s | Mech: {mech_w:.1f} W | Bio: {bio_w:.1f} W")

    if not scatter_data:
        print("\nNo valid metabolic windows were processed. Exiting.")
        return

    print(f"\nSuccessfully generated {len(scatter_data)} total data points across all trials.")
    plot_correlation(scatter_data)


if __name__ == "__main__":
    main()