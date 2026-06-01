# File: src/analysis/analyze_poggensee.py

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

from online_analyze.energy_analyzer import EnergyAnalyzer

try:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
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

NOTION_EXO_COLOR = "#D3E5EF"      # Soft blue
NOTION_MUSCLE_COLOR = "#D4B89F"   # Deeper reddish-brown
LINE_EXO_COLOR = "#759CB4"        # Darker blue for swing lines
LINE_MUSCLE_COLOR = "#A37C5B"     # Darker brown for swing lines

def find_zero_crossings(y):
    return np.where(np.diff(np.sign(y)))[0]

def plot_tufte_symmetric_stance(filename, aggs, mean_stance_dur):
    """
    Plots the Symmetrically Aggregated Stance vs Swing energetics.
    Stacks Stance forces, lines Swing forces.
    """
    stance_pct = np.linspace(0, 100, 100)
    
    # Stance Leg Averages (Filled)
    st_exo = np.array(aggs['stance_exo_mean'])
    st_hum = np.array(aggs['stance_hum_mean'])
    st_sys = np.array(aggs['stance_sys_mean'])
    
    # Swing Leg Averages (Lines)
    sw_exo = np.array(aggs['swing_exo_mean'])
    sw_hum = np.array(aggs['swing_hum_mean'])

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    
    fig.text(0.04, 0.93, f"Symmetric Stance Energetics: {filename}", fontsize=16, fontweight='bold', color=NOTION_TEXT)
    fig.text(0.04, 0.88, "Stance Leg (Filled) vs Swing Leg (Lines) - Stationary Ground Frame", fontsize=12, color=NOTION_SUBTEXT)
    plt.subplots_adjust(top=0.80, bottom=0.15)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    ax.tick_params(axis='y', colors=NOTION_TEXT, length=0)
    ax.tick_params(axis='x', colors=NOTION_TEXT, length=0) 
    ax.grid(axis='y', color='#EDEDED', linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)

    # 1. Fill Stance Leg Power
    st_exo_pos = np.maximum(st_exo, 0)
    st_exo_neg = np.minimum(st_exo, 0)
    st_hum_pos = np.maximum(st_hum, 0)
    st_hum_neg = np.minimum(st_hum, 0)

    ax.fill_between(stance_pct, 0, st_exo_pos, color=NOTION_EXO_COLOR, alpha=0.9, label="Stance Exo")
    ax.fill_between(stance_pct, st_exo_pos, st_exo_pos + st_hum_pos, color=NOTION_MUSCLE_COLOR, alpha=0.9, label="Stance Human")
    
    ax.fill_between(stance_pct, 0, st_exo_neg, color=NOTION_EXO_COLOR, alpha=0.9)
    ax.fill_between(stance_pct, st_exo_neg, st_exo_neg + st_hum_neg, color=NOTION_MUSCLE_COLOR, alpha=0.9)

    # 2. Plot Swing Leg Power Lines
    ax.plot(stance_pct, sw_exo, color=LINE_EXO_COLOR, linestyle='--', linewidth=2, label="Swing Exo")
    ax.plot(stance_pct, sw_hum, color=LINE_MUSCLE_COLOR, linestyle='--', linewidth=2, label="Swing Human")

    # 3. Label Stance Energy Blocks (Joules)
    crossings = find_zero_crossings(st_sys)
    boundaries = [0] + list(crossings) + [len(st_sys)-1]
    
    stroke = [pe.withStroke(linewidth=3, foreground=NOTION_BG)]
    dt = mean_stance_dur / 100.0  
    
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if (end - start) < 5:
            continue
            
        m_exo_j = np.sum(st_exo[start:end]) * dt
        m_hum_j = np.sum(st_hum[start:end]) * dt
        
        peak_idx = start + np.argmax(np.abs(st_sys[start:end]))
        cx = stance_pct[peak_idx]
        p_e = st_exo[peak_idx]
        p_h = st_hum[peak_idx]
        
        y_exo = p_e / 2
        y_hum = p_e + (p_h / 2) if np.sign(p_e) == np.sign(p_h) else p_h / 2
            
        if abs(m_exo_j) >= 0.5 and abs(p_e) >= 4.0:
            ax.text(cx, y_exo, f"{m_exo_j:+.1f} J", color=LINE_EXO_COLOR, 
                    fontsize=10, fontweight='bold', ha='center', va='center', path_effects=stroke)
            
        if abs(m_hum_j) >= 0.5 and abs(p_h) >= 4.0:
            ax.text(cx, y_hum, f"{m_hum_j:+.1f} J", color=LINE_MUSCLE_COLOR, 
                    fontsize=10, fontweight='bold', ha='center', va='center', path_effects=stroke)

    ax.set_xlim(0, 100) 
    ax.set_xlabel("Stance Phase (%) - Heel Strike to Toe Off", fontsize=11, fontweight='bold', color=NOTION_SUBTEXT)
    ax.set_ylabel("Mechanical Power (W)", fontsize=11, fontweight='bold', color=NOTION_TEXT)
    ax.axhline(0, color=NOTION_TEXT, linewidth=1.5) 
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
            
    # Ordering legend for visual coherence
    ax.legend(unique_handles[::-1], unique_labels[::-1], frameon=False, loc="upper left", 
              fontsize=11, labelcolor=NOTION_TEXT, bbox_to_anchor=(0, 1.05), ncol=2)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze Poggensee exoskeleton trial datasets.")
    parser.add_argument("--dir", type=str, default=POGGENSEE_DIR)
    args = parser.parse_args()

    if plt is None:
        print("Error: The 'matplotlib' library is required to display plots. Please install it: pip install matplotlib")
        sys.exit(1)

    files = glob.glob(os.path.join(os.path.abspath(args.dir), "*.parquet"))
    
    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        print(f"\nProcessing trial: {filename}")
        df = pd.read_parquet(file_path)

        force_cols = [col for col in df.columns if col.endswith("_force_y")]
        contact_bodies = [col.replace("_force_y", "") for col in force_cols]
        left_body = next((cb for cb in contact_bodies if cb.endswith("_l") or "left" in cb.lower()), contact_bodies[0])
        right_body = next((cb for cb in contact_bodies if cb.endswith("_r") or "right" in cb.lower()), contact_bodies[1])

        times = df["time"].values
        dts = np.diff(times)
        default_dt = np.median(dts) if len(dts) > 0 else 0.01

        f_total_y = df[f"{left_body}_force_y"].values + df[f"{right_body}_force_y"].values
        active_fy = f_total_y[f_total_y > 50.0]
        calc_mass = np.mean(active_fy) / 9.81 if len(active_fy) > 0 else 70.0

        analyzer = EnergyAnalyzer(initial_mass=calc_mass, foot_roll_length=0.254)

        for i in range(min(len(df), 150000)):
            if i % 25000 == 0:
                print(f"  Processing frame {i}/{len(df)}...")
                
            forces = {
                'left': np.array([df[f"{left_body}_force_x"].values[i], df[f"{left_body}_force_y"].values[i], df[f"{left_body}_force_z"].values[i]]),
                'right': np.array([df[f"{right_body}_force_x"].values[i], df[f"{right_body}_force_y"].values[i], df[f"{right_body}_force_z"].values[i]])
            }
            cops = {
                'left': np.array([df[f"{left_body}_cop_x"].values[i], df[f"{left_body}_cop_y"].values[i], df[f"{left_body}_cop_z"].values[i]]),
                'right': np.array([df[f"{right_body}_cop_x"].values[i], df[f"{right_body}_cop_y"].values[i], df[f"{right_body}_cop_z"].values[i]])
            }
            dt = dts[i] if i < len(dts) else default_dt
            
            # Retrieve Exo Power natively 
            tauL = df['tauL'].values[i] if 'tauL' in df.columns else 0.0
            velL = df['velaL'].values[i] if 'velaL' in df.columns else 0.0
            tauR = df['tauR'].values[i] if 'tauR' in df.columns else 0.0
            velR = df['velaR'].values[i] if 'velaR' in df.columns else 0.0
            
            analyzer.update(times[i], forces, cops, dt, 
                            exo_power_left=(tauL * velL), 
                            exo_power_right=(tauR * velR))

        stats = analyzer.stride_analyzer.get_metrics_summary()
        mean_stance_dur = stats.get('stance_duration_mean', 0.6)
        
        print(f"  Completed Symmetric Stances: {len(analyzer.stance_profiles['stance_sys'])}")
        print(f"  Treadmill Speed Estimate: {stats.get('estimated_belt_speed', 0.0):.3f} m/s")

        if len(analyzer.stance_profiles['stance_sys']) > 0:
            aggs = analyzer.get_stance_aggregates()
            plot_tufte_symmetric_stance(filename, aggs, mean_stance_dur)

if __name__ == "__main__":
    main()