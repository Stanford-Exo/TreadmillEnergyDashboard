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
    from matplotlib.path import Path
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

# Pastel Solid Colors (Reference Leg)
NOTION_REF_EXO = "#D3E5EF"      
NOTION_REF_HUM = "#D4B89F"      

# Lighter hatched faces with slightly darker edges (Contralateral Leg)
NOTION_CON_EXO_FACE = "#F2F6F9"
NOTION_CON_EXO_EDGE = "#8EB0C6"
NOTION_CON_HUM_FACE = "#F7F2EE"
NOTION_CON_HUM_EDGE = "#B59475"

TEXT_COLOR_EXO = "#3A637A"
TEXT_COLOR_HUM = "#7A583A"

def find_zero_crossings(y):
    return np.where(np.diff(np.sign(y)))[0]

def get_rotated_foot_marker(angle_deg):
    verts = np.array([
        [ 0.3, -0.15],  [-0.3, -0.15],  [-0.45, -0.05],
        [-0.4,  0.05],  [-0.1,  0.15],  [ 0.1,  0.45],
        [ 0.3,  0.45],  [ 0.4,  0.1],   [ 0.3, -0.15]
    ])
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return Path(np.dot(verts, R.T), [Path.MOVETO] + [Path.LINETO]*7 + [Path.CLOSEPOLY])

def plot_tufte_symmetric_stride(filename, aggs, mean_stride_dur, mean_duty_factor):
    """
    Plots the Symmetrically Aggregated Full Stride energetics.
    Stacks Reference Leg (Solid) and Contralateral Leg (Striped).
    """
    stride_pct = np.linspace(0, 100, 100)
    
    # Reference Leg Averages
    ref_exo = np.array(aggs['ref_exo_mean'])
    ref_hum = np.array(aggs['ref_hum_mean'])
    ref_sys = np.array(aggs['ref_sys_mean'])
    
    # Contralateral Leg Averages
    con_exo = np.array(aggs['contra_exo_mean'])
    con_hum = np.array(aggs['contra_hum_mean'])
    con_sys = np.array(aggs['contra_sys_mean'])

    # Raw arrays for calculating standard deviations
    raw_ref_exo = aggs['raw']['ref_exo']
    raw_ref_hum = aggs['raw']['ref_hum']
    raw_con_exo = aggs['raw']['contra_exo']
    raw_con_hum = aggs['raw']['contra_hum']

    # Total System (Net)
    tot_sys = ref_sys + con_sys

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    
    fig.text(0.04, 0.93, f"Bilateral System Energetics: {filename}", fontsize=16, fontweight='bold', color=NOTION_TEXT)
    fig.text(0.04, 0.88, "Full Stride Cycle: Reference Leg (Solid) & Contralateral Leg (Striped) in Stationary Frame", fontsize=12, color=NOTION_SUBTEXT)
    plt.subplots_adjust(top=0.80, bottom=0.18)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    ax.tick_params(axis='y', colors=NOTION_TEXT, length=0)
    ax.tick_params(axis='x', colors=NOTION_TEXT, length=0) 
    ax.grid(axis='y', color='#EDEDED', linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)

    # --- 1. Compute Cumulative Stack Geometries ---
    z = np.zeros(100)
    
    pos_ref_exo = np.maximum(ref_exo, 0)
    pos_ref_hum = pos_ref_exo + np.maximum(ref_hum, 0)
    pos_con_exo = pos_ref_hum + np.maximum(con_exo, 0)
    pos_con_hum = pos_con_exo + np.maximum(con_hum, 0)

    neg_ref_exo = np.minimum(ref_exo, 0)
    neg_ref_hum = neg_ref_exo + np.minimum(ref_hum, 0)
    neg_con_exo = neg_ref_hum + np.minimum(con_exo, 0)
    neg_con_hum = neg_con_exo + np.minimum(con_hum, 0)

    # --- 2. Fill Polygons ---
    ax.fill_between(stride_pct, z, pos_ref_exo, color=NOTION_REF_EXO, alpha=0.95, linewidth=0, label="Ref Leg Exo")
    ax.fill_between(stride_pct, pos_ref_exo, pos_ref_hum, color=NOTION_REF_HUM, alpha=0.95, linewidth=0, label="Ref Leg Human")
    ax.fill_between(stride_pct, pos_ref_hum, pos_con_exo, facecolor=NOTION_CON_EXO_FACE, edgecolor=NOTION_CON_EXO_EDGE, hatch='////', linewidth=0.5, label="Contra Leg Exo")
    ax.fill_between(stride_pct, pos_con_exo, pos_con_hum, facecolor=NOTION_CON_HUM_FACE, edgecolor=NOTION_CON_HUM_EDGE, hatch='////', linewidth=0.5, label="Contra Leg Human")

    ax.fill_between(stride_pct, z, neg_ref_exo, color=NOTION_REF_EXO, alpha=0.95, linewidth=0)
    ax.fill_between(stride_pct, neg_ref_exo, neg_ref_hum, color=NOTION_REF_HUM, alpha=0.95, linewidth=0)
    ax.fill_between(stride_pct, neg_ref_hum, neg_con_exo, facecolor=NOTION_CON_EXO_FACE, edgecolor=NOTION_CON_EXO_EDGE, hatch='////', linewidth=0.5)
    ax.fill_between(stride_pct, neg_con_exo, neg_con_hum, facecolor=NOTION_CON_HUM_FACE, edgecolor=NOTION_CON_HUM_EDGE, hatch='////', linewidth=0.5)

    # --- 3. Net System Power Line ---
    ax.plot(stride_pct, tot_sys, color="#111827", linestyle=":", linewidth=2.5, label="Net System Power", zorder=10)

    # --- 4. Draw Centered Text Labels ---
    stroke = [pe.withStroke(linewidth=3, foreground=NOTION_BG, alpha=0.8)]
    dt = mean_stride_dur / 100.0  
    
    def label_blocks(sys_curve, exo_curve, hum_curve, raw_exo, raw_hum, pos_base, pos_top, neg_base, neg_top, t_color_exo, t_color_hum):
        crossings = find_zero_crossings(sys_curve)
        boundaries = [0] + list(crossings) + [len(sys_curve)-1]
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if (end - start) < 5: continue
            
            exo_j_list = [np.trapz(raw_exo[s_idx][start:end], dx=dt) for s_idx in range(len(raw_exo))]
            hum_j_list = [np.trapz(raw_hum[s_idx][start:end], dx=dt) for s_idx in range(len(raw_hum))]
            
            m_exo_j, s_exo_j = np.mean(exo_j_list), np.std(exo_j_list)
            m_hum_j, s_hum_j = np.mean(hum_j_list), np.std(hum_j_list)
            
            peak_idx = start + np.argmax(np.abs(sys_curve[start:end]))
            cx = stride_pct[peak_idx]
            
            p_e = exo_curve[peak_idx]
            p_h = hum_curve[peak_idx]
            
            y_exo = pos_base[peak_idx] + p_e/2 if p_e > 0 else neg_base[peak_idx] + p_e/2
            y_hum = pos_top[peak_idx] + p_h/2 if p_h > 0 else neg_top[peak_idx] + p_h/2
                
            if abs(m_exo_j) >= 0.5 and abs(p_e) >= 4.0:
                ax.text(cx, y_exo, f"{m_exo_j:+.1f} J\n±{s_exo_j:.1f} J", color=t_color_exo, fontsize=9, fontweight='bold', ha='center', va='center', path_effects=stroke, zorder=15)
            if abs(m_hum_j) >= 0.5 and abs(p_h) >= 4.0:
                ax.text(cx, y_hum, f"{m_hum_j:+.1f} J\n±{s_hum_j:.1f} J", color=t_color_hum, fontsize=9, fontweight='bold', ha='center', va='center', path_effects=stroke, zorder=15)

    label_blocks(ref_sys, ref_exo, ref_hum, raw_ref_exo, raw_ref_hum,
                 pos_base=z, pos_top=pos_ref_exo, neg_base=z, neg_top=neg_ref_exo, 
                 t_color_exo=TEXT_COLOR_EXO, t_color_hum=TEXT_COLOR_HUM)
    
    label_blocks(con_sys, con_exo, con_hum, raw_con_exo, raw_con_hum,
                 pos_base=pos_ref_hum, pos_top=pos_con_exo, neg_base=neg_ref_hum, neg_top=neg_con_exo, 
                 t_color_exo=TEXT_COLOR_EXO, t_color_hum=TEXT_COLOR_HUM)

    # --- 5. Custom X-Axis formatting (Ticks and Foot Icons) ---
    ax.set_xlim(0, 100) 
    
    duty_pct = mean_duty_factor * 100
    ax.set_xticks([0, duty_pct, 100])
    ax.set_xticklabels([]) 
    
    # Tick lines dropping down
    ax.axvline(0, color='black', linewidth=1.5, linestyle='-', zorder=5)
    ax.axvline(duty_pct, color='black', linewidth=1.5, linestyle='-', zorder=5)
    ax.axvline(100, color='black', linewidth=1.5, linestyle='-', zorder=5)

    # Place Foot Icons & Labels
    icon_y_pos = -0.12
    text_y_pos = -0.22
    
    ax.scatter(0, icon_y_pos, s=700, marker=get_rotated_foot_marker(-25), color=NOTION_TEXT, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(0, text_y_pos, "Heel-Strike", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=11, fontweight='500')
    
    ax.scatter(duty_pct, icon_y_pos, s=700, marker=get_rotated_foot_marker(30), color=NOTION_TEXT, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(duty_pct, text_y_pos, "Toe-Off", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=11, fontweight='500')

    ax.scatter(100, icon_y_pos, s=700, marker=get_rotated_foot_marker(-25), color=NOTION_TEXT, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(100, text_y_pos, "Heel-Strike", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=11, fontweight='500')

    ax.text(0.5, -0.35, "Full Stride Cycle (%)", transform=ax.transAxes, ha='center', va='top', color=NOTION_SUBTEXT, fontsize=11, fontweight='bold')
    
    ax.set_ylabel("Mechanical Power (W)", fontsize=11, fontweight='bold', color=NOTION_TEXT)
    ax.axhline(0, color=NOTION_TEXT, linewidth=1.5) 
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
            
    ax.legend(unique_handles[::-1], unique_labels[::-1], frameon=False, loc="upper left", 
              fontsize=11, labelcolor=NOTION_TEXT, bbox_to_anchor=(0, 1.05), ncol=5)
    
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
            
            tauL = df['tauL'].values[i] if 'tauL' in df.columns else 0.0
            velL = df['velaL'].values[i] if 'velaL' in df.columns else 0.0
            tauR = df['tauR'].values[i] if 'tauR' in df.columns else 0.0
            velR = df['velaR'].values[i] if 'velaR' in df.columns else 0.0
            
            analyzer.update(times[i], forces, cops, dt, 
                            exo_power_left=(tauL * velL), 
                            exo_power_right=(tauR * velR))

        stats = analyzer.stride_analyzer.get_metrics_summary()
        mean_stride_dur = stats.get('stride_duration_mean', 1.0)
        mean_duty_factor = stats.get('duty_factor_mean', 0.6)
        
        print(f"  Completed Symmetric Strides: {len(analyzer.stride_profiles['ref_sys'])}")
        print(f"  Treadmill Speed Estimate: {stats.get('estimated_belt_speed', 0.0):.3f} m/s")

        if len(analyzer.stride_profiles['ref_sys']) > 0:
            aggs = analyzer.get_stride_aggregates()
            aggs['raw'] = analyzer.stride_profiles 
            plot_tufte_symmetric_stride(filename, aggs, mean_stride_dur, mean_duty_factor)

if __name__ == "__main__":
    main()