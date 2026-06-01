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
NOTION_REF_ACH = "#EBE0C5"  # Lighter, elastic golden-beige
NOTION_REF_MUS = "#D4B89F"  # Deeper, active reddish-brown

# Lighter hatched faces with slightly darker edges (Contralateral Leg)
NOTION_CON_EXO_FACE = "#F2F6F9"
NOTION_CON_EXO_EDGE = "#8EB0C6"
NOTION_CON_ACH_FACE = "#FDFBEF"
NOTION_CON_ACH_EDGE = "#D1C090"
NOTION_CON_MUS_FACE = "#F7F2EE"
NOTION_CON_MUS_EDGE = "#B59475"

TEXT_COLOR_EXO = "#3A637A"
TEXT_COLOR_ACH = "#9E814D"
TEXT_COLOR_MUS = "#7A583A"


def find_zero_crossings(y):
    return np.where(np.diff(np.sign(y)))[0]


def extract_biological_components(human_power, dt):
    """
    Splits human power into 'Likely Achilles' (balanced zero-net energy) and 'Muscle Power'.
    Achilles acts as a spring matching the negative loading lump with the positive push-off lump.
    We double the array to smoothly handle contralateral phase wrapping across the 0/100 boundary.
    """
    N = len(human_power)
    doubled_power = np.concatenate([human_power, human_power])
    achilles_doubled = np.zeros_like(doubled_power)
    
    # Peak positive power between N//2 and 3N//2 ensures we have space backwards and forwards
    search_area = doubled_power[N//2 : N + N//2]
    if len(search_area) == 0:
        return human_power, np.zeros_like(human_power)
        
    peak_local = np.argmax(search_area)
    peak_idx = N//2 + peak_local
    
    # If the absolute peak is negative or zero, there's no push-off. Achilles = 0
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
        
        # e_neg must actually be negative for it to be elastic storage
        if e_pos > 0 and e_neg < 0:
            achilles_energy = min(e_pos, abs(e_neg))
            
            scale_pos = achilles_energy / e_pos if e_pos != 0 else 0
            scale_neg = achilles_energy / abs(e_neg) if e_neg != 0 else 0
            
            achilles_doubled[pos_start:pos_end] = pos_chunk * scale_pos
            achilles_doubled[neg_start:neg_end] = neg_chunk * scale_neg
            
    # Fold the doubled array back onto the 0-100% boundary
    achilles_power = achilles_doubled[:N] + achilles_doubled[N:]
    muscle_power = human_power - achilles_power
    
    return muscle_power, achilles_power


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


def plot_tufte_symmetric_stride(filename, aggs, mean_stride_dur, mean_duty_factor, net_bio_watts=None):
    """
    Plots the Symmetrically Aggregated Full Stride energetics.
    Stacks Reference Leg (Solid) and Contralateral Leg (Striped).
    """
    dt = mean_stride_dur / 100.0  
    stride_pct = np.linspace(0, 100, 100)
    
    # --- Reference Leg Averages ---
    ref_exo = np.array(aggs['ref_exo_mean'])
    ref_hum = np.array(aggs['ref_hum_mean'])
    ref_sys = np.array(aggs['ref_sys_mean'])
    ref_mus, ref_ach = extract_biological_components(ref_hum, dt)
    
    # --- Contralateral Leg Averages ---
    con_exo = np.array(aggs['contra_exo_mean'])
    con_hum = np.array(aggs['contra_hum_mean'])
    con_sys = np.array(aggs['contra_sys_mean'])
    con_mus, con_ach = extract_biological_components(con_hum, dt)

    # --- Compute Mechanical Metabolic Estimates ---
    j_pos_ref = np.trapz(np.maximum(ref_mus, 0), dx=dt)
    j_neg_ref = abs(np.trapz(np.minimum(ref_mus, 0), dx=dt))
    
    j_pos_con = np.trapz(np.maximum(con_mus, 0), dx=dt)
    j_neg_con = abs(np.trapz(np.minimum(con_mus, 0), dx=dt))
    
    # 4x Positive Cost, 1x Negative Cost for muscle
    est_metabolic_watts = ((4 * j_pos_ref + 1 * j_neg_ref) + (4 * j_pos_con + 1 * j_neg_con)) / mean_stride_dur

    # --- Decompose Raw Stride Buffers for Standard Deviation calculations ---
    raw_ref_exo = aggs['raw']['ref_exo']
    raw_ref_ach = []
    raw_ref_mus = []
    for rh in aggs['raw']['ref_hum']:
        rm, ra = extract_biological_components(np.array(rh), dt)
        raw_ref_mus.append(rm)
        raw_ref_ach.append(ra)
        
    raw_con_exo = aggs['raw']['contra_exo']
    raw_con_ach = []
    raw_con_mus = []
    for ch in aggs['raw']['contra_hum']:
        cm, ca = extract_biological_components(np.array(ch), dt)
        raw_con_mus.append(cm)
        raw_con_ach.append(ca)

    # Total System (Net)
    tot_sys = ref_sys + con_sys

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    
    fig.text(0.04, 0.93, f"Bilateral System Energetics: {filename}", fontsize=16, fontweight='bold', color=NOTION_TEXT)
    fig.text(0.04, 0.88, "Full Stride Cycle: Reference Leg (Solid) & Contralateral Leg (Striped) in Stationary Frame", fontsize=12, color=NOTION_SUBTEXT)
    
    # Margins expanded slightly to fit new Contra labels cleanly
    plt.subplots_adjust(top=0.77, bottom=0.20)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    ax.tick_params(axis='y', colors=NOTION_TEXT, length=0)
    ax.tick_params(axis='x', colors=NOTION_TEXT, length=0) 
    ax.grid(axis='y', color='#EDEDED', linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)

    # --- 1. Compute Cumulative Stack Geometries ---
    z = np.zeros(100)
    
    pos_ref_exo = np.maximum(ref_exo, 0)
    pos_ref_ach = pos_ref_exo + np.maximum(ref_ach, 0)
    pos_ref_mus = pos_ref_ach + np.maximum(ref_mus, 0)
    
    pos_con_exo = pos_ref_mus + np.maximum(con_exo, 0)
    pos_con_ach = pos_con_exo + np.maximum(con_ach, 0)
    pos_con_mus = pos_con_ach + np.maximum(con_mus, 0)

    neg_ref_exo = np.minimum(ref_exo, 0)
    neg_ref_ach = neg_ref_exo + np.minimum(ref_ach, 0)
    neg_ref_mus = neg_ref_ach + np.minimum(ref_mus, 0)
    
    neg_con_exo = neg_ref_mus + np.minimum(con_exo, 0)
    neg_con_ach = neg_con_exo + np.minimum(con_ach, 0)
    neg_con_mus = neg_con_ach + np.minimum(con_mus, 0)

    # Calculate padding bounds to keep data below the legend and limit vertical lines
    max_power = max(np.max(pos_ref_mus), np.max(pos_con_mus), np.max(tot_sys))
    min_power = min(np.min(neg_ref_mus), np.min(neg_con_mus), np.min(tot_sys))
    
    y_upper = max_power * 1.45  # 45% headroom specifically for the legend
    y_lower = min_power * 1.15  # 15% footroom below the deepest negative power
    
    ax.set_ylim(y_lower, y_upper)

    # --- 2. Fill Polygons ---
    # Reference Fills
    ax.fill_between(stride_pct, z, pos_ref_exo, color=NOTION_REF_EXO, alpha=0.95, linewidth=0, label="Ref Leg Exo")
    ax.fill_between(stride_pct, pos_ref_exo, pos_ref_ach, color=NOTION_REF_ACH, alpha=0.95, linewidth=0, label="Ref Leg Likely Achilles")
    ax.fill_between(stride_pct, pos_ref_ach, pos_ref_mus, color=NOTION_REF_MUS, alpha=0.95, linewidth=0, label="Ref Leg Muscle")

    # Contralateral Fills
    ax.fill_between(stride_pct, pos_ref_mus, pos_con_exo, facecolor=NOTION_CON_EXO_FACE, edgecolor=NOTION_CON_EXO_EDGE, hatch='////', linewidth=0.5, label="Contra Leg Exo")
    ax.fill_between(stride_pct, pos_con_exo, pos_con_ach, facecolor=NOTION_CON_ACH_FACE, edgecolor=NOTION_CON_ACH_EDGE, hatch='////', linewidth=0.5, label="Contra Leg Likely Achilles")
    ax.fill_between(stride_pct, pos_con_ach, pos_con_mus, facecolor=NOTION_CON_MUS_FACE, edgecolor=NOTION_CON_MUS_EDGE, hatch='////', linewidth=0.5, label="Contra Leg Muscle")

    # Negative Fills
    ax.fill_between(stride_pct, z, neg_ref_exo, color=NOTION_REF_EXO, alpha=0.95, linewidth=0)
    ax.fill_between(stride_pct, neg_ref_exo, neg_ref_ach, color=NOTION_REF_ACH, alpha=0.95, linewidth=0)
    ax.fill_between(stride_pct, neg_ref_ach, neg_ref_mus, color=NOTION_REF_MUS, alpha=0.95, linewidth=0)

    ax.fill_between(stride_pct, neg_ref_mus, neg_con_exo, facecolor=NOTION_CON_EXO_FACE, edgecolor=NOTION_CON_EXO_EDGE, hatch='////', linewidth=0.5)
    ax.fill_between(stride_pct, neg_con_exo, neg_con_ach, facecolor=NOTION_CON_ACH_FACE, edgecolor=NOTION_CON_ACH_EDGE, hatch='////', linewidth=0.5)
    ax.fill_between(stride_pct, neg_con_ach, neg_con_mus, facecolor=NOTION_CON_MUS_FACE, edgecolor=NOTION_CON_MUS_EDGE, hatch='////', linewidth=0.5)

    # --- 3. Net System Power Line ---
    ax.plot(stride_pct, tot_sys, color="#111827", linestyle=":", linewidth=2.5, label="Net System Power", zorder=10)

    # --- 4. Draw Centered Text Labels ---
    stroke = [pe.withStroke(linewidth=3, foreground=NOTION_BG, alpha=0.8)]
    
    def label_blocks(sys_curve, exo_curve, ach_curve, mus_curve, 
                     raw_exo, raw_ach, raw_mus, 
                     pos_exo_base, pos_ach_base, pos_mus_base, 
                     neg_exo_base, neg_ach_base, neg_mus_base, 
                     t_color_exo, t_color_ach, t_color_mus):
        
        crossings = find_zero_crossings(sys_curve)
        boundaries = [0] + [c + 1 for c in crossings] + [len(sys_curve)]
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if (end - start) < 5: continue
            
            exo_j_list = [np.trapz(raw_exo[s_idx][start:end], dx=dt) for s_idx in range(len(raw_exo))]
            ach_j_list = [np.trapz(raw_ach[s_idx][start:end], dx=dt) for s_idx in range(len(raw_ach))]
            mus_j_list = [np.trapz(raw_mus[s_idx][start:end], dx=dt) for s_idx in range(len(raw_mus))]
            
            m_exo_j, s_exo_j = np.mean(exo_j_list), np.std(exo_j_list)
            m_ach_j, s_ach_j = np.mean(ach_j_list), np.std(ach_j_list)
            m_mus_j, s_mus_j = np.mean(mus_j_list), np.std(mus_j_list)
            
            peak_idx = start + np.argmax(np.abs(sys_curve[start:end]))
            cx = stride_pct[peak_idx]
            
            p_e = exo_curve[peak_idx]
            p_a = ach_curve[peak_idx]
            p_m = mus_curve[peak_idx]
            
            y_exo = pos_exo_base[peak_idx] + p_e/2 if p_e > 0 else neg_exo_base[peak_idx] + p_e/2
            y_ach = pos_ach_base[peak_idx] + p_a/2 if p_a > 0 else neg_ach_base[peak_idx] + p_a/2
            y_mus = pos_mus_base[peak_idx] + p_m/2 if p_m > 0 else neg_mus_base[peak_idx] + p_m/2
                
            if abs(m_exo_j) >= 0.5 and abs(p_e) >= 4.0:
                ax.text(cx, y_exo, f"{m_exo_j:+.1f} J\n±{s_exo_j:.1f} J", color=t_color_exo, fontsize=9, fontweight='bold', ha='center', va='center', path_effects=stroke, zorder=15)
            if abs(m_ach_j) >= 0.5 and abs(p_a) >= 4.0:
                ax.text(cx, y_ach, f"{m_ach_j:+.1f} J\n±{s_ach_j:.1f} J", color=t_color_ach, fontsize=9, fontweight='bold', ha='center', va='center', path_effects=stroke, zorder=15)
            if abs(m_mus_j) >= 0.5 and abs(p_m) >= 4.0:
                ax.text(cx, y_mus, f"{m_mus_j:+.1f} J\n±{s_mus_j:.1f} J", color=t_color_mus, fontsize=9, fontweight='bold', ha='center', va='center', path_effects=stroke, zorder=15)

    label_blocks(ref_sys, ref_exo, ref_ach, ref_mus, 
                 raw_ref_exo, raw_ref_ach, raw_ref_mus,
                 pos_exo_base=z, pos_ach_base=pos_ref_exo, pos_mus_base=pos_ref_ach, 
                 neg_exo_base=z, neg_ach_base=neg_ref_exo, neg_mus_base=neg_ref_ach, 
                 t_color_exo=TEXT_COLOR_EXO, t_color_ach=TEXT_COLOR_ACH, t_color_mus=TEXT_COLOR_MUS)
    
    label_blocks(con_sys, con_exo, con_ach, con_mus, 
                 raw_con_exo, raw_con_ach, raw_con_mus,
                 pos_exo_base=pos_ref_mus, pos_ach_base=pos_con_exo, pos_mus_base=pos_con_ach, 
                 neg_exo_base=neg_ref_mus, neg_ach_base=neg_con_exo, neg_mus_base=neg_con_ach, 
                 t_color_exo=TEXT_COLOR_EXO, t_color_ach=TEXT_COLOR_ACH, t_color_mus=TEXT_COLOR_MUS)

    # --- 5. Custom X-Axis formatting (Ticks and Foot Icons) ---
    ax.set_xlim(0, 100) 
    
    duty_pct = mean_duty_factor * 100
    contra_hs = 50.0  # Idealized 50% offset
    contra_to = (duty_pct + 50.0) % 100.0

    ax.set_xticks([0, contra_to, contra_hs, duty_pct, 100])
    ax.set_xticklabels([]) 
    
    # Tick lines dropping down capped 10% above the peak to avoid legend overlap
    line_top = max_power * 1.10
    
    ax.vlines([0, duty_pct, 100], ymin=y_lower, ymax=line_top, colors='black', linewidths=1.5, linestyles='-', zorder=5)
    ax.vlines([contra_to, contra_hs], ymin=y_lower, ymax=line_top, colors=NOTION_SUBTEXT, linewidths=1.5, linestyles='--', alpha=0.8, zorder=5)

    # Place Foot Icons & Labels
    icon_y_pos = -0.12
    text_y_pos = -0.22
    
    # 1. Reference Heel-Strike (0%)
    ax.scatter(0, icon_y_pos, s=700, marker=get_rotated_foot_marker(-25), facecolors=NOTION_TEXT, edgecolors='none', transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(0, text_y_pos, "Heel-Strike", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=11, fontweight='500')
    
    # 2. Contralateral Toe-Off (~10-15%)
    ax.scatter(contra_to, icon_y_pos, s=700, marker=get_rotated_foot_marker(30), facecolors='none', edgecolors=NOTION_SUBTEXT, linestyles='--', linewidths=1.5, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(contra_to, text_y_pos, "Contra\nToe-Off", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_SUBTEXT, fontsize=10, fontweight='500')
    
    # 3. Contralateral Heel-Strike (50%)
    ax.scatter(contra_hs, icon_y_pos, s=700, marker=get_rotated_foot_marker(-25), facecolors='none', edgecolors=NOTION_SUBTEXT, linestyles='--', linewidths=1.5, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(contra_hs, text_y_pos, "Contra\nHeel-Strike", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_SUBTEXT, fontsize=10, fontweight='500')

    # 4. Reference Toe-Off (duty_pct)
    ax.scatter(duty_pct, icon_y_pos, s=700, marker=get_rotated_foot_marker(30), facecolors=NOTION_TEXT, edgecolors='none', transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(duty_pct, text_y_pos, "Toe-Off", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=11, fontweight='500')

    # 5. Reference Heel-Strike (100%)
    ax.scatter(100, icon_y_pos, s=700, marker=get_rotated_foot_marker(-25), facecolors=NOTION_TEXT, edgecolors='none', transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(100, text_y_pos, "Heel-Strike", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=11, fontweight='500')

    # X-Axis Title
    ax.text(0.5, -0.38, "Full Stride Cycle (%)", transform=ax.transAxes, ha='center', va='top', color=NOTION_SUBTEXT, fontsize=11, fontweight='bold')
    
    ax.set_ylabel("Mechanical Power (W)", fontsize=11, fontweight='bold', color=NOTION_TEXT)
    ax.axhline(0, color=NOTION_TEXT, linewidth=1.5) 
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
            
    ax.legend(unique_handles[::-1], unique_labels[::-1], frameon=False, loc="upper left", 
              fontsize=10, labelcolor=NOTION_TEXT, bbox_to_anchor=(0, 1.08), ncol=3)

    # --- 6. Plot Floating Metabolic Cost Estimates Box ---
    bio_text = f"{net_bio_watts:.1f} W" if net_bio_watts is not None else "N/A"
    summary_str = (
        f"Metabolic Cost Estimates\n"
        f"Mechanical Muscle Cost:  {est_metabolic_watts:.1f} W\n"
        f"Respirometry Mask (Net):  {bio_text}"
    )
    ax.text(0.98, 0.95, summary_str, transform=ax.transAxes,
            fontsize=10, va='top', ha='right', color=NOTION_TEXT,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=NOTION_BG, edgecolor='#EDEDED', alpha=0.9))
    
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

        # Safely extract respirometry data and compute Biological Power (Weir equation)
        vo2_col = next((c for c in df.columns if c.lower() == 'vo2'), None)
        vco2_col = next((c for c in df.columns if c.lower() == 'vco2'), None)
        
        net_bio_watts = None
        if vo2_col:
            vo2_mean = df[vo2_col].replace(0, np.nan).dropna().mean()
            vco2_mean = df[vco2_col].replace(0, np.nan).dropna().mean() if vco2_col else 0.85 * vo2_mean
            
            if not np.isnan(vo2_mean) and vo2_mean > 0:
                # Weir Equation: cal/min = 3.941 * VO2(mL/min) + 1.106 * VCO2(mL/min)
                # Convert to Watts: 1 cal = 4.184 J, 1 min = 60s
                cal_per_min = 3.941 * vo2_mean + 1.106 * vco2_mean
                bio_watts = cal_per_min * 4.184 / 60.0
                net_bio_watts = bio_watts - 70.0  # Subtract 70W assumed basal/standing rate
                
                print(f"  Respirometry Data Found:")
                print(f"    - VO2: {vo2_mean:.1f} mL/min | VCO2: {vco2_mean:.1f} mL/min")
                print(f"    - Gross Biological Power: {bio_watts:.1f} W")
                print(f"    - Net Biological Power (Gross - 70W): {net_bio_watts:.1f} W")

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
            plot_tufte_symmetric_stride(filename, aggs, mean_stride_dur, mean_duty_factor, net_bio_watts)

if __name__ == "__main__":
    main()