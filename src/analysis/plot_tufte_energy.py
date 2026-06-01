# File: src/analysis/plot_tufte_energy.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path

# --- Tufte & Notion Aesthetic Configuration ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Inter', 
    '-apple-system', 
    'BlinkMacSystemFont', 
    'Segoe UI', 
    'Helvetica Neue', 
    'Helvetica', 
    'Arial', 
    'sans-serif'
]

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"

# Adjusted colors for the 3-layer stack
NOTION_EXO_COLOR = "#D3E5EF"      # Soft blue
NOTION_ACHILLES_COLOR = "#EBE0C5" # Lighter, elastic golden-beige
NOTION_MUSCLE_COLOR = "#D4B89F"   # Deeper, active reddish-brown

# Deeper shades for text legibility inside the fills
TEXT_EXO_COLOR = "#3A637A"
TEXT_ACHILLES_COLOR = "#9E814D"
TEXT_MUSCLE_COLOR = "#7A583A"

def find_zero_crossings(y):
    """Find indices where the array crosses zero."""
    return np.where(np.diff(np.sign(y)))[0]

def get_rotated_foot_marker(angle_deg):
    """Returns a Matplotlib Path object of a footprint pointing LEFT."""
    verts = np.array([
        [ 0.3, -0.15],  [-0.3, -0.15],  [-0.45, -0.05],
        [-0.4,  0.05],  [-0.1,  0.15],  [ 0.1,  0.45],
        [ 0.3,  0.45],  [ 0.4,  0.1],   [ 0.3, -0.15]
    ])
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return Path(np.dot(verts, R.T), [Path.MOVETO] + [Path.LINETO]*7 + [Path.CLOSEPOLY])

def extract_biological_components(time, human_power):
    """
    Splits human power into 'Achilles' (balanced zero-net energy) and 'Muscle'.
    Achilles is bounded to late stance (>50% of stride) and balances the final 
    positive push-off lump with the preceding negative lump.
    """
    achilles_power = np.zeros_like(human_power)
    
    # 1. Find boundaries of power lumps
    crossings = find_zero_crossings(human_power)
    boundaries = [0] + list(crossings) + [len(human_power)-1]
    
    last_pos_start, last_pos_end = -1, -1
    last_neg_start, last_neg_end = -1, -1
    
    # 2. Iterate backwards to find the final positive lump and preceding negative lump
    for i in range(len(boundaries)-2, -1, -1):
        start, end = boundaries[i], boundaries[i+1]
        if np.mean(human_power[start:end]) > 0:
            last_pos_start, last_pos_end = start, end
            if i > 0:
                last_neg_start, last_neg_end = boundaries[i-1], boundaries[i]
            break
            
    if last_pos_start != -1 and last_neg_start != -1:
        # Enforce 50% stance phase rule for the negative (storage) lump
        idx_50 = len(time) // 2
        if last_neg_start < idx_50:
            last_neg_start = idx_50
            
        if last_neg_start < last_neg_end:
            t_pos = time[last_pos_start:last_pos_end]
            t_neg = time[last_neg_start:last_neg_end]
            pos_chunk = human_power[last_pos_start:last_pos_end]
            neg_chunk = human_power[last_neg_start:last_neg_end]
            
            e_pos = np.trapz(pos_chunk, t_pos)
            e_neg = np.trapz(neg_chunk, t_neg) # Negative value
            
            # 3. Balance the energy to find the Achilles contribution
            if e_pos > 0 and e_neg < 0:
                achilles_energy = min(e_pos, abs(e_neg))
                
                scale_pos = achilles_energy / e_pos
                scale_neg = achilles_energy / abs(e_neg)
                
                achilles_power[last_pos_start:last_pos_end] = pos_chunk * scale_pos
                achilles_power[last_neg_start:last_neg_end] = neg_chunk * scale_neg
                
    muscle_power = human_power - achilles_power
    return muscle_power, achilles_power

def plot_tufte_energy_loops(cop, time, human_power, exo_power):
    """
    Generates a Tufte-style stacked area chart of Exo, Achilles, and Muscle power.
    """
    # Decompose biological power
    muscle_power, achilles_power = extract_biological_components(time, human_power)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    
    # Titles
    fig.text(0.04, 0.93, "Stance Phase Energy Distribution", fontsize=18, fontweight='bold', color=NOTION_TEXT)
    fig.text(0.04, 0.88, "Exoskeleton vs. Biological Work (Muscle + Achilles) across CoP trajectory", fontsize=12, color=NOTION_SUBTEXT)
    plt.subplots_adjust(top=0.80, bottom=0.25)

    # Clean up spines
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    ax.tick_params(axis='y', colors=NOTION_TEXT, length=0)
    ax.tick_params(axis='x', bottom=False, labelbottom=False) 
    ax.grid(axis='y', color=NOTION_GRID, linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)

    # Separate into positive/negative components for stacking
    exo_pos = np.maximum(exo_power, 0)
    exo_neg = np.minimum(exo_power, 0)
    ach_pos = np.maximum(achilles_power, 0)
    ach_neg = np.minimum(achilles_power, 0)
    mus_pos = np.maximum(muscle_power, 0)
    mus_neg = np.minimum(muscle_power, 0)

    # --- Plot Stacked Areas ---
    # Positive Stacks (0 -> Exo -> Achilles -> Muscle)
    ax.fill_between(cop, 0, exo_pos, color=NOTION_EXO_COLOR, alpha=0.9, label="Exo Power")
    ax.fill_between(cop, exo_pos, exo_pos + ach_pos, color=NOTION_ACHILLES_COLOR, alpha=0.9, label="Likely Achilles")
    ax.fill_between(cop, exo_pos + ach_pos, exo_pos + ach_pos + mus_pos, color=NOTION_MUSCLE_COLOR, alpha=0.9, label="Muscle Work")
    
    # Negative Stacks
    ax.fill_between(cop, 0, exo_neg, color=NOTION_EXO_COLOR, alpha=0.9)
    ax.fill_between(cop, exo_neg, exo_neg + ach_neg, color=NOTION_ACHILLES_COLOR, alpha=0.9)
    ax.fill_between(cop, exo_neg + ach_neg, exo_neg + ach_neg + mus_neg, color=NOTION_MUSCLE_COLOR, alpha=0.9)

    # --- Label "Lumps" ---
    total_power = exo_power + achilles_power + muscle_power
    crossings = find_zero_crossings(total_power)
    boundaries = [0] + list(crossings) + [len(total_power)-1]
    
    stroke = [pe.withStroke(linewidth=3, foreground=NOTION_BG)]
    MIN_FIT_WATTS = 6.0 
    
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if (end - start) < 5:
            continue
            
        chunk_total = total_power[start:end]
        chunk_time = time[start:end]
        
        # Calculate Energies (Joules)
        j_exo = np.trapz(exo_power[start:end], chunk_time)
        j_ach = np.trapz(achilles_power[start:end], chunk_time)
        j_mus = np.trapz(muscle_power[start:end], chunk_time)
        
        # Find peak CoP for alignment
        peak_idx = np.argmax(np.abs(chunk_total))
        cx = cop[start:end][peak_idx]
        
        # Local powers at the peak CoP
        p_e = exo_power[start:end][peak_idx]
        p_a = achilles_power[start:end][peak_idx]
        p_m = muscle_power[start:end][peak_idx]
        
        # Calculate geometric Y-centers for each stack layer
        y_exo = p_e / 2
        y_ach = p_e + (p_a / 2)
        y_mus = p_e + p_a + (p_m / 2)
            
        # Draw Labels conditionally
        if abs(j_exo) >= 0.5 and abs(p_e) >= MIN_FIT_WATTS:
            ax.text(cx, y_exo, f"{j_exo:+.1f} J\n± {abs(j_exo)*0.15:.1f} J", color=TEXT_EXO_COLOR, 
                    fontsize=9, fontweight='bold', ha='center', va='center', path_effects=stroke)
            
        if abs(j_ach) >= 0.5 and abs(p_a) >= MIN_FIT_WATTS:
            ax.text(cx, y_ach, f"{j_ach:+.1f} J\n± {abs(j_ach)*0.15:.1f} J", color=TEXT_ACHILLES_COLOR, 
                    fontsize=9, fontweight='bold', ha='center', va='center', path_effects=stroke)
                    
        if abs(j_mus) >= 0.5 and abs(p_m) >= MIN_FIT_WATTS:
            ax.text(cx, y_mus, f"{j_mus:+.1f} J\n± {abs(j_mus)*0.15:.1f} J", color=TEXT_MUSCLE_COLOR, 
                    fontsize=9, fontweight='bold', ha='center', va='center', path_effects=stroke)

    # 5. X-Axis Formatting
    ax.set_xlim(max(cop), min(cop)) 
    
    hs_x = max(cop) - 0.02
    ms_x = np.mean(cop)
    to_x = min(cop) + 0.02

    for x, angle, label in [(hs_x, -25, "Heel-Strike"), (ms_x, 0, "Mid-Stance"), (to_x, 30, "Toe-Off")]:
        ax.scatter(x, -0.06, s=700, marker=get_rotated_foot_marker(angle), color=NOTION_TEXT, 
                   transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(x, -0.13, label, transform=ax.get_xaxis_transform(), 
                ha='center', va='top', color=NOTION_TEXT, fontsize=11, fontweight='500')
                
    ax.text(0.5, -0.26, "Center of Pressure along Treadmill Belt (m)", 
            transform=ax.transAxes, ha='center', va='top', color=NOTION_SUBTEXT, fontsize=11, fontweight='bold')
    
    # 6. Y-Axis Label and styling
    ax.set_ylabel("Mechanical Power (W)", fontsize=11, fontweight='bold', color=NOTION_TEXT)
    ax.axhline(0, color=NOTION_TEXT, linewidth=1.5) 
    
    # Legend ordering: Exo (bottom), Achilles (middle), Muscle (top)
    handles, labels = ax.get_legend_handles_labels()
    
    # Deduplicate legend items (caused by separate positive/negative fills)
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
            
    # Reverse lists using slicing [::-1] so Muscle is on top, Exo is on bottom
    ax.legend(unique_handles[::-1], unique_labels[::-1], frameon=False, loc="upper left", 
              fontsize=11, labelcolor=NOTION_TEXT, bbox_to_anchor=(0, 1.05))
    
    plt.show()

if __name__ == "__main__":
    # --- Generate Synthetic Stance-Phase Data ---
    t = np.linspace(0, 0.7, 300) 
    cop = np.linspace(0.15, -0.15, 300)
    
    exo_power = -15 * np.exp(-((t - 0.1) ** 2) / 0.005) + 80 * np.exp(-((t - 0.62) ** 2) / 0.003)    
    
    human_power = (
        -30 * np.exp(-((t - 0.05) ** 2) / 0.002) +  
        25 * np.exp(-((t - 0.2) ** 2) / 0.01) -     
        40 * np.exp(-((t - 0.45) ** 2) / 0.01) +   # Pre-pushoff negative storage
        70 * np.exp(-((t - 0.6) ** 2) / 0.005)     # Pushoff release
    )

    print("Generating Tufte-style Energy Area Chart...")
    plot_tufte_energy_loops(cop, t, human_power, exo_power)