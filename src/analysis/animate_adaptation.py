# File: src/analysis/animate_adaptation.py

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patheffects as pe
    from matplotlib.path import Path
except ImportError:
    print("Error: matplotlib is required. Please run: pip install matplotlib")
    sys.exit(1)

DEFAULT_PARQUET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee/precomputed_poggensee.parquet"))
POGGENSEE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee"))

# --- Notion Aesthetics ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', '-apple-system', 'Arial', 'sans-serif']

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"

# Colors for Stance Leg (Solid)
NOTION_REF_EXO = "#D3E5EF"      
NOTION_REF_MUS = "#D4B89F"  

# Colors for Swing Leg (Extremely Light Solid)
NOTION_CON_EXO_FACE = "#F2F6F9"
NOTION_CON_EXO_EDGE = "#8EB0C6"
NOTION_CON_MUS_FACE = "#F7F2EE"
NOTION_CON_MUS_EDGE = "#B59475"


def find_zero_crossings(y):
    return np.where(np.diff(np.sign(y)))[0]


def get_rotated_foot_marker(angle_deg):
    """Generates a custom path for a human foot marker rotated to specified angle."""
    verts = np.array([
        [ 0.3, -0.15],  [-0.3, -0.15],  [-0.45, -0.05],
        [-0.4,  0.05],  [-0.1,  0.15],  [ 0.1,  0.45],
        [ 0.3,  0.45],  [ 0.4,  0.1],   [ 0.3, -0.15]
    ])
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return Path(np.dot(verts, R.T), [Path.MOVETO] + [Path.LINETO]*7 + [Path.CLOSEPOLY])


def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(description="Animate adaptation changes over time from precomputed gait cycles.")
    parser.add_argument("--file", type=str, default=DEFAULT_PARQUET_PATH, help="Path to the precomputed parquet database.")
    parser.add_argument("--trial", type=str, default="Static_Training_1_Validation_Day1_ZT1", 
                        help="Exact trial name to filter.")
    parser.add_argument("--save", action="store_true", help="Save the output as a GIF instead of showing it interactively.")
    args = parser.parse_args()

    parquet_path = os.path.abspath(args.file)
    if not os.path.exists(parquet_path):
        print(f"Error: Could not find precomputed parquet file at: {parquet_path}")
        print("Please run 'python src/analysis/precompute_poggensee.py' first.")
        sys.exit(1)

    df_all = pd.read_parquet(parquet_path)
    
    # 1. Filter and isolate target adaptation trial
    df_trial = df_all[df_all['trial_name'] == args.trial].copy()
    if df_trial.empty:
        print(f"Error: Trial '{args.trial}' not found in the dataset.")
        available = df_all['trial_name'].unique()
        print("\nAvailable trial names in this file:")
        for name in sorted(available):
            print(f"  - {name}")
        sys.exit(1)

    # Sort sequentially by window start time
    df_trial = df_trial.sort_values(by="window_start_s").reset_index(drop=True)
    num_frames = len(df_trial)
    print(f"Loaded trial: '{args.trial}' with {num_frames} chronological windows.")

    # 2. Setup Figure with Edward Tufte timeline plot on top and stride profiles below
    fig, (ax_top, ax) = plt.subplots(
        2, 1, 
        figsize=(11, 8.5), 
        facecolor=NOTION_BG, 
        gridspec_kw={'height_ratios': [1, 3]}
    )
    ax_top.set_facecolor(NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    plt.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.92, hspace=0.42)

    # Precompute cost improvement (negative Watts change down from the highest observed cost)
    bio_costs = df_trial['net_bio_cost_w'].values
    mech_powers = df_trial['mechanical_power_no_achilles'].values
    times_min = df_trial['window_start_s'].values / 60.0

    bio_improvement = bio_costs - np.max(bio_costs)
    mech_improvement = mech_powers - np.max(mech_powers)

    # Style and plot components of the top timeline axes in Edward Tufte style
    for spine in ['top', 'right', 'left']:
        ax_top.spines[spine].set_visible(False)
    ax_top.spines['bottom'].set_color(NOTION_TEXT)
    ax_top.tick_params(axis='both', colors=NOTION_TEXT, length=3, labelsize=8)
    ax_top.grid(axis='y', color='#EDEDED', linestyle='-', linewidth=0.7)
    ax_top.set_axisbelow(True)

    # Edward Tufte style curves (minimalist, low ink, clear visual priority)
    ax_top.plot(times_min, bio_improvement, color=NOTION_TEXT, linestyle='-', linewidth=1.5, 
                label="Metabolic Improvement", marker='o', markersize=4)
    ax_top.plot(times_min, mech_improvement, color=NOTION_SUBTEXT, linestyle='--', linewidth=1.2, 
                label="Mechanical Improvement", marker='s', markersize=4)
    
    ax_top.set_ylabel("Improvement (W)", fontsize=9, fontweight='bold', color=NOTION_TEXT)
    ax_top.set_xlabel("Trial Time (minutes)", fontsize=8, color=NOTION_SUBTEXT)
    ax_top.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=NOTION_TEXT)

    progress_line = None

    stride_pct = np.linspace(0, 100, 100)
    z = np.zeros(100)

    # Pre-determine absolute Y limits across all windows to avoid a jumping Y-axis scale during animation
    y_max_global = 0.0
    y_min_global = 0.0
    for idx in range(num_frames):
        row = df_trial.iloc[idx]
        ref_exo = np.array([row[f'ref_exo_w_{i:02d}'] for i in range(100)])
        ref_ach = np.array([row[f'ref_ach_w_{i:02d}'] for i in range(100)])
        ref_mus = np.array([row[f'ref_mus_w_{i:02d}'] for i in range(100)])
        con_exo = np.array([row[f'con_exo_w_{i:02d}'] for i in range(100)])
        con_ach = np.array([row[f'con_ach_w_{i:02d}'] for i in range(100)])
        con_mus = np.array([row[f'con_mus_w_{i:02d}'] for i in range(100)])

        ref_hum = ref_mus + ref_ach
        con_hum = con_mus + con_ach

        pos_ref_hum = np.maximum(ref_exo, 0) + np.maximum(ref_hum, 0)
        pos_con_hum = pos_ref_hum + np.maximum(con_exo, 0) + np.maximum(con_hum, 0)
        
        neg_ref_hum = np.minimum(ref_exo, 0) + np.minimum(ref_hum, 0)
        neg_con_hum = neg_ref_hum + np.minimum(con_exo, 0) + np.minimum(con_hum, 0)

        y_max_global = max(y_max_global, np.max(pos_con_hum))
        y_min_global = min(y_min_global, np.min(neg_con_hum))

    y_upper = y_max_global * 1.5   # 50% headroom for legend
    y_lower = y_min_global * 1.15  # 15% footroom
    
    # 3. Animation Update Function
    def update(frame_idx):
        nonlocal progress_line
        if progress_line is not None:
            try:
                progress_line.remove()
            except ValueError:
                pass

        ax.clear()
        row = df_trial.iloc[frame_idx]

        # Draw current chronological progress indicator matching the timeline aesthetics
        current_time_min = row['window_start_s'] / 60.0
        progress_line = ax_top.axvline(current_time_min, color=NOTION_TEXT, linestyle=':', linewidth=1.5, zorder=5)

        # Extract 1D mean arrays
        ref_exo = np.array([row[f'ref_exo_w_{i:02d}'] for i in range(100)])
        ref_ach = np.array([row[f'ref_ach_w_{i:02d}'] for i in range(100)])
        ref_mus = np.array([row[f'ref_mus_w_{i:02d}'] for i in range(100)])
        con_exo = np.array([row[f'con_exo_w_{i:02d}'] for i in range(100)])
        con_ach = np.array([row[f'con_ach_w_{i:02d}'] for i in range(100)])
        con_mus = np.array([row[f'con_mus_w_{i:02d}'] for i in range(100)])

        # Consolidate general human biological power parameters (Achilles is kept inside general human power)
        ref_hum = ref_mus + ref_ach
        con_hum = con_mus + con_ach

        # Positive Fills Stacking Order
        pos_ref_exo = np.maximum(ref_exo, 0)
        pos_ref_hum = pos_ref_exo + np.maximum(ref_hum, 0)
        
        pos_con_exo = pos_ref_hum + np.maximum(con_exo, 0)
        pos_con_hum = pos_con_exo + np.maximum(con_hum, 0)

        # Negative Fills Stacking Order
        neg_ref_exo = np.minimum(ref_exo, 0)
        neg_ref_hum = neg_ref_exo + np.minimum(ref_hum, 0)
        
        neg_con_exo = neg_ref_hum + np.minimum(con_exo, 0)
        neg_con_hum = neg_con_exo + np.minimum(con_hum, 0)

        # Plot Stance Leg Solid Fills
        ax.fill_between(stride_pct, z, pos_ref_exo, color=NOTION_REF_EXO, alpha=0.9, linewidth=0, label="Stance Leg Exo")
        ax.fill_between(stride_pct, pos_ref_exo, pos_ref_hum, color=NOTION_REF_MUS, alpha=0.9, linewidth=0, label="Stance Leg Human")

        # Plot Swing Leg Extremely Light Solid Fills (Subtle bounds, outline only)
        ax.fill_between(stride_pct, pos_ref_hum, pos_con_exo, facecolor=NOTION_CON_EXO_FACE, edgecolor=NOTION_CON_EXO_EDGE, linewidth=0.5, label="Swing Leg Exo")
        ax.fill_between(stride_pct, pos_con_exo, pos_con_hum, facecolor=NOTION_CON_MUS_FACE, edgecolor=NOTION_CON_MUS_EDGE, linewidth=0.5, label="Swing Leg Human")

        # Plot Negative Solid Fills
        ax.fill_between(stride_pct, z, neg_ref_exo, color=NOTION_REF_EXO, alpha=0.9, linewidth=0)
        ax.fill_between(stride_pct, neg_ref_exo, neg_ref_hum, color=NOTION_REF_MUS, alpha=0.9, linewidth=0)

        # Plot Negative Extremely Light Solid Fills
        ax.fill_between(stride_pct, neg_ref_hum, neg_con_exo, facecolor=NOTION_CON_EXO_FACE, edgecolor=NOTION_CON_EXO_EDGE, linewidth=0.5)
        ax.fill_between(stride_pct, neg_con_exo, neg_con_hum, facecolor=NOTION_CON_MUS_FACE, edgecolor=NOTION_CON_MUS_EDGE, linewidth=0.5)

        # Draw centered bump work labels (Joules)
        dt = row['mean_stride_duration_s'] / 100.0
        TEXT_COLOR_EXO = "#3A637A"
        TEXT_COLOR_HUM = "#7A583A"
        stroke = [pe.withStroke(linewidth=3, foreground=NOTION_BG, alpha=0.8)]

        def label_blocks(sys_curve, exo_curve, hum_curve,
                         pos_exo_base, pos_hum_base,
                         neg_exo_base, neg_hum_base,
                         t_color_exo, t_color_hum):
            
            crossings = find_zero_crossings(sys_curve)
            boundaries = [0] + [c + 1 for c in crossings] + [len(sys_curve)]
            
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i+1]
                if (end - start) < 5: 
                    continue
                
                # Integrate curves over the current cycle segment
                exo_j = np.trapz(exo_curve[start:end], dx=dt)
                hum_j = np.trapz(hum_curve[start:end], dx=dt)
                
                # Align coordinate positions to peak segment dynamics
                peak_idx = start + np.argmax(np.abs(sys_curve[start:end]))
                cx = stride_pct[peak_idx]
                
                p_e = exo_curve[peak_idx]
                p_h = hum_curve[peak_idx]
                
                y_exo = pos_exo_base[peak_idx] + p_e/2 if p_e > 0 else neg_exo_base[peak_idx] + p_e/2
                y_hum = pos_hum_base[peak_idx] + p_h/2 if p_h > 0 else neg_hum_base[peak_idx] + p_h/2
                
                if abs(exo_j) >= 0.5 and abs(p_e) >= 4.0:
                    ax.text(cx, y_exo, f"{exo_j:+.1f} J", color=t_color_exo, fontsize=8, fontweight='bold', ha='center', va='center', path_effects=stroke, zorder=15)
                
                if abs(hum_j) >= 0.5 and abs(p_h) >= 4.0:
                    ax.text(cx, y_hum, f"{hum_j:+.1f} J", color=t_color_hum, fontsize=8, fontweight='bold', ha='center', va='center', path_effects=stroke, zorder=15)

        # Stance (Reference) Leg Bump Labels
        ref_sys = ref_exo + ref_hum
        label_blocks(ref_sys, ref_exo, ref_hum,
                     pos_exo_base=z, pos_hum_base=pos_ref_exo,
                     neg_exo_base=z, neg_hum_base=neg_ref_exo,
                     t_color_exo=TEXT_COLOR_EXO, t_color_hum=TEXT_COLOR_HUM)

        # Swing (Contralateral) Leg Bump Labels
        con_sys = con_exo + con_hum
        label_blocks(con_sys, con_exo, con_hum,
                     pos_exo_base=pos_ref_hum, pos_hum_base=pos_con_exo,
                     neg_exo_base=neg_ref_hum, neg_hum_base=neg_con_exo,
                     t_color_exo=TEXT_COLOR_EXO, t_color_hum=TEXT_COLOR_HUM)

        # Layout styling
        ax.set_ylim(y_lower, y_upper)
        ax.set_xlim(0, 100)
        ax.axhline(0, color=NOTION_TEXT, linewidth=1.5)

        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', colors=NOTION_TEXT, length=0)
        ax.grid(axis='y', color='#EDEDED', linestyle='-', linewidth=1.0)
        ax.set_axisbelow(True)

        # Custom dividing markers for Foot Timing
        mean_duty_factor = row['mean_duty_factor']
        duty_pct = mean_duty_factor * 100
        contra_hs = 50.0
        contra_to = (duty_pct + 50.0) % 100.0

        ax.set_xticks([0, contra_to, contra_hs, duty_pct, 100])
        ax.set_xticklabels([])

        ax.vlines([0, duty_pct, 100], ymin=y_lower, ymax=y_upper * 0.75, colors='black', linewidths=1.2)
        ax.vlines([contra_to, contra_hs], ymin=y_lower, ymax=y_upper * 0.75, colors=NOTION_SUBTEXT, linewidths=1.2, linestyles='--')

        # Rotation Angles for custom foot symbols
        ax.scatter(0, -0.1, s=450, marker=get_rotated_foot_marker(-25), facecolors=NOTION_TEXT, edgecolors='none', transform=ax.get_xaxis_transform(), clip_on=False)
        ax.scatter(contra_to, -0.1, s=450, marker=get_rotated_foot_marker(30), facecolors='none', edgecolors=NOTION_SUBTEXT, linestyles='--', linewidths=1.2, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.scatter(contra_hs, -0.1, s=450, marker=get_rotated_foot_marker(-25), facecolors='none', edgecolors=NOTION_SUBTEXT, linestyles='--', linewidths=1.2, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.scatter(duty_pct, -0.1, s=450, marker=get_rotated_foot_marker(30), facecolors=NOTION_TEXT, edgecolors='none', transform=ax.get_xaxis_transform(), clip_on=False)
        ax.scatter(100, -0.1, s=450, marker=get_rotated_foot_marker(-25), facecolors=NOTION_TEXT, edgecolors='none', transform=ax.get_xaxis_transform(), clip_on=False)

        # Label Texts on X Axis
        ax.text(0, -0.18, "Stance HS", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=9, fontweight='500')
        ax.text(contra_to, -0.18, "Swing TO", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_SUBTEXT, fontsize=8, fontweight='500')
        ax.text(contra_hs, -0.18, "Swing HS", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_SUBTEXT, fontsize=8, fontweight='500')
        ax.text(duty_pct, -0.18, "Stance TO", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=9, fontweight='500')
        ax.text(100, -0.18, "Stance HS", transform=ax.get_xaxis_transform(), ha='center', va='top', color=NOTION_TEXT, fontsize=9, fontweight='500')

        ax.set_ylabel("Mechanical Power (W)", fontsize=11, fontweight='bold', color=NOTION_TEXT)
        ax.text(0.5, -0.3, "Full Stride Cycle (%)", transform=ax.transAxes, ha='center', va='top', color=NOTION_SUBTEXT, fontsize=10, fontweight='bold')

        # Static legends placement (Adjusted to 2 columns given the reduced category counts)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], frameon=False, loc="upper left", fontsize=9, labelcolor=NOTION_TEXT, ncol=2)

        # Big Chronological Indicator in Top Title
        fig.suptitle(f"Chronological Gait Adaptation: Day 5", fontsize=15, fontweight='bold', color=NOTION_TEXT, y=0.96)

    # 5. Build and Save/Show Animation
    # Set interval to 2000ms (2 seconds per 5-minute chunk) to allow observation of shifts
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=2000, repeat=True)

    if args.save:
        gif_path = os.path.join(POGGENSEE_DIR, f"{args.trial}_adaptation.gif")
        print(f"Saving animation to {gif_path}...")
        try:
            # Pillow is a default writer that doesn't require ffmpeg
            ani.save(gif_path, writer='pillow', fps=0.5)
            print("Successfully saved!")
        except Exception as e:
            print(f"Failed to save GIF: {e}")
    else:
        print("Rendering interactive window. Close the plot to finish.")
        plt.show()


if __name__ == "__main__":
    main()