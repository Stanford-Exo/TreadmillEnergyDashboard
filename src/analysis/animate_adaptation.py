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
    from matplotlib.path import Path
except ImportError:
    print("Error: matplotlib is required. Please run: pip install matplotlib")
    sys.exit(1)

DEFAULT_PARQUET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee/precomputed_poggensee.parquet"))

# --- Notion Aesthetics ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', '-apple-system', 'Arial', 'sans-serif']

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"

# Colors for Stance Leg (Solid)
NOTION_REF_EXO = "#D3E5EF"      
NOTION_REF_ACH = "#EBE0C5"  
NOTION_REF_MUS = "#D4B89F"  

# Colors for Swing Leg (Striped)
NOTION_CON_EXO_FACE = "#F2F6F9"
NOTION_CON_EXO_EDGE = "#8EB0C6"
NOTION_CON_ACH_FACE = "#FDFBEF"
NOTION_CON_ACH_EDGE = "#D1C090"
NOTION_CON_MUS_FACE = "#F7F2EE"
NOTION_CON_MUS_EDGE = "#B59475"


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
    parser.add_argument("--trial", type=str, default="Static_Training_1_Adaptation_adaptation_Day3_ADAPT1", 
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

    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    plt.subplots_adjust(top=0.78, bottom=0.22, left=0.08, right=0.92)

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

        pos_ref_mus = np.maximum(ref_exo, 0) + np.maximum(ref_ach, 0) + np.maximum(ref_mus, 0)
        pos_con_mus = pos_ref_mus + np.maximum(con_exo, 0) + np.maximum(con_ach, 0) + np.maximum(con_mus, 0)
        neg_ref_mus = np.minimum(ref_exo, 0) + np.minimum(ref_ach, 0) + np.minimum(ref_mus, 0)
        neg_con_mus = neg_ref_mus + np.minimum(con_exo, 0) + np.minimum(con_ach, 0) + np.minimum(con_mus, 0)

        y_max_global = max(y_max_global, np.max(pos_con_mus))
        y_min_global = min(y_min_global, np.min(neg_con_mus))

    y_upper = y_max_global * 1.5   # 50% headroom for legend + metrics card
    y_lower = y_min_global * 1.15  # 15% footroom
    
    # 3. Animation Update Function
    def update(frame_idx):
        ax.clear()
        row = df_trial.iloc[frame_idx]

        # Extract 1D mean arrays
        ref_exo = np.array([row[f'ref_exo_w_{i:02d}'] for i in range(100)])
        ref_ach = np.array([row[f'ref_ach_w_{i:02d}'] for i in range(100)])
        ref_mus = np.array([row[f'ref_mus_w_{i:02d}'] for i in range(100)])
        con_exo = np.array([row[f'con_exo_w_{i:02d}'] for i in range(100)])
        con_ach = np.array([row[f'con_ach_w_{i:02d}'] for i in range(100)])
        con_mus = np.array([row[f'con_mus_w_{i:02d}'] for i in range(100)])

        # Standard Deviation arrays (for transparency ribbons)
        ref_mus_std = np.array([row[f'ref_mus_std_{i:02d}'] for i in range(100)])
        con_mus_std = np.array([row[f'con_mus_std_{i:02d}'] for i in range(100)])

        # Positive Fills Stacking Order
        pos_ref_exo = np.maximum(ref_exo, 0)
        pos_ref_ach = pos_ref_exo + np.maximum(ref_ach, 0)
        pos_ref_mus = pos_ref_ach + np.maximum(ref_mus, 0)
        
        pos_con_exo = pos_ref_mus + np.maximum(con_exo, 0)
        pos_con_ach = pos_con_exo + np.maximum(con_ach, 0)
        pos_con_mus = pos_con_ach + np.maximum(con_mus, 0)

        # Negative Fills Stacking Order
        neg_ref_exo = np.minimum(ref_exo, 0)
        neg_ref_ach = neg_ref_exo + np.minimum(ref_ach, 0)
        neg_ref_mus = neg_ref_ach + np.minimum(ref_mus, 0)
        
        neg_con_exo = neg_ref_mus + np.minimum(con_exo, 0)
        neg_con_ach = neg_con_exo + np.minimum(con_ach, 0)
        neg_con_mus = neg_con_ach + np.minimum(con_mus, 0)

        # Plot Stance Leg Solid Fills
        ax.fill_between(stride_pct, z, pos_ref_exo, color=NOTION_REF_EXO, alpha=0.9, linewidth=0, label="Stance Leg Exo")
        ax.fill_between(stride_pct, pos_ref_exo, pos_ref_ach, color=NOTION_REF_ACH, alpha=0.9, linewidth=0, label="Stance Leg Achilles")
        ax.fill_between(stride_pct, pos_ref_ach, pos_ref_mus, color=NOTION_REF_MUS, alpha=0.9, linewidth=0, label="Stance Leg Muscle")

        # Plot Swing Leg Hatched Fills
        ax.fill_between(stride_pct, pos_ref_mus, pos_con_exo, facecolor=NOTION_CON_EXO_FACE, edgecolor=NOTION_CON_EXO_EDGE, hatch='////', linewidth=0.5, label="Swing Leg Exo")
        ax.fill_between(stride_pct, pos_con_exo, pos_con_ach, facecolor=NOTION_CON_ACH_FACE, edgecolor=NOTION_CON_ACH_EDGE, hatch='////', linewidth=0.5, label="Swing Leg Achilles")
        ax.fill_between(stride_pct, pos_con_ach, pos_con_mus, facecolor=NOTION_CON_MUS_FACE, edgecolor=NOTION_CON_MUS_EDGE, hatch='////', linewidth=0.5, label="Swing Leg Muscle")

        # Plot Negative Solid Fills
        ax.fill_between(stride_pct, z, neg_ref_exo, color=NOTION_REF_EXO, alpha=0.9, linewidth=0)
        ax.fill_between(stride_pct, neg_ref_exo, neg_ref_ach, color=NOTION_REF_ACH, alpha=0.9, linewidth=0)
        ax.fill_between(stride_pct, neg_ref_ach, neg_ref_mus, color=NOTION_REF_MUS, alpha=0.9, linewidth=0)

        # Plot Negative Hatched Fills
        ax.fill_between(stride_pct, neg_ref_mus, neg_con_exo, facecolor=NOTION_CON_EXO_FACE, edgecolor=NOTION_CON_EXO_EDGE, hatch='////', linewidth=0.5)
        ax.fill_between(stride_pct, neg_con_exo, neg_con_ach, facecolor=NOTION_CON_ACH_FACE, edgecolor=NOTION_CON_ACH_EDGE, hatch='////', linewidth=0.5)
        ax.fill_between(stride_pct, neg_con_ach, neg_con_mus, facecolor=NOTION_CON_MUS_FACE, edgecolor=NOTION_CON_MUS_EDGE, hatch='////', linewidth=0.5)

        # Plot Standard Deviation ribbons on top
        ax.fill_between(stride_pct, pos_ref_mus - ref_mus_std, pos_ref_mus + ref_mus_std, color=NOTION_REF_MUS, alpha=0.15, linewidth=0)
        ax.fill_between(stride_pct, pos_con_mus - con_mus_std, pos_con_mus + con_mus_std, color=NOTION_CON_MUS_EDGE, alpha=0.1, linewidth=0)

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

        # Static legends placement
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], frameon=False, loc="upper left", fontsize=9, labelcolor=NOTION_TEXT, ncol=3)

        # 4. Floating Adaption Metric Card
        window_start = row['window_start_s']
        window_end = window_start + 300.0  # Assuming 5-minute blocks
        
        metrics_text = (
            f"Adaptation Metrics\n"
            f"────────────────────────\n"
            f"Active Period:  {format_time(window_start)} - {format_time(window_end)}\n"
            f"Net Bio Cost:   {row['net_bio_cost_w']:.1f} W\n"
            f"Est. Muscle:    {row['mechanical_power']:.1f} W\n"
            f"Net Exo Assist: {row['exo_power']:+.1f} W"
        )
        ax.text(0.98, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=9.5, va='top', ha='right', color=NOTION_TEXT, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=NOTION_BG, edgecolor='#EDEDED', alpha=0.95))

        # Big Chronological Indicator in Top Title
        fig.suptitle(f"Chronological Gait Adaptation: Day 3", fontsize=15, fontweight='bold', color=NOTION_TEXT, y=0.96)
        ax.set_title(f"Trial Segment: {format_time(window_start)} - {format_time(window_end)} ({frame_idx + 1}/{num_frames})", 
                     fontsize=11.5, color=NOTION_SUBTEXT, pad=20)

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