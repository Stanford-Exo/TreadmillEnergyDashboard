# File: src/analysis/correlate_metabolics.py

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Setup optional plotting module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    plt = None

# Pointing to the precomputed Wide-Format Parquet file
DEFAULT_PARQUET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee/precomputed_poggensee.parquet"))

# --- Tufte/Notion Aesthetics ---
if plt is not None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', '-apple-system', 'Arial', 'sans-serif']

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"


def process_within_trial_changes(df, use_no_achilles=False):
    """
    Processes the raw windowed dataset to compute within-trial changes (deltas)
    after applying filters to remove metabolic outliers and trial cooldowns.
    """
    delta_data = []
    
    # Analyze on a per-trial basis
    for trial_name, group in df.groupby('trial_name'):
        # 1. Sort sequentially by chronological time
        group = group.sort_values(by='window_start_s').reset_index(drop=True)
        
        # 2. Drop the last two windows (last 10 minutes) to eliminate cooldown/slowdown artifacts
        if len(group) > 2:
            group = group.iloc[:-2].reset_index(drop=True)
        else:
            continue
            
        # 3. Apply absolute bounds to catch mask-off outliers (< 50W or > 600W)
        group = group[(group['net_bio_cost_w'] >= 50.0) & (group['net_bio_cost_w'] <= 600.0)].reset_index(drop=True)
        if group.empty:
            continue
            
        # 4. Filter out extreme transients (deviating from the trial median by 100W or more)
        trial_median_bio = group['net_bio_cost_w'].median()
        group = group[np.abs(group['net_bio_cost_w'] - trial_median_bio) < 100.0].reset_index(drop=True)
        
        if len(group) < 2:
            # We need at least a baseline and one comparison window to evaluate changes
            continue
            
        # 5. Define the first chronological remaining window as the baseline state
        baseline = group.iloc[0]
        
        # Define target column based on the Achilles extraction model
        mech_col = 'mechanical_power_no_achilles' if use_no_achilles else 'mechanical_power'
        
        for _, row in group.iterrows():
            delta_bio = row['net_bio_cost_w'] - baseline['net_bio_cost_w']
            delta_mech = row[mech_col] - baseline[mech_col]
            
            delta_data.append({
                'trial_name': trial_name,
                'window_start_s': row['window_start_s'],
                'delta_bio_w': delta_bio,
                'delta_mech_w': delta_mech,
                'raw_bio_w': row['net_bio_cost_w'],
                'raw_mech_w': row[mech_col]
            })
            
    return pd.DataFrame(delta_data)


def plot_delta_correlation(df_deltas, use_no_achilles=False):
    """
    Plots the scatter correlation between the change in Estimated Mechanical Power 
    and the change in Measured Net Biological Cost relative to trial baselines.
    """
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    
    # Title & Subtitle based on the Achilles tendon configuration
    if use_no_achilles:
        title = "Within-Trial Changes: Delta Mechanics (NO Achilles Model) vs. Delta Metabolics"
        subtitle = "Correlating within-subject adaptation savings relative to the first valid window (cooldowns & outliers excluded)."
        x_label = "Change in Est. Mechanical Cost ($\Delta$ Mechanical Power) (W)"
    else:
        title = "Within-Trial Changes: Delta Mechanics (Achilles Extracted) vs. Delta Metabolics"
        subtitle = "Correlating within-subject adaptation savings relative to the first valid window (cooldowns & outliers excluded)."
        x_label = "Change in Est. Mechanical Muscle Cost ($\Delta$ Muscle Power) (W)"

    fig.text(0.05, 0.93, title, fontsize=14, fontweight='bold', color=NOTION_TEXT)
    fig.text(0.05, 0.89, subtitle, fontsize=10, color=NOTION_SUBTEXT)
    plt.subplots_adjust(top=0.82, right=0.95, bottom=0.15)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(NOTION_TEXT)
        
    ax.tick_params(axis='both', colors=NOTION_TEXT, length=4)
    ax.grid(color=NOTION_GRID, linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)

    x_vals = df_deltas['delta_mech_w'].values
    y_vals = df_deltas['delta_bio_w'].values

    # Color coordinate different trials to highlight within-subject trajectory groupings
    unique_trials = df_deltas['trial_name'].unique()
    colormap = cm.get_cmap('tab10', max(len(unique_trials), 10))
    
    for i, trial_name in enumerate(unique_trials):
        sub_df = df_deltas[df_deltas['trial_name'] == trial_name]
        ax.scatter(sub_df['delta_mech_w'], sub_df['delta_bio_w'], 
                   color=colormap(i), label=trial_name, s=70, alpha=0.8, 
                   edgecolors='white', linewidths=0.5, zorder=3)

    # Reference cross-hairs at delta origin (0, 0)
    ax.axhline(0, color=NOTION_SUBTEXT, linestyle=':', linewidth=0.8, zorder=1)
    ax.axvline(0, color=NOTION_SUBTEXT, linestyle=':', linewidth=0.8, zorder=1)

    # Perform linear regression on changes
    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    
    min_x, max_x = np.min(x_vals), np.max(x_vals)
    pad_x = (max_x - min_x) * 0.1 if max_x > min_x else 1.0
    x_line = np.linspace(min_x - pad_x, max_x + pad_x, 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, color=NOTION_TEXT, linewidth=2.0, zorder=2, 
            label=f"Linear Fit (y = {slope:.2f}x {'+' if intercept>=0 else '-'} {abs(intercept):.1f})")

    stats_text = (
        f"$R^2$: {r_value**2:.3f}\n"
        f"Pearson $r$: {r_value:.3f}\n"
        f"p-value: {p_value:.2e}\n"
        f"N: {len(df_deltas)} segments"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', color=NOTION_TEXT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=NOTION_BG, edgecolor='#EDEDED', alpha=0.9))

    ax.set_xlabel(x_label, fontsize=10, fontweight='bold', color=NOTION_TEXT)
    ax.set_ylabel("Change in Net Measured Biological Cost ($\Delta$ Metabolics) (W)", fontsize=10, fontweight='bold', color=NOTION_TEXT)
    
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    
    ax.legend(frameon=True, facecolor=NOTION_BG, edgecolor=NOTION_GRID, fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Correlate within-trial adaptation changes in estimated mechanics vs metabolics.")
    parser.add_argument("--file", type=str, default=DEFAULT_PARQUET_PATH, help="Path to the precomputed_poggensee.parquet file.")
    parser.add_argument("--no-achilles", action="store_true", help="Correlate muscle cost changes assuming NO Achilles tendon storage.")
    args = parser.parse_args()

    if plt is None:
        print("Error: matplotlib required. pip install matplotlib")
        sys.exit(1)

    parquet_path = os.path.abspath(args.file)
    if not os.path.exists(parquet_path):
        print(f"Error: Could not find Parquet file at {parquet_path}")
        print("Please run 'python src/analysis/precompute_poggensee.py' first to generate the data.")
        sys.exit(1)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading Parquet: {e}")
        sys.exit(1)

    if df.empty:
        print(f"The file at {parquet_path} is empty. No data to plot.")
        sys.exit(1)

    # Process and extract within-trial changes relative to first valid windows
    df_deltas = process_within_trial_changes(df, use_no_achilles=args.no_achilles)

    if df_deltas.empty:
        print("Error: No valid within-trial change points remain after filtering. Verify your dataset values.")
        sys.exit(1)

    print(f"Successfully compiled {len(df_deltas)} change segments across {len(df_deltas['trial_name'].unique())} unique trials.")
    plot_delta_correlation(df_deltas, use_no_achilles=args.no_achilles)


if __name__ == "__main__":
    main()