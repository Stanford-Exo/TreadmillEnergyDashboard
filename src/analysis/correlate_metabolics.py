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


def get_raw_work_rates(row, use_no_achilles=False):
    """
    Reconstructs raw positive and negative mechanical power rates (unmultiplied)
    from the 100 stride profile columns stored in the parquet database.
    """
    dt = row['mean_stride_duration_s'] / 100.0
    
    # Reconstruct the continuous power curves across 100 stride points
    if use_no_achilles:
        # Achilles tendon storage is treated as active muscle power
        ref_curve = np.array([row[f'ref_mus_w_{i:02d}'] + row[f'ref_ach_w_{i:02d}'] for i in range(100)])
        con_curve = np.array([row[f'con_mus_w_{i:02d}'] + row[f'con_ach_w_{i:02d}'] for i in range(100)])
    else:
        # Achilles tendon storage is extracted as passive/metabolically free
        ref_curve = np.array([row[f'ref_mus_w_{i:02d}'] for i in range(100)])
        con_curve = np.array([row[f'con_mus_w_{i:02d}'] for i in range(100)])
        
    # Integrate positive and negative work cycles
    j_pos_ref = np.trapz(np.maximum(ref_curve, 0), dx=dt)
    j_neg_ref = np.trapz(np.minimum(ref_curve, 0), dx=dt)
    
    j_pos_con = np.trapz(np.maximum(con_curve, 0), dx=dt)
    j_neg_con = np.trapz(np.minimum(con_curve, 0), dx=dt)
    
    # Standardize to average power rates (Watts)
    pos_rate = (j_pos_ref + j_pos_con) / row['mean_stride_duration_s']
    neg_rate = (abs(j_neg_ref) + abs(j_neg_con)) / row['mean_stride_duration_s']
    
    return pos_rate, neg_rate


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
            
        # 5. Extract raw positive/negative mechanical work rates
        pos_rates = []
        neg_rates = []
        for _, row in group.iterrows():
            pos_p, neg_p = get_raw_work_rates(row, use_no_achilles=use_no_achilles)
            pos_rates.append(pos_p)
            neg_rates.append(neg_p)
        group['raw_pos_mech_w'] = pos_rates
        group['raw_neg_mech_w'] = neg_rates

        # Define the first chronological remaining window as the baseline state
        baseline = group.iloc[0]
        
        for _, row in group.iterrows():
            delta_bio = row['net_bio_cost_w'] - baseline['net_bio_cost_w']
            delta_pos = row['raw_pos_mech_w'] - baseline['raw_pos_mech_w']
            delta_neg = row['raw_neg_mech_w'] - baseline['raw_neg_mech_w']
            
            # Combine raw change (unmultiplied delta) for outlier screening
            # (Ensures we keep the change negative relative to baseline)
            if (delta_pos + delta_neg) > 0:
                continue
            
            delta_data.append({
                'trial_name': trial_name,
                'window_start_s': row['window_start_s'],
                'delta_bio_w': delta_bio,
                'delta_pos_w': delta_pos,
                'delta_neg_w': delta_neg,
                'raw_bio_w': row['net_bio_cost_w']
            })
            
    return pd.DataFrame(delta_data)


def plot_fitted_correlation(df_deltas, use_no_achilles=False):
    """
    Solves for the optimal multipliers of positive and negative work
    and plots the correlation against the fitted mechanical cost.
    """
    # 1. Solve the OLS regression through the origin (no intercept) for two variables:
    # delta_bio_w = beta_pos * delta_pos_w + beta_neg * delta_neg_w
    X_global = np.column_stack((df_deltas['delta_pos_w'].values, df_deltas['delta_neg_w'].values))
    y_global = df_deltas['delta_bio_w'].values
    
    # Solve least squares
    beta, _, _, _ = np.linalg.lstsq(X_global, y_global, rcond=None)
    beta_pos, beta_neg = beta[0], beta[1]

    # Reconstruct the optimal fitted mechanical change coordinate for each point
    df_deltas['delta_fitted_mech_w'] = beta_pos * df_deltas['delta_pos_w'] + beta_neg * df_deltas['delta_neg_w']

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    
    # Configure headings
    title_str = "Within-Trial Changes: Coefs Fitted to Positive & Negative Mechanical Work"
    sub_str = (
        f"Multiple Linear Regression through the origin. Optimal coefficients: "
        f"Positive Multiplier ($\\beta_{{pos}}$) = {beta_pos:.2f}, "
        f"Negative Multiplier ($\\beta_{{neg}}$) = {beta_neg:.2f}"
    )
    fig.text(0.05, 0.93, title_str, fontsize=14, fontweight='bold', color=NOTION_TEXT)
    fig.text(0.05, 0.89, sub_str, fontsize=10, color=NOTION_SUBTEXT)
    
    # Set right margin to 0.68 to reserve 32% of canvas width for the extended legend labels
    plt.subplots_adjust(top=0.82, right=0.68, bottom=0.15)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(NOTION_TEXT)
        
    ax.tick_params(axis='both', colors=NOTION_TEXT, length=4)
    ax.grid(color=NOTION_GRID, linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)

    x_vals = df_deltas['delta_fitted_mech_w'].values
    y_vals = df_deltas['delta_bio_w'].values

    # Color coordinate different trials to highlight within-subject trajectory groupings
    unique_trials = df_deltas['trial_name'].unique()
    colormap = cm.get_cmap('tab10', max(len(unique_trials), 10))
    
    # Plot individual subject fit lines and scatter groupings
    for i, trial_name in enumerate(unique_trials):
        sub_df = df_deltas[df_deltas['trial_name'] == trial_name]
        x_sub = sub_df['delta_fitted_mech_w'].values
        y_sub = sub_df['delta_bio_w'].values
        
        slope_str = "N/A"
        r_val_str = "N/A"
        if len(sub_df) >= 2 and np.std(x_sub) > 0 and np.std(y_sub) > 0:
            # Fit line constrained through origin
            sub_slope = np.sum(x_sub * y_sub) / np.sum(x_sub ** 2)
            slope_str = f"{sub_slope:.2f}"
            
            # Standard Pearson correlation coefficient
            _, _, sub_r, _, _ = linregress(x_sub, y_sub)
            r_val_str = f"{sub_r:+.2f}"
            
            # Plot constrained RTO line terminating exactly at (0,0)
            x_range = np.linspace(np.min(x_sub), 0.0, 50)
            y_range = sub_slope * x_range
            ax.plot(x_range, y_range, color=colormap(i), linestyle='-', linewidth=1.2, alpha=0.5, zorder=2)
            
        ax.scatter(sub_df['delta_fitted_mech_w'], sub_df['delta_bio_w'], 
                   color=colormap(i), label=f"{trial_name}\n  (m = {slope_str}, r = {r_val_str})", s=70, alpha=0.8, 
                   edgecolors='white', linewidths=0.5, zorder=3)

    # Reference cross-hairs at delta origin (0, 0)
    ax.axhline(0, color=NOTION_SUBTEXT, linestyle=':', linewidth=0.8, zorder=1)
    ax.axvline(0, color=NOTION_SUBTEXT, linestyle=':', linewidth=0.8, zorder=1)

    # Global linear regression (The global fit on fitted work naturally has a slope of 1.0)
    global_slope = np.sum(x_vals * y_vals) / np.sum(x_vals ** 2)
    _, _, r_value, p_value, _ = linregress(x_vals, y_vals)
    
    min_x, max_x = np.min(x_vals), np.max(x_vals)
    pad_x = (max_x - min_x) * 0.1 if max_x > min_x else 1.0
    x_line = np.linspace(min_x - pad_x, 0.0, 100)
    y_line = global_slope * x_line
    
    ax.plot(x_line, y_line, color=NOTION_TEXT, linewidth=2.0, zorder=2, 
            label=f"Global Fit (y = {global_slope:.2f}x)")

    stats_text = (
        f"Global Dataset Stats\n"
        f"Slope $m$: {global_slope:.2f}\n"
        f"Pearson $r$: {r_value:.3f}\n"
        f"p-value: {p_value:.2e}\n"
        f"N: {len(df_deltas)} segments"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', color=NOTION_TEXT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=NOTION_BG, edgecolor='#EDEDED', alpha=0.9))

    x_label = "Change in Fitted Mechanical Muscle Cost ($\\beta_{pos}\\Delta P_{pos} + \\beta_{neg}\\Delta P_{neg}$) (W)"
    ax.set_xlabel(x_label, fontsize=10, fontweight='bold', color=NOTION_TEXT)
    ax.set_ylabel("Change in Net Measured Biological Cost ($\Delta$ Metabolics) (W)", fontsize=10, fontweight='bold', color=NOTION_TEXT)
    
    ax.set_xlim(min_x - pad_x, 2.0)
    
    # Place the legend on the right side of the axes bounding box
    ax.legend(frameon=True, facecolor=NOTION_BG, edgecolor=NOTION_GRID, fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
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
    plot_fitted_correlation(df_deltas, use_no_achilles=args.no_achilles)


if __name__ == "__main__":
    main()