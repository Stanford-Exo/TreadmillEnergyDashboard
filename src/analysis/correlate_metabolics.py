# File: src/analysis/correlate_metabolics.py

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    plt = None

# Default path points to where build_metabolics_csv.py writes
DEFAULT_CSV_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../exported_pogensee/metabolics_summary.csv"))

# --- Tufte Aesthetics (Matching analyze_poggensee.py) ---
if plt is not None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', '-apple-system', 'Arial', 'sans-serif']

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"


def plot_correlation(df, use_no_achilles=False):
    """
    Plots the scatter correlation between Estimated Mechanical Power and Net Biological Cost.
    Using Tufte constraints and Notion color themes.
    """
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=NOTION_BG)
    ax.set_facecolor(NOTION_BG)
    
    # Title & Subtitle
    if use_no_achilles:
        title = "Metabolic Validation: Mechanics (NO Achilles Model) vs. Respirometry"
        subtitle = "Each point is a 5-minute window. Achilles storage treated as 4x positive / 1x negative muscle work."
        x_col = 'mechanical_power_no_achilles'
        x_label = "Est. Mechanical Cost (No Achilles) (W)"
    else:
        title = "Metabolic Validation: Mechanics (Standard Achilles Model) vs. Respirometry"
        subtitle = "Each point is a 5-minute window. Achilles storage excluded from 4x/1x Muscle heuristic."
        x_col = 'mechanical_power'
        x_label = "Est. Mechanical Muscle Cost (W)"

    fig.text(0.05, 0.93, title, fontsize=15, fontweight='bold', color=NOTION_TEXT)
    fig.text(0.05, 0.89, subtitle, fontsize=11, color=NOTION_SUBTEXT)
    plt.subplots_adjust(top=0.82, right=0.95, bottom=0.15)

    # Clean borders (Tufte style)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(NOTION_TEXT)
        
    ax.tick_params(axis='both', colors=NOTION_TEXT, length=4)
    ax.grid(color=NOTION_GRID, linestyle='-', linewidth=1.0)
    ax.set_axisbelow(True)

    x_vals = df[x_col].values
    y_vals = df['metabolic_power'].values

    # Generate distinct colors for different files
    unique_files = df['trial_name'].unique()
    colormap = cm.get_cmap('tab10', len(unique_files))
    
    # Plot Scatter Points by File
    for i, file_name in enumerate(unique_files):
        sub_df = df[df['trial_name'] == file_name]
        
        ax.scatter(sub_df[x_col], sub_df['metabolic_power'], 
                   color=colormap(i), label=file_name, s=80, alpha=0.8, 
                   edgecolors='white', linewidths=0.5, zorder=3)

    # 1:1 Identity Line
    max_val = max(np.max(x_vals), np.max(y_vals)) * 1.1
    min_val = min(np.min(x_vals), np.min(y_vals)) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], color=NOTION_SUBTEXT, linestyle='--', zorder=1, label="1:1 Unity Line (y=x)")

    # Linear Regression Fit
    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    x_line = np.linspace(min_val, max_val, 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, color=NOTION_TEXT, linewidth=2.5, zorder=2, label=f"Linear Fit (y = {slope:.2f}x {'+' if intercept>0 else '-'} {abs(intercept):.1f})")

    # Add R-squared text box
    stats_text = f"$R^2$: {r_value**2:.3f}\nPearson $r$: {r_value:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=12, va='top', ha='left', color=NOTION_TEXT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=NOTION_BG, edgecolor='#EDEDED', alpha=0.9))

    # Formatting
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold', color=NOTION_TEXT)
    ax.set_ylabel("Net Measured Biological Cost (W)", fontsize=11, fontweight='bold', color=NOTION_TEXT)
    
    # Position legend cleanly
    ax.legend(frameon=True, facecolor=NOTION_BG, edgecolor=NOTION_GRID, fontsize=9, loc='lower right')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot mechanically estimated metabolics vs actual mask data from a CSV.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to the metabolics_summary.csv file.")
    parser.add_argument("--no-achilles", action="store_true", help="Plot the raw muscle cost assuming NO Achilles tendon storage.")
    args = parser.parse_args()

    if plt is None:
        print("Error: matplotlib required. pip install matplotlib")
        sys.exit(1)

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV file at {csv_path}")
        print("Please run 'python src/analysis/build_metabolics_csv.py' first to generate the data.")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if df.empty:
        print(f"The CSV at {csv_path} is empty. No data to plot.")
        sys.exit(1)

    if args.no_achilles and 'mechanical_power_no_achilles' not in df.columns:
        print("Error: Your CSV does not contain the 'mechanical_power_no_achilles' column.")
        print("Please delete your old CSV and re-run build_metabolics_csv.py.")
        sys.exit(1)

    print(f"Successfully loaded {len(df)} total data points across {len(df['trial_name'].unique())} trials.")
    plot_correlation(df, use_no_achilles=args.no_achilles)


if __name__ == "__main__":
    main()