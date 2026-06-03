# File: src/analysis/fit_metabolic_coefficients.py

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

# Setup relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../")))

try:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except ImportError:
    plt = None

DEFAULT_PARQUET_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../exported_pogensee/precomputed_poggensee.parquet")
)

# --- Tufte/Notion Aesthetics ---
if plt is not None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "Arial", "sans-serif"]

NOTION_BG = "#FFFFFF"
NOTION_TEXT = "#37352F"
NOTION_SUBTEXT = "#787774"
NOTION_GRID = "#EDEDED"


def compile_regression_deltas(df):
    """
    Groups data by trial, filters outliers, and calculates deltas of each component
    relative to the trial maximums, matching correlate_metabolics.py.
    """
    print(f"\n[DEBUG] Starting compilation. DataFrame shape: {df.shape}")
    if df.empty:
        return pd.DataFrame()

    unique_trials = df["trial_name"].unique()
    delta_rows = []

    stats = {
        "total_trials_seen": len(unique_trials),
        "skipped_too_short": 0,
        "skipped_empty_after_metabolic_bounds": 0,
        "skipped_too_few_after_transient_filter": 0,
        "valid_trials_compiled": 0,
        "total_windows_loaded": len(df),
        "windows_removed_by_cooldown": 0,
        "windows_removed_by_metabolic_bounds": 0,
        "windows_removed_by_transient_filter": 0,
    }

    for trial_name, group in df.groupby("trial_name"):
        group = group.sort_values(by="window_start_s").reset_index(drop=True)

        # 1. Filter: Exclude cooldown tail transients
        if len(group) > 2:
            group = group.iloc[:-2].reset_index(drop=True)
            stats["windows_removed_by_cooldown"] += 2
        else:
            stats["skipped_too_short"] += 1
            continue

        # 2. Filter: Raw metabolic bounds [50.0, 600.0]
        n_pre_bounds = len(group)
        group_filtered = group[
            (group["net_bio_cost_w"] >= 50.0) & (group["net_bio_cost_w"] <= 600.0)
        ].reset_index(drop=True)
        n_removed_bounds = n_pre_bounds - len(group_filtered)
        stats["windows_removed_by_metabolic_bounds"] += n_removed_bounds

        if group_filtered.empty:
            stats["skipped_empty_after_metabolic_bounds"] += 1
            continue

        group = group_filtered

        # 3. Filter: Extreme metabolic transients (deviating from trial median by >= 100W)
        n_pre_transients = len(group)
        trial_median_bio = group["net_bio_cost_w"].median()
        within_transient_mask = (
            np.abs(group["net_bio_cost_w"] - trial_median_bio) < 100.0
        )

        group_filtered = group[within_transient_mask].reset_index(drop=True)
        n_removed_transients = n_pre_transients - len(group_filtered)
        stats["windows_removed_by_transient_filter"] += n_removed_transients

        if len(group_filtered) < 2:
            stats["skipped_too_few_after_transient_filter"] += 1
            continue

        group = group_filtered
        stats["valid_trials_compiled"] += 1

        # 4. Determine maximum value of each component independently (prevents referencing a single index)
        max_bio_idx = group["net_bio_cost_w"].idxmax()
        max_bio = group.loc[max_bio_idx, "net_bio_cost_w"]
        max_bio_std = (
            group.loc[max_bio_idx, "net_bio_cost_std_w"]
            if "net_bio_cost_std_w" in group.columns
            else 0.0
        )

        max_mech_idx = group["mechanical_power"].idxmax()
        max_mech_std = (
            group.loc[max_mech_idx, "mechanical_power_std"]
            if "mechanical_power_std" in group.columns
            else 0.0
        )

        max_pos_mus = (
            group["ref_mus_pos_power_w"] + group["con_mus_pos_power_w"]
        ).max()
        max_neg_mus = (
            group["ref_mus_neg_power_w"] + group["con_mus_neg_power_w"]
        ).max()
        max_pos_ach = (
            group["ref_ach_pos_power_w"] + group["con_ach_pos_power_w"]
        ).max()

        for idx, row in group.iterrows():
            pos_mus = row["ref_mus_pos_power_w"] + row["con_mus_pos_power_w"]
            neg_mus = row["ref_mus_neg_power_w"] + row["con_mus_neg_power_w"]
            pos_ach = row["ref_ach_pos_power_w"] + row["con_ach_pos_power_w"]

            curr_bio_std = (
                row["net_bio_cost_std_w"] if "net_bio_cost_std_w" in row else 5.0
            )
            delta_bio_std = (
                np.sqrt(curr_bio_std**2 + max_bio_std**2)
                if max_bio_std > 0
                else curr_bio_std
            )

            curr_mech_std = (
                row["mechanical_power_std"] if "mechanical_power_std" in row else 0.0
            )
            delta_mech_std = (
                np.sqrt(curr_mech_std**2 + max_mech_std**2)
                if max_mech_std > 0
                else curr_mech_std
            )

            # Compute deltas relative to baseline peak window
            delta_rows.append(
                {
                    "trial_name": trial_name,
                    "window_start_s": row["window_start_s"],
                    "delta_bio_w": row["net_bio_cost_w"] - max_bio,
                    "delta_bio_std_w": delta_bio_std,
                    "delta_mech_std_w": delta_mech_std,
                    "num_valid_strides": row["num_valid_strides"]
                    if "num_valid_strides" in row
                    else 250,
                    "delta_pos_mus_w": pos_mus - max_pos_mus,
                    "delta_neg_mus_w": neg_mus - max_neg_mus,
                    "delta_pos_ach_w": pos_ach - max_pos_ach,
                    # Keep raw values to compute raw predictions during plotting
                    "raw_pos_mus_w": pos_mus,
                    "raw_neg_mus_w": neg_mus,
                    "raw_pos_ach_w": pos_ach,
                }
            )

    print("\n[DEBUG] Filter Statistics Summary:")
    for key, val in stats.items():
        print(f"  - {key:38}: {val}")
    print(f"  - Total delta rows compiled for OLS: {len(delta_rows)}\n")

    return pd.DataFrame(delta_rows)


def fit_coefficients(df_deltas, use_wls=True):
    """
    Fits standard OLS or Weighted Least Squares (WLS) forced through the origin: y = X * beta
    """
    features = [
        "delta_pos_mus_w",
        "delta_neg_mus_w",
        "delta_pos_ach_w",
    ]
    X = df_deltas[features].values
    y = df_deltas["delta_bio_w"].values

    if use_wls and "delta_bio_std_w" in df_deltas.columns:
        # Enforce minimum uncertainty bounds to avoid division by zero
        stds = np.maximum(df_deltas["delta_bio_std_w"].values, 1.0)
        weights = 1.0 / (stds**2)

        # Scale weights to average 1.0 for unbiased residual estimation
        weights = weights / np.mean(weights)

        # Construct weighted design matrices
        sqrt_W = np.diag(np.sqrt(weights))
        X_w = sqrt_W @ X
        y_w = np.sqrt(weights) * y

        XTX = X_w.T @ X_w
        XTy = X_w.T @ y_w
    else:
        XTX = X.T @ X
        XTy = X.T @ y

    beta = np.linalg.solve(XTX, XTy)

    residuals = y - X @ beta
    rss = np.sum(residuals**2)
    tss = np.sum(y**2)

    n, p = X.shape
    s2 = rss / (n - p) if n > p else 0.0

    cov_beta = s2 * np.linalg.inv(XTX) if n > p else np.zeros((p, p))
    se_beta = np.sqrt(np.diag(cov_beta))

    t_stats = beta / se_beta
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=n - p))
    r2 = 1.0 - (rss / tss) if tss > 0 else 0.0

    results = {
        "coefficients": beta,
        "standard_errors": se_beta,
        "t_statistics": t_stats,
        "p_values": p_values,
        "r2": r2,
        "residuals": residuals,
        "n": n,
        "p": p,
        "features": features,
    }
    return results


def plot_fitting_results(df_deltas, results):
    """
    Plots Actual vs. Predicted Delta Metabolics alongside estimated coefficients.
    Forces both predicted and actual maximums to be 0 for each trial, scattering down/left.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=NOTION_BG)
    ax1.set_facecolor(NOTION_BG)
    ax2.set_facecolor(NOTION_BG)

    title = "Empirical Regression: Fitting Biological and Elastic Cost Coefficients"
    subtitle = f"Multi-variable Weighted Least Squares forced through the origin | N = {results['n']} windows | Combined R^2 = {results['r2']:.3f}"

    fig.text(0.04, 0.94, title, fontsize=15, fontweight="bold", color=NOTION_TEXT)
    fig.text(0.04, 0.90, subtitle, fontsize=10.5, color=NOTION_SUBTEXT)
    plt.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.92, wspace=0.28)

    for ax in [ax1, ax2]:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(NOTION_TEXT)
        ax.tick_params(axis="both", colors=NOTION_TEXT, length=4, labelsize=9)
        ax.grid(
            color="#F1F5F9", linestyle="-", linewidth=0.5
        )  # Muted, light Tufte grid
        ax.set_axisbelow(True)

    # ---------------------------------------------------------
    # Subplot 1: Actual vs. Predicted Delta Metabolics
    # ---------------------------------------------------------
    unique_trials = df_deltas["trial_name"].unique()
    colormap = cm.get_cmap("tab10", max(len(unique_trials), 10))

    all_actual_deltas = []
    all_pred_deltas = []

    for i, trial_name in enumerate(unique_trials):
        sub_df = df_deltas[df_deltas["trial_name"] == trial_name].copy()

        actual_delta = sub_df["delta_bio_w"].values

        raw_pred = (
            results["coefficients"][0] * sub_df["raw_pos_mus_w"].values
            + results["coefficients"][1] * sub_df["raw_neg_mus_w"].values
            + results["coefficients"][2] * sub_df["raw_pos_ach_w"].values
        )

        pred_delta = raw_pred - np.max(raw_pred)

        all_actual_deltas.extend(actual_delta)
        all_pred_deltas.extend(pred_delta)

        # Plot centers (slightly smaller, crisp white border)
        ax1.scatter(
            pred_delta,
            actual_delta,
            color=colormap(i),
            s=30,
            alpha=0.90,
            edgecolors="white",
            linewidths=0.5,
            label=trial_name,
            zorder=4,
        )

        # Plot uncertainty ellipses scaled down to Standard Error of the Mean (SEM)
        # Using hollow ellipses (facecolor='none') to eliminate color overlaps/mud
        for idx, (_, row_val) in enumerate(sub_df.iterrows()):
            pt_pred_delta = pred_delta[idx]
            pt_actual_delta = actual_delta[idx]

            y_std = row_val["delta_bio_std_w"] if "delta_bio_std_w" in row_val else 0.0
            x_std = (
                row_val["delta_mech_std_w"] if "delta_mech_std_w" in row_val else 0.0
            )

            n_strides = (
                row_val["num_valid_strides"] if "num_valid_strides" in row_val else 250
            )
            if pd.isna(n_strides) or n_strides <= 0:
                n_strides = 250

            # Approximately 15 breath-by-breath measurements are collected per 5-minute window
            n_breaths = 15

            x_sem = x_std / np.sqrt(n_strides)
            y_sem = y_std / np.sqrt(n_breaths)

            ellipse = Ellipse(
                xy=(pt_pred_delta, pt_actual_delta),
                width=2 * x_sem,
                height=2 * y_sem,
                angle=0,
                facecolor="none",  # Hollow faces prevent color mud
                edgecolor=colormap(i),
                alpha=0.35,  # Legible outline alpha
                linewidth=0.6,  # Delicate Tufte line weight
                zorder=3,
            )
            ax1.add_patch(ellipse)

    # Reference line of perfect correlation: both drop identically
    min_val = min(np.min(all_actual_deltas), np.min(all_pred_deltas))
    ax1.plot(
        [min_val - 5.0, 5.0],
        [min_val - 5.0, 5.0],
        color=NOTION_TEXT,
        linestyle="--",
        linewidth=1.2,
        label="Perfect Fit (y=x)",
    )

    # Anchor coordinates at origin (0, 0) top-right
    ax1.set_xlim(min_val - 5.0, 5.0)
    ax1.set_ylim(min_val - 5.0, 5.0)
    ax1.set_xlabel(
        "Change in Predicted Metabolic Cost ($\Delta$ Predicted) (W)",
        fontsize=10,
        fontweight="bold",
        color=NOTION_TEXT,
    )
    ax1.set_ylabel(
        "Change in Measured Metabolic Cost ($\Delta$ Observed) (W)",
        fontsize=10,
        fontweight="bold",
        color=NOTION_TEXT,
    )
    ax1.set_title(
        "Aligned Model Performance (Deviations from Peaks)",
        fontsize=11,
        fontweight="bold",
        color=NOTION_TEXT,
        pad=10,
    )
    ax1.legend(
        frameon=True,
        facecolor=NOTION_BG,
        edgecolor=NOTION_GRID,
        fontsize=7.5,
        loc="lower left",
        ncol=2,
    )

    # ---------------------------------------------------------
    # Subplot 2: Bar Chart of Estimated Coefficients
    # ---------------------------------------------------------
    labels = [
        "Positive Muscle\n(c_pos_mus)",
        "Negative Muscle\n(c_neg_mus)",
        "Achilles Elastic\n(c_ach)",
    ]

    coeffs = results["coefficients"]
    errors = results["standard_errors"]
    p_vals = results["p_values"]

    bar_colors = ["#FCA5A5", "#FCA5A5", "#D1D5DB"]
    edge_colors = ["#DC2626", "#DC2626", "#4B5563"]

    bars = ax2.bar(
        labels,
        coeffs,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=1.2,
        yerr=errors,
        capsize=5,
        error_kw=dict(ecolor=NOTION_TEXT, elinewidth=1.5, markeredgewidth=1.5),
    )

    # Superimpose standard 4:1 guideline references
    literature_defaults = [4.0, 1.0, 0.0]
    for idx, baseline in enumerate(literature_defaults):
        ax2.hlines(
            baseline,
            xmin=idx - 0.3,
            xmax=idx + 0.3,
            colors="#DC2626",
            linestyles=":",
            linewidths=1.5,
            zorder=5,
        )

    ax2.axhline(0, color=NOTION_TEXT, linewidth=0.8)
    ax2.set_ylabel(
        "Fitted Coefficient Value", fontsize=10, fontweight="bold", color=NOTION_TEXT
    )
    ax2.set_title(
        "Estimated Cost Parameters",
        fontsize=11,
        fontweight="bold",
        color=NOTION_TEXT,
        pad=10,
    )

    # Apply responsive Y headroom to keep the labels from colliding with the upper frame
    max_upper_bound = max([coeffs[j] + errors[j] for j in range(len(coeffs))])
    ax2.set_ylim(0.0, max_upper_bound * 1.35)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        val = coeffs[idx]
        err = errors[idx]
        p_val = p_vals[idx]

        text_str = (
            f"{val:+.2f}\n±{err:.2f}\np={p_val:.3f}"
            if p_val >= 0.001
            else f"{val:+.2f}\n±{err:.2f}\np<0.001"
        )
        va_dir = "bottom" if height >= 0 else "top"

        # Position the label text comfortably above the top error bar cap
        y_pos = (
            height + errors[idx] + (max_upper_bound * 0.05)
            if height >= 0
            else height - errors[idx] - (max_upper_bound * 0.15)
        )

        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            text_str,
            ha="center",
            va=va_dir,
            color=NOTION_TEXT,
            fontsize=8,
            fontweight="bold",
        )

    ax2.plot(
        [], [], color="#DC2626", linestyle=":", label="Theoretical Lit Value (4:1:0:0)"
    )
    ax2.legend(frameon=False, loc="upper right", fontsize=8, labelcolor=NOTION_TEXT)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Fits metabolic coefficients directly to the wide-format dataset."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_PARQUET_PATH,
        help="Path to precomputed parquet file.",
    )
    args = parser.parse_args()

    if plt is None:
        print("Error: matplotlib is required. Please run: pip install matplotlib")
        sys.exit(1)

    parquet_path = os.path.abspath(args.file)
    if not os.path.exists(parquet_path):
        print(f"Error: Precomputed parquet database not found at: {parquet_path}")
        print("Please process files using: make precompute-poggensee first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    if df.empty:
        print("Empty dataset. Check database contents.")
        sys.exit(1)

    df_deltas = compile_regression_deltas(df)

    if df_deltas.empty:
        print("No valid trial windows remain after clearing transients and outliers.")
        sys.exit(1)

    results = fit_coefficients(df_deltas)

    print("\n==================================================")
    print("      ESTIMATED METABOLIC REGRESSION COEFFS       ")
    print("==================================================")
    print(
        f"Fit on {results['n']} windows across {len(df_deltas['trial_name'].unique())} unique trials."
    )
    print(f"Forced through origin | Combined R^2: {results['r2']:.4f}\n")

    for idx, col in enumerate(results["features"]):
        name = col.replace("delta_", "").replace("_w", "")
        beta_val = results["coefficients"][idx]
        se_val = results["standard_errors"][idx]
        t_val = results["t_statistics"][idx]
        p_val = results["p_values"][idx]

        p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"

        print(f"  Coefficient {name:15}: {beta_val:+.4f} ± {se_val:.4f}")
        print(f"    - t-statistic: {t_val:+.3f}")
        print(f"    - p-value    : {p_str}")
        print("-" * 50)

    plot_fitting_results(df_deltas, results)


if __name__ == "__main__":
    main()
