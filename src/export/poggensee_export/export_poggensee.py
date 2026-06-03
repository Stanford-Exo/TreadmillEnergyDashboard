import argparse
import os
import sys
import zipfile
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.io

# Ensure required libraries for Parquet serialization are present
try:
    import pyarrow
except ImportError:
    print("Warning: 'pyarrow' is not installed. Saving parquet files might fail.")
    print("Please run: pip install pyarrow")

# Setup script-relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
DEFAULT_INPUT_DIR = "/Users/keenonwerling/Desktop/data/Katie Exoskeleton"
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "exported_pogensee")


def get_worker_count():
    """Returns the number of workers based on the OS."""
    if sys.platform.startswith('linux'):
        try:
            # sched_getaffinity returns the set of CPUs the process is eligible to run on
            return len(os.sched_getaffinity(0))
        except AttributeError:
            pass
    # Fallback for Windows, macOS, or if sched_getaffinity is unavailable
    return os.cpu_count() or 4


def load_mat_file(file_path):
    """Loads a .mat file, falling back to h5py for v7.3 formats."""
    try:
        return scipy.io.loadmat(file_path), False
    except NotImplementedError:
        try:
            import h5py

            return h5py.File(file_path, "r"), True
        except ImportError:
            print(
                f"Error: {file_path} is in MATLAB v7.3 format which requires the 'h5py' package."
            )
            print("Please run: pip install h5py")
            raise


def get_array(data, key, is_v73):
    """Retrieves and flattens a numeric array from the mat data dictionary."""
    if is_v73:
        arr = np.array(data[key])
        if arr.ndim == 2:
            arr = arr.T
        return arr.flatten()
    else:
        return data[key].flatten()


def half_sample_mode(data):
    """
    Computes the Half-Sample Mode of a 1D array.
    Recursively isolates the densest interval containing 50% of the samples.
    """
    if len(data) == 0:
        return 0.0
    pts = np.sort(data)
    while len(pts) > 3:
        n = len(pts)
        half = n // 2
        ranges = pts[half - 1 :] - pts[: -half + 1]
        min_idx = np.argmin(ranges)
        pts = pts[min_idx : min_idx + half]
    return float(np.mean(pts))


def calculate_signals_with_biases(raw_signals, biases, quiet_masks):
    """Applies a set of biases to yield clean and zero-gated signals."""
    zeroed_clean = {"L": {}, "R": {}}
    zeroed_forced = {"L": {}, "R": {}}

    for side in ["L", "R"]:
        for axis in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
            sig_raw = raw_signals[side][axis]
            bias_val = biases[side][axis]
            sig_clean = sig_raw - bias_val

            # Enforce strictly non-negative vertical force
            if axis == "Fz":
                sig_clean = np.maximum(sig_clean, 0.0)

            zeroed_clean[side][axis] = sig_clean

        # Define quiet/swing state mask (quiet baseline window OR vertical force below 10N)
        corrected_fz = zeroed_clean[side]["Fz"]
        swing_gate_mask = quiet_masks[side] | (corrected_fz < 10.0)

        for axis in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
            sig_forced = zeroed_clean[side][axis].copy()
            sig_forced[swing_gate_mask] = 0.0
            zeroed_forced[side][axis] = sig_forced

    return zeroed_clean, zeroed_forced


def apply_percentage_gating(
    zeroed_clean, zeroed_forced, window_size=7, threshold_pct=0.03
):
    """
    Measures the vertical load percentage carried by each belt.
    Smoothes this percentage with a moving boxcar average of the specified window size.
    Anywhere a belt carries less than the threshold percentage (e.g., 3%),
    all force and torque channels for that belt are forced to exactly 0.0.
    """
    Fz_L = zeroed_clean["L"]["Fz"]
    Fz_R = zeroed_clean["R"]["Fz"]
    Fz_total = Fz_L + Fz_R

    # Calculate instantaneous load distribution ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_L = np.where(Fz_total > 0.0, Fz_L / Fz_total, 0.0)
        pct_R = np.where(Fz_total > 0.0, Fz_R / Fz_total, 0.0)

    # Boxcar smoothing filter to suppress high-frequency transitions/noise
    def smooth_signal(sig, w_size):
        if len(sig) < w_size:
            return sig
        window = np.ones(w_size) / w_size
        padded = np.pad(sig, w_size // 2, mode="edge")
        smoothed = np.convolve(padded, window, mode="valid")
        if len(smoothed) > len(sig):
            smoothed = smoothed[: len(sig)]
        elif len(smoothed) < len(sig):
            smoothed = np.pad(smoothed, (0, len(sig) - len(smoothed)), mode="edge")
        return smoothed

    pct_L_smoothed = smooth_signal(pct_L, window_size)
    pct_R_smoothed = smooth_signal(pct_R, window_size)

    # Define binary masks where load drops below the threshold
    gate_L = pct_L_smoothed < threshold_pct
    gate_R = pct_R_smoothed < threshold_pct

    # Enforce zeros across all channels (forces and torques) on both clean and forced sets
    for axis in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
        zeroed_clean["L"][axis] = zeroed_clean["L"][axis].copy()
        zeroed_clean["R"][axis] = zeroed_clean["R"][axis].copy()
        zeroed_forced["L"][axis] = zeroed_forced["L"][axis].copy()
        zeroed_forced["R"][axis] = zeroed_forced["R"][axis].copy()

        zeroed_clean["L"][axis][gate_L] = 0.0
        zeroed_clean["R"][axis][gate_R] = 0.0
        zeroed_forced["L"][axis][gate_L] = 0.0
        zeroed_forced["R"][axis][gate_R] = 0.0

    return zeroed_clean, zeroed_forced


def optimize_offsets(raw_signals, max_fz_adjust=5.0, contact_thresh=30.0):
    """
    Executes a three-pass optimization on raw multi-axial force plate signals:
    Pass 1: Identifies baseline quiet offsets using a percentile-filtered Half-Sample Mode.
    Pass 1.5: Applies percentage-based gating to zero out light/spurious contacts below 3% load.
    Pass 2: Equalizes vertical single-support means and centers horizontal global means to 0.0.
    """
    biases = {"L": {}, "R": {}}
    quiet_masks = {}

    # Pass 1: Baseline Quiet Offsets
    for side in ["L", "R"]:
        fz_raw = raw_signals[side]["Fz"]
        fz_low_threshold = np.percentile(fz_raw, 30)
        fz_quiet_subset = fz_raw[fz_raw <= fz_low_threshold]
        bias_Fz_base = half_sample_mode(fz_quiet_subset)

        quiet_mask = (fz_raw >= bias_Fz_base - 10.0) & (fz_raw <= bias_Fz_base + 10.0)
        quiet_masks[side] = quiet_mask

        for axis in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
            sig_raw = raw_signals[side][axis]
            biases[side][axis] = (
                float(np.mean(sig_raw[quiet_mask])) if np.any(quiet_mask) else 0.0
            )

    zeroed_clean, zeroed_forced = calculate_signals_with_biases(
        raw_signals, biases, quiet_masks
    )

    # Pass 1.5: Gating out low force load shares (< 3%) using a smoothed percentage
    zeroed_clean, zeroed_forced = apply_percentage_gating(
        zeroed_clean, zeroed_forced, window_size=15, threshold_pct=0.03
    )

    # Pass 2: Vertical Equalization & Horizontal Centering
    L_SS_mask = (zeroed_forced["L"]["Fz"] > contact_thresh) & (
        zeroed_forced["R"]["Fz"] == 0.0
    )
    R_SS_mask = (zeroed_forced["R"]["Fz"] > contact_thresh) & (
        zeroed_forced["L"]["Fz"] == 0.0
    )

    # Equalize vertical support loads
    f_z_adjust_L, f_z_adjust_R = 0.0, 0.0
    if np.any(L_SS_mask) and np.any(R_SS_mask):
        mean_L_SS = np.mean(zeroed_forced["L"]["Fz"][L_SS_mask])
        mean_R_SS = np.mean(zeroed_forced["R"]["Fz"][R_SS_mask])

        difference_fz = mean_L_SS - mean_R_SS
        f_z_adjust_L = difference_fz / 2.0
        f_z_adjust_R = -difference_fz / 2.0

        biases["L"]["Fz"] += f_z_adjust_L
        biases["R"]["Fz"] += f_z_adjust_R

    # Centering horizontal channels
    horizontal_adjustments = {"L": {"Fx": 0.0, "Fy": 0.0}, "R": {"Fx": 0.0, "Fy": 0.0}}
    for side in ["L", "R"]:
        for axis in ["Fx", "Fy"]:
            trial_mean = np.mean(zeroed_clean[side][axis])
            horizontal_adjustments[side][axis] = trial_mean
            biases[side][axis] += trial_mean

    # Recompute base signals using refined calibration biases
    zeroed_clean, zeroed_forced = calculate_signals_with_biases(
        raw_signals, biases, quiet_masks
    )

    # Re-apply Pass 1.5 gating on final signals to preserve zeroed intervals in output
    zeroed_clean, zeroed_forced = apply_percentage_gating(
        zeroed_clean, zeroed_forced, window_size=15, threshold_pct=0.03
    )

    return (
        zeroed_clean,
        zeroed_forced,
        biases,
        (f_z_adjust_L, f_z_adjust_R),
        horizontal_adjustments,
    )


def calculate_qs_baseline(file_path):
    """Loads a QS .mat file and extracts the average metabolic Watts."""
    try:
        mat_data, is_v73 = load_mat_file(file_path)
        keys = list(mat_data.keys())

        vo2_key = next((k for k in keys if k.lower() == "vo2"), None)
        vco2_key = next((k for k in keys if k.lower() == "vco2"), None)

        if vo2_key:
            vo2 = get_array(mat_data, vo2_key, is_v73)
            vo2 = vo2[(~np.isnan(vo2)) & (vo2 > 0)]

            if len(vo2) > 0:
                vo2_mean = np.nanmean(vo2)

                if vco2_key:
                    vco2 = get_array(mat_data, vco2_key, is_v73)
                    vco2 = vco2[(~np.isnan(vco2)) & (vco2 > 0)]
                    vco2_mean = np.nanmean(vco2) if len(vco2) > 0 else 0.85 * vo2_mean
                else:
                    vco2_mean = 0.85 * vo2_mean

                # Convert to Watts
                cal_per_min = 3.941 * vo2_mean + 1.106 * vco2_mean
                bio_watts = cal_per_min * 4.184 / 60.0

                if is_v73:
                    mat_data.close()
                return bio_watts

        if is_v73:
            mat_data.close()
        return None
    except Exception as e:
        print(f"    Failed to extract baseline from {file_path}: {e}")
        return None


def get_best_qs_match(target_path, qs_baselines):
    """Finds the QS file that shares the longest common directory path with the target."""
    if not qs_baselines:
        return None, None

    best_qs = None
    max_prefix_len = -1

    for qs_path in qs_baselines.keys():
        prefix = os.path.commonprefix([target_path, qs_path])
        if len(prefix) > max_prefix_len:
            max_prefix_len = len(prefix)
            best_qs = qs_path

    return best_qs, qs_baselines[best_qs] if best_qs else None


def export_trial(
    file_path, input_dir, zip_name, output_dir, baseline_w=None, matched_qs_path=None
):
    """Translates a single Pogensee .mat file to a transformed, zero-corrected Parquet file."""
    rel_path_for_display = os.path.relpath(file_path, input_dir)
    print(f"\n  Processing: {rel_path_for_display} (From: {zip_name}.zip)")

    try:
        mat_data, is_v73 = load_mat_file(file_path)
    except Exception as e:
        print(f"    Skipping {file_path}: Unable to parse mat file structure. {e}")
        return

    # Extract raw force, torque, and temporal arrays
    try:
        raw_signals = {
            "L": {
                "Fx": get_array(mat_data, "LFx", is_v73).copy(),
                "Fy": get_array(mat_data, "LFy", is_v73).copy(),
                "Fz": get_array(mat_data, "LFz", is_v73).copy(),
                "Mx": get_array(mat_data, "LMx", is_v73).copy(),
                "My": get_array(mat_data, "LMy", is_v73).copy(),
                "Mz": get_array(mat_data, "LMz", is_v73).copy(),
            },
            "R": {
                "Fx": get_array(mat_data, "RFx", is_v73).copy(),
                "Fy": get_array(mat_data, "RFy", is_v73).copy(),
                "Fz": get_array(mat_data, "RFz", is_v73).copy(),
                "Mx": get_array(mat_data, "RMx", is_v73).copy(),
                "My": get_array(mat_data, "RMy", is_v73).copy(),
                "Mz": get_array(mat_data, "RMz", is_v73).copy(),
            },
            "time": get_array(mat_data, "time", is_v73),
        }
    except KeyError as e:
        print(f"    Skipping {file_path}: Missing target variables {e}")
        if is_v73:
            mat_data.close()
        return

    # Run the optimized Two-Pass Calibration
    _, zeroed_forced, biases, vertical_adjusts, horizontal_adjusts = optimize_offsets(
        raw_signals
    )

    # Deconstruct optimized forced signals for global biomechanical calculations
    LFx_clean = zeroed_forced["L"]["Fx"]
    LFy_clean = zeroed_forced["L"]["Fy"]
    LFz_clean = zeroed_forced["L"]["Fz"]
    LMx_clean = zeroed_forced["L"]["Mx"]
    LMy_clean = zeroed_forced["L"]["My"]
    LMz_clean = zeroed_forced["L"]["Mz"]

    RFx_clean = zeroed_forced["R"]["Fx"]
    RFy_clean = zeroed_forced["R"]["Fy"]
    RFz_clean = zeroed_forced["R"]["Fz"]
    RMx_clean = zeroed_forced["R"]["Mx"]
    RMy_clean = zeroed_forced["R"]["My"]
    RMz_clean = zeroed_forced["R"]["Mz"]

    print(f"    [{zip_name}] Optimized Zero-Bias Offsets:")
    print(
        f"      Left Plate  -> Fx: {biases['L']['Fx']:+.3f} N (Centered), Fy: {biases['L']['Fy']:+.3f} N (Centered), Fz: {biases['L']['Fz']:+.3f} N"
    )
    print(f"                  -> Equalization adjustment: {vertical_adjusts[0]:+.3f} N")
    print(
        f"      Right Plate -> Fx: {biases['R']['Fx']:+.3f} N (Centered), Fy: {biases['R']['Fy']:+.3f} N (Centered), Fz: {biases['R']['Fz']:+.3f} N"
    )
    print(f"                  -> Equalization adjustment: {vertical_adjusts[1]:+.3f} N")

    # 15mm sensor depth offset in meters (vertical dimension)
    z0 = -0.015

    # Compute Local Center of Pressures (CoPs) with division guards
    with np.errstate(divide="ignore", invalid="ignore"):
        cop_l_x_local = np.where(
            np.abs(LFz_clean) > 0.0, (-LMy_clean + z0 * LFx_clean) / LFz_clean, 0.0
        )
        cop_l_y_local = np.where(
            np.abs(LFz_clean) > 0.0, (LMx_clean + z0 * LFy_clean) / LFz_clean, 0.0
        )

        cop_r_x_local = np.where(
            np.abs(RFz_clean) > 0.0, (-RMy_clean + z0 * RFx_clean) / RFz_clean, 0.0
        )
        cop_r_y_local = np.where(
            np.abs(RFz_clean) > 0.0, (RMx_clean + z0 * RFy_clean) / RFz_clean, 0.0
        )

    # Compute Local Free Torque about vertical axis at CoP: Tz = Mz - (x * Fy - y * Fx)
    torque_l_z_local = LMz_clean - (
        cop_l_x_local * LFy_clean - cop_l_y_local * LFx_clean
    )
    torque_r_z_local = RMz_clean - (
        cop_r_x_local * RFy_clean - cop_r_y_local * RFx_clean
    )

    # Populate primary output mapping
    df = pd.DataFrame()
    df["frame"] = np.arange(len(raw_signals["time"]))
    df["time"] = raw_signals["time"]

    # Insert Dynamic Quiet Standing (QS) baseline if found
    if baseline_w is not None:
        df["qs_baseline_w"] = baseline_w
        print(
            f"    Applied QS Baseline: {baseline_w:.1f} W (Matched: {os.path.basename(matched_qs_path)})"
        )

    # Map Left Foot GRF and CoP to global (Y-up, right-handed system)
    df["calcn_l_force_x"] = -LFy_clean
    df["calcn_l_force_y"] = LFz_clean
    df["calcn_l_force_z"] = LFx_clean

    df["calcn_l_cop_x"] = cop_l_y_local
    df["calcn_l_cop_y"] = 0.0
    df["calcn_l_cop_z"] = -cop_l_x_local - 0.5

    df["calcn_l_torque_x"] = 0.0
    df["calcn_l_torque_y"] = -torque_l_z_local
    df["calcn_l_torque_z"] = 0.0

    # Map Right Foot GRF and CoP to global
    df["calcn_r_force_x"] = -RFy_clean
    df["calcn_r_force_y"] = RFz_clean
    df["calcn_r_force_z"] = RFx_clean

    df["calcn_r_cop_x"] = cop_r_y_local
    df["calcn_r_cop_y"] = 0.0
    df["calcn_r_cop_z"] = -cop_r_x_local + 0.5

    df["calcn_r_torque_x"] = 0.0
    df["calcn_r_torque_y"] = -torque_r_z_local
    df["calcn_r_torque_z"] = 0.0

    # Populate zeroed estimation placeholders to prevent downstream KeyErrors
    for col in [
        "com_pos_x",
        "com_pos_y",
        "com_pos_z",
        "com_vel_x",
        "com_vel_y",
        "com_vel_z",
        "com_acc_x",
        "com_acc_y",
        "com_acc_z",
    ]:
        df[col] = 0.0

    # Translate and copy auxiliary non-coordinate parameters from the .mat source
    keys_to_skip = {
        "LFx",
        "LFy",
        "LFz",
        "LMx",
        "LMy",
        "LMz",
        "RFx",
        "RFy",
        "RFz",
        "RMx",
        "RMy",
        "RMz",
        "time",
    }

    source_keys = list(mat_data.keys())
    for key in source_keys:
        if key not in keys_to_skip and not key.startswith("_"):
            try:
                df[key] = get_array(mat_data, key, is_v73)
            except Exception as e:
                print(
                    f"      Warning: Could not translate auxiliary parameter '{key}': {e}"
                )

    # Ensure v7.3 files are cleanly released
    if is_v73:
        mat_data.close()

    # Create safe unique filename based on the zip name and the relative folder path
    rel_path = os.path.relpath(file_path, input_dir)
    rel_path_no_ext = os.path.splitext(rel_path)[0]

    # Combine zip name and path, replacing problem characters with underscores
    safe_name = f"{zip_name}_{rel_path_no_ext}".replace(os.sep, "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    
    out_file_path = os.path.join(output_dir, f"{safe_name}.parquet")

    df.to_parquet(out_file_path, index=False)
    print(f"    Saved Parquet -> {out_file_path}")


def process_zip(zip_path, output_dir):
    """Worker function to extract a zip to a temporary dir and process its .mat files."""
    print(f"Opening Zip Archive: {zip_path}")
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile:
            print(f"Error: Could not read {zip_path} as a valid zip file.")
            return

        mat_files = [ ]
        qs_files = [ ]

        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.endswith(".mat") and not f.startswith("."):
                    full_path = os.path.join(root, f)
                    if "QS" in f:
                        qs_files.append(full_path)
                    else:
                        mat_files.append(full_path)

        if not mat_files:
            print(f"No active trial .mat files found inside: {zip_path}")
            return

        print(f"[{zip_name}] Found {len(mat_files)} active trial file(s) and {len(qs_files)} QS file(s).")

        qs_baselines = {}
        for qs_file in qs_files:
            baseline = calculate_qs_baseline(qs_file)
            if baseline is not None:
                qs_baselines[qs_file] = baseline
                print(f"  [{zip_name}] Extracted baseline: {baseline:.1f} W from {os.path.basename(qs_file)}")
            else:
                print(f"  [{zip_name}] No valid metabolic data in {os.path.basename(qs_file)}")

        for file_path in sorted(mat_files):
            best_qs_path, baseline_w = get_best_qs_match(file_path, qs_baselines)
            export_trial(file_path, temp_dir, zip_name, output_dir, baseline_w, best_qs_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process Katie Exoskeleton .zip trial files to Parquet."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Path to the directory containing Katie Exoskeleton zip files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Target output directory for processed Parquet files",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.dir)
    output_dir = os.path.abspath(args.out)

    if not os.path.exists(input_dir):
        # Fallback to local execution directory search if default pathing is absent
        print(f"Warning: Primary input folder path not found: {input_dir}")
        input_dir = os.path.abspath("./")
        print(f"Scanning from current working directory: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    zip_files = [ ]
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".zip") and not f.startswith("."):
                zip_files.append(os.path.join(root, f))

    if not zip_files:
        print(f"No active trial .zip files found in: {input_dir}")
        return
        
    worker_count = get_worker_count()
    print(f"\nDiscovered {len(zip_files)} zip file(s) to process.")
    print(f"Starting multiprocessing pool with {worker_count} workers based on the OS...")

    # Spawn worker processes for each zip file
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [ ]
        for zip_file in zip_files:
            futures.append(executor.submit(process_zip, zip_file, output_dir))
        
        # Monitor the completions for error reporting
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"\nWorker generated an exception: {e}")

    print("\nExport process completed.")


if __name__ == "__main__":
    # Needed for windows multiprocessing safety
    multiprocessing.freeze_support()
    main()