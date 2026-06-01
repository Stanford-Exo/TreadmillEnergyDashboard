import os
import sys
import argparse
import scipy.io
import pandas as pd
import numpy as np

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


def load_mat_file(file_path):
    """Loads a .mat file, falling back to h5py for v7.3 formats."""
    try:
        return scipy.io.loadmat(file_path), False
    except NotImplementedError:
        try:
            import h5py
            return h5py.File(file_path, 'r'), True
        except ImportError:
            print(f"Error: {file_path} is in MATLAB v7.3 format which requires the 'h5py' package.")
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
    Recursively isolates the densest interval containing 50% of the samples
    until a small subset remains, returning their mean.
    """
    if len(data) == 0:
        return 0.0
    
    # Sort data to enable sliding-window range checks
    pts = np.sort(data)
    
    while len(pts) > 3:
        n = len(pts)
        half = n // 2
        # Calculate the span of each window containing 'half' elements
        ranges = pts[half - 1:] - pts[:-half + 1]
        min_idx = np.argmin(ranges)
        # Keep only the elements within the densest interval
        pts = pts[min_idx : min_idx + half]
        
    return float(np.mean(pts))


def estimate_channel_bias(data, threshold=40.0):
    """Filters data below absolute threshold and computes the Half-Sample Mode."""
    candidates = data[np.abs(data) < threshold]
    if len(candidates) == 0:
        return 0.0
    return half_sample_mode(candidates)


def export_trial(file_path, input_dir, output_dir):
    """Translates a single Pogensee .mat file to a transformed, zero-corrected Parquet file."""
    # Display the relative path so nested files are easy to identify in the console
    rel_path_for_display = os.path.relpath(file_path, input_dir)
    print(f"  Processing: {rel_path_for_display}")
    
    try:
        mat_data, is_v73 = load_mat_file(file_path)
    except Exception as e:
        print(f"    Skipping {file_path}: Unable to parse mat file structure. {e}")
        return

    # Extract raw force, torque, and temporal arrays
    try:
        LFx = get_array(mat_data, 'LFx', is_v73).copy()
        LFy = get_array(mat_data, 'LFy', is_v73).copy()
        LFz = get_array(mat_data, 'LFz', is_v73).copy()
        LMx = get_array(mat_data, 'LMx', is_v73).copy()
        LMy = get_array(mat_data, 'LMy', is_v73).copy()
        LMz = get_array(mat_data, 'LMz', is_v73).copy()

        RFx = get_array(mat_data, 'RFx', is_v73).copy()
        RFy = get_array(mat_data, 'RFy', is_v73).copy()
        RFz = get_array(mat_data, 'RFz', is_v73).copy()
        RMx = get_array(mat_data, 'RMx', is_v73).copy()
        RMy = get_array(mat_data, 'RMy', is_v73).copy()
        RMz = get_array(mat_data, 'RMz', is_v73).copy()

        time = get_array(mat_data, 'time', is_v73)
    except KeyError as e:
        print(f"    Skipping {file_path}: Missing target variables {e}")
        if is_v73:
            mat_data.close()
        return

    # 1. Estimate biases using the Half-Sample Mode on low-magnitude absolute values (< 40N)
    bias_LFx = estimate_channel_bias(LFx, threshold=40.0)
    bias_LFy = estimate_channel_bias(LFy, threshold=40.0)
    bias_LFz = estimate_channel_bias(LFz, threshold=40.0)

    bias_RFx = estimate_channel_bias(RFx, threshold=40.0)
    bias_RFy = estimate_channel_bias(RFy, threshold=40.0)
    bias_RFz = estimate_channel_bias(RFz, threshold=40.0)

    # 2. Subtract the estimated biases
    LFx_clean = LFx - bias_LFx
    LFy_clean = LFy - bias_LFy
    LFz_clean = LFz - bias_LFz

    RFx_clean = RFx - bias_RFx
    RFy_clean = RFy - bias_RFy
    RFz_clean = RFz - bias_RFz

    # 3. Post-filter step: Zero out timesteps where 3D force magnitude is under 5N
    norm_l = np.sqrt(LFx_clean**2 + LFy_clean**2 + LFz_clean**2)
    norm_r = np.sqrt(RFx_clean**2 + RFy_clean**2 + RFz_clean**2)

    under_threshold_l = norm_l < 5.0
    under_threshold_r = norm_r < 5.0

    # Clean left foot
    LFx_clean[under_threshold_l] = 0.0
    LFy_clean[under_threshold_l] = 0.0
    LFz_clean[under_threshold_l] = 0.0
    LMx[under_threshold_l] = 0.0
    LMy[under_threshold_l] = 0.0
    LMz[under_threshold_l] = 0.0

    # Clean right foot
    RFx_clean[under_threshold_r] = 0.0
    RFy_clean[under_threshold_r] = 0.0
    RFz_clean[under_threshold_r] = 0.0
    RMx[under_threshold_r] = 0.0
    RMy[under_threshold_r] = 0.0
    RMz[under_threshold_r] = 0.0

    # Print log of estimated offsets to standard output for review
    print(f"    Calculated Zero-Bias Offsets (HSM):")
    print(f"      Left Plate  -> Fx: {bias_LFx:+.3f} N, Fy: {bias_LFy:+.3f} N, Fz: {bias_LFz:+.3f} N")
    print(f"      Right Plate -> Fx: {bias_RFx:+.3f} N, Fy: {bias_RFy:+.3f} N, Fz: {bias_RFz:+.3f} N")

    # 15mm sensor depth offset in meters (vertical dimension)
    z0 = -0.015

    # Compute Local Center of Pressures (CoPs) with division guards
    with np.errstate(divide='ignore', invalid='ignore'):
        cop_l_x_local = np.where(np.abs(LFz_clean) > 0.0, (-LMy + z0 * LFx_clean) / LFz_clean, 0.0)
        cop_l_y_local = np.where(np.abs(LFz_clean) > 0.0, (LMx + z0 * LFy_clean) / LFz_clean, 0.0)

        cop_r_x_local = np.where(np.abs(RFz_clean) > 0.0, (-RMy + z0 * RFx_clean) / RFz_clean, 0.0)
        cop_r_y_local = np.where(np.abs(RFz_clean) > 0.0, (RMx + z0 * RFy_clean) / RFz_clean, 0.0)

    # Compute Local Free Torque about vertical axis at CoP: Tz = Mz - (x * Fy - y * Fx)
    torque_l_z_local = LMz - (cop_l_x_local * LFy_clean - cop_l_y_local * LFx_clean)
    torque_r_z_local = RMz - (cop_r_x_local * RFy_clean - cop_r_y_local * RFx_clean)

    # Populate primary output mapping
    df = pd.DataFrame()
    df["frame"] = np.arange(len(time))
    df["time"] = time

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
    for col in ["com_pos_x", "com_pos_y", "com_pos_z",
                "com_vel_x", "com_vel_y", "com_vel_z",
                "com_acc_x", "com_acc_y", "com_acc_z"]:
        df[col] = 0.0

    # Translate and copy auxiliary non-coordinate parameters from the .mat source
    keys_to_skip = {
        'LFx', 'LFy', 'LFz', 'LMx', 'LMy', 'LMz',
        'RFx', 'RFy', 'RFz', 'RMx', 'RMy', 'RMz', 'time'
    }
    
    source_keys = list(mat_data.keys())
    for key in source_keys:
        if key not in keys_to_skip and not key.startswith('_'):
            try:
                df[key] = get_array(mat_data, key, is_v73)
            except Exception as e:
                print(f"      Warning: Could not translate auxiliary parameter '{key}': {e}")

    # Ensure v7.3 files are cleanly released
    if is_v73:
        mat_data.close()

    # Create safe unique filename based on folder path
    rel_path = os.path.relpath(file_path, input_dir)
    rel_path_no_ext = os.path.splitext(rel_path)[0]
    
    # Replace slashes, backslashes, and spaces with underscores
    safe_name = rel_path_no_ext.replace(os.sep, '_').replace('/', '_').replace('\\', '_').replace(' ', '_')
    out_file_path = os.path.join(output_dir, f"{safe_name}.parquet")
    
    df.to_parquet(out_file_path, index=False)
    print(f"    Saved Parquet -> {out_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Process Katie Exoskeleton .mat trial files to Parquet.")
    parser.add_argument("--dir", type=str, default=DEFAULT_INPUT_DIR,
                        help="Path to the directory containing Katie Exoskeleton mat subfolders")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Target output directory for processed Parquet files")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.dir)
    output_dir = os.path.abspath(args.out)

    if not os.path.exists(input_dir):
        # Fallback to local execution directory search if default pathing is absent
        print(f"Warning: Primary input folder path not found: {input_dir}")
        input_dir = os.path.abspath("./")
        print(f"Scanning from current working directory: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    mat_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            # Gather relevant trial MAT files while ignoring static calibration trials
            if f.endswith(".mat") and not f.startswith(".") and "QS" not in f:
                mat_files.append(os.path.join(root, f))

    if not mat_files:
        print(f"No trial .mat files found in: {input_dir}")
        return

    print(f"Found {len(mat_files)} trial file(s) for conversion. Beginning export processing...")
    for file_path in sorted(mat_files):
        # Pass input_dir so we can compute the relative path inside export_trial
        export_trial(file_path, input_dir, output_dir)
    print("\nExport process completed.")


if __name__ == "__main__":
    main()