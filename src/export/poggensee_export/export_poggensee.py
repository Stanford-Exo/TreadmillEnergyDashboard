import argparse
import os
import sys
import zipfile
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.io

try:
    import pyarrow
except ImportError:
    print("Warning: 'pyarrow' is not installed. Saving parquet files might fail.")
    print("Please run: pip install pyarrow")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
DEFAULT_INPUT_DIR = "../../pognc_data"
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "exported_pogensee")


def load_mat_file(file_bytes):
    """Loads a .mat file directly from memory bytes, falling back to h5py for v7.3 formats."""
    try:
        return scipy.io.loadmat(io.BytesIO(file_bytes)), False
    except NotImplementedError:
        try:
            import h5py

            return h5py.File(io.BytesIO(file_bytes), "r"), True
        except ImportError:
            print("Error: MATLAB v7.3 format requires the 'h5py' package.")
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


def estimate_channel_bias(data, threshold=40.0):
    candidates = data[np.abs(data) < threshold]
    if len(candidates) == 0:
        return 0.0
    return half_sample_mode(candidates)


def estimate_moment_bias(moment_data, vertical_force, force_threshold=40.0):
    unloaded_idx = np.abs(vertical_force) < force_threshold
    candidates = moment_data[unloaded_idx]
    if len(candidates) == 0:
        return 0.0
    return half_sample_mode(candidates)


def process_qs_worker(zip_path, internal_path):
    """Worker function to process a single QS file from a zip in memory."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_bytes = zf.read(internal_path)

        mat_data, is_v73 = load_mat_file(file_bytes)
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

                cal_per_min = 3.941 * vo2_mean + 1.106 * vco2_mean
                bio_watts = cal_per_min * 4.184 / 60.0

                if is_v73:
                    mat_data.close()
                return f"{zip_path}/{internal_path}", bio_watts

        if is_v73:
            mat_data.close()
        return f"{zip_path}/{internal_path}", None

    except Exception as e:
        print(
            f"    Failed to extract baseline from {internal_path} inside {os.path.basename(zip_path)}: {e}"
        )
        return f"{zip_path}/{internal_path}", None


def get_best_qs_match(target_path, qs_baselines):
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


def export_trial_worker(
    zip_path, internal_path, output_dir, baseline_w=None, matched_qs_path=None
):
    """Worker function to read, transform, clean, and export a trial completely in memory."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_bytes = zf.read(internal_path)

        mat_data, is_v73 = load_mat_file(file_bytes)
    except Exception as e:
        return f"Skipping {internal_path}: Unable to parse mat file structure. {e}"

    try:
        LFx = get_array(mat_data, "LFx", is_v73).copy()
        LFy = get_array(mat_data, "LFy", is_v73).copy()
        LFz = get_array(mat_data, "LFz", is_v73).copy()
        LMx = get_array(mat_data, "LMx", is_v73).copy()
        LMy = get_array(mat_data, "LMy", is_v73).copy()
        LMz = get_array(mat_data, "LMz", is_v73).copy()

        RFx = get_array(mat_data, "RFx", is_v73).copy()
        RFy = get_array(mat_data, "RFy", is_v73).copy()
        RFz = get_array(mat_data, "RFz", is_v73).copy()
        RMx = get_array(mat_data, "RMx", is_v73).copy()
        RMy = get_array(mat_data, "RMy", is_v73).copy()
        RMz = get_array(mat_data, "RMz", is_v73).copy()

        time = get_array(mat_data, "time", is_v73)
    except KeyError as e:
        if is_v73:
            mat_data.close()
        return f"Skipping {internal_path}: Missing target variables {e}"

    bias_LFx = estimate_channel_bias(LFx, threshold=40.0)
    bias_LFy = estimate_channel_bias(LFy, threshold=40.0)
    bias_LFz = estimate_channel_bias(LFz, threshold=40.0)

    bias_RFx = estimate_channel_bias(RFx, threshold=40.0)
    bias_RFy = estimate_channel_bias(RFy, threshold=40.0)
    bias_RFz = estimate_channel_bias(RFz, threshold=40.0)

    bias_LMx = estimate_moment_bias(LMx, LFz, force_threshold=40.0)
    bias_LMy = estimate_moment_bias(LMy, LFz, force_threshold=40.0)
    bias_LMz = estimate_moment_bias(LMz, LFz, force_threshold=40.0)

    bias_RMx = estimate_moment_bias(RMx, RFz, force_threshold=40.0)
    bias_RMy = estimate_moment_bias(RMy, RFz, force_threshold=40.0)
    bias_RMz = estimate_moment_bias(RMz, RFz, force_threshold=40.0)

    LFx_clean = LFx - bias_LFx
    LFy_clean = LFy - bias_LFy
    LFz_clean = LFz - bias_LFz
    LMx_clean = LMx - bias_LMx
    LMy_clean = LMy - bias_LMy
    LMz_clean = LMz - bias_LMz

    RFx_clean = RFx - bias_RFx
    RFy_clean = RFy - bias_RFy
    RFz_clean = RFz - bias_RFz
    RMx_clean = RMx - bias_RMx
    RMy_clean = RMy - bias_RMy
    RMz_clean = RMz - bias_RMz

    norm_l = np.sqrt(LFx_clean**2 + LFy_clean**2 + LFz_clean**2)
    norm_r = np.sqrt(RFx_clean**2 + RFy_clean**2 + RFz_clean**2)

    under_threshold_l = norm_l < 5.0
    under_threshold_r = norm_r < 5.0

    LFx_clean[under_threshold_l] = 0.0
    LFy_clean[under_threshold_l] = 0.0
    LFz_clean[under_threshold_l] = 0.0
    LMx_clean[under_threshold_l] = 0.0
    LMy_clean[under_threshold_l] = 0.0
    LMz_clean[under_threshold_l] = 0.0

    RFx_clean[under_threshold_r] = 0.0
    RFy_clean[under_threshold_r] = 0.0
    RFz_clean[under_threshold_r] = 0.0
    RMx_clean[under_threshold_r] = 0.0
    RMy_clean[under_threshold_r] = 0.0
    RMz_clean[under_threshold_r] = 0.0

    z0 = -0.015

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

    torque_l_z_local = LMz_clean - (
        cop_l_x_local * LFy_clean - cop_l_y_local * LFx_clean
    )
    torque_r_z_local = RMz_clean - (
        cop_r_x_local * RFy_clean - cop_r_y_local * RFx_clean
    )

    df = pd.DataFrame()
    df["frame"] = np.arange(len(time))
    df["time"] = time

    if baseline_w is not None:
        df["qs_baseline_w"] = baseline_w

    df["calcn_l_force_x"] = -LFy_clean
    df["calcn_l_force_y"] = LFz_clean
    df["calcn_l_force_z"] = LFx_clean

    df["calcn_l_cop_x"] = cop_l_y_local
    df["calcn_l_cop_y"] = 0.0
    df["calcn_l_cop_z"] = -cop_l_x_local - 0.5

    df["calcn_l_torque_x"] = 0.0
    df["calcn_l_torque_y"] = -torque_l_z_local
    df["calcn_l_torque_z"] = 0.0

    df["calcn_r_force_x"] = -RFy_clean
    df["calcn_r_force_y"] = RFz_clean
    df["calcn_r_force_z"] = RFx_clean

    df["calcn_r_cop_x"] = cop_r_y_local
    df["calcn_r_cop_y"] = 0.0
    df["calcn_r_cop_z"] = -cop_r_x_local + 0.5

    df["calcn_r_torque_x"] = 0.0
    df["calcn_r_torque_y"] = -torque_r_z_local
    df["calcn_r_torque_z"] = 0.0

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
            except Exception:
                pass

    if is_v73:
        mat_data.close()

    # Prepend the zip name to ensure uniqueness if multiple zips have matching folder structures
    zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
    rel_path_no_ext = os.path.splitext(internal_path)[0]

    safe_name = (
        f"{zip_basename}_{rel_path_no_ext}".replace(os.sep, "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    out_file_path = os.path.join(output_dir, f"{safe_name}.parquet")
    df.to_parquet(out_file_path, index=False)

    return f"Saved Parquet -> {out_file_path}"


def main():
    parser = argparse.ArgumentParser(
        description="Process Katie Exoskeleton zip archives to Parquet concurrently in-memory."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Path to the directory containing .zip files with MATLAB data",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Target output directory for processed Parquet files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel workers to use",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.dir)
    output_dir = os.path.abspath(args.out)

    if not os.path.exists(input_dir):
        print(f"Warning: Primary input folder path not found: {input_dir}")
        input_dir = os.path.abspath("./")
        print(f"Scanning from current working directory: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    zip_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".zip"):
                zip_files.append(os.path.join(root, f))

    if not zip_files:
        print(f"No active .zip files found in: {input_dir}")
        return

    qs_tasks = []
    trial_tasks = []

    for z in zip_files:
        try:
            with zipfile.ZipFile(z, "r") as zf:
                for name in zf.namelist():
                    # Filter for active .mat files while ignoring macos artifact folders
                    if (
                        name.endswith(".mat")
                        and not name.startswith(".")
                        and "__MACOSX" not in name
                    ):
                        if "QS" in name:
                            qs_tasks.append((z, name))
                        else:
                            trial_tasks.append((z, name))
        except Exception as e:
            print(f"Warning: Could not open {z} as a zip file. Error: {e}")

    print(
        f"Discovered {len(zip_files)} zip archive(s) containing {len(trial_tasks)} trials and {len(qs_tasks)} QS baselines."
    )

    # 1. Process all QS Baselines
    print("\n[Phase 1] Extracting baselines from Quiet Standing (QS) trials...")
    qs_baselines = dict()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_qs_worker, z_path, i_path): (z_path, i_path)
            for z_path, i_path in qs_tasks
        }
        for future in as_completed(futures):
            path_key, baseline_val = future.result()
            if baseline_val is not None:
                qs_baselines[path_key] = baseline_val
                print(f"  -> Extracted baseline: {baseline_val:.1f} W from {path_key}")

    # 2. Process all normal Trial Mat Files
    print(
        f"\n[Phase 2] Executing parallel export on {len(trial_tasks)} trials using {args.workers} workers..."
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for z_path, internal_path in trial_tasks:
            target_path_key = f"{z_path}/{internal_path}"
            best_qs_path, baseline_w = get_best_qs_match(target_path_key, qs_baselines)

            futures.append(
                executor.submit(
                    export_trial_worker,
                    z_path,
                    internal_path,
                    output_dir,
                    baseline_w,
                    best_qs_path,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            print(f"  {result}")

    print("\nExport process completed.")


if __name__ == "__main__":
    main()
