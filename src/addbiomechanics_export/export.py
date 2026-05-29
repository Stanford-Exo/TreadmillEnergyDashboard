import os
import sys
import re
from pathlib import Path
import numpy as np
import nimblephysics as nimble
from addbiomechanics_export.classification_pass import classification_pass

# Verify dependencies for parquet writing
try:
    import pandas as pd
except ImportError:
    print("Error: The 'pandas' and 'pyarrow' modules are required to save parquet files.")
    print("Please install them: pip install pandas pyarrow")
    sys.exit(1)

# --- Script-relative Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/Users/keenonwerling/Desktop/data/addb_dataset_publication/'
GEOMETRY_FOLDER = os.path.abspath(os.path.join(SCRIPT_DIR, '../../Geometry'))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../exported_csvs'))  # Output folder holds subdirectories

# Target subdirectories for sorting the outputs
COM_VAL_DIR = os.path.join(OUTPUT_DIR, 'com_validation')
TORQUE_VAL_DIR = os.path.join(OUTPUT_DIR, 'torque_validation')
RENDERING_DIR = os.path.join(OUTPUT_DIR, 'rendering')

EDGE_TRIM_FRAMES = 1

# Ensure target directories exist
os.makedirs(COM_VAL_DIR, exist_ok=True)
os.makedirs(TORQUE_VAL_DIR, exist_ok=True)
os.makedirs(RENDERING_DIR, exist_ok=True)

def download_geometry_if_needed():
    """Ensures the Geometry folder is available at the target location."""
    if not os.path.exists(GEOMETRY_FOLDER):
        print(f"Geometry folder not found at {GEOMETRY_FOLDER}. Downloading...")
        exit_code = os.system('wget https://addbiomechanics.org/resources/Geometry.zip')
        if exit_code != 0:
            print("ERROR: Failed to download Geometry.zip automatically. Please install 'wget' or download manually.")
            return False
        os.system(f'unzip ./Geometry.zip -d {os.path.dirname(GEOMETRY_FOLDER)}')
        os.system('rm ./Geometry.zip')
    return True

def get_b3d_files(base_path: str) -> list:
    """Recursively finds all valid .b3d files within the given path."""
    b3d_files = []
    if not os.path.exists(base_path):
        print(f"Warning: Base directory '{base_path}' does not exist.")
        return b3d_files
        
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.b3d') and os.path.getsize(os.path.join(root, file)) > 0:
                b3d_files.append(os.path.join(root, file))
    return b3d_files

def parse_trial_segment(trial_name: str):
    """Extracts base name and segment index from split trials.
    
    Example: "PDS26CR1_segment_2" -> ("PDS26CR1", 2)
             "PDS26CR1" -> ("PDS26CR1", 0)
    """
    match = re.search(r"^(.*?)(?:_segment_(\d+))?$", trial_name, re.IGNORECASE)
    if match:
        base_name = match.group(1)
        segment_num = match.group(2)
        segment_idx = int(segment_num) if segment_num is not None else 0
        return base_name, segment_idx
    return trial_name, 0

def interpolate_boundaries(df, segment_lengths, W):
    """
    Linearly interpolates data columns across segment boundaries in a consolidated DataFrame.
    
    Leaves index 0 ('frame') and index 1 ('time') untouched to avoid type mismatches
    and pandas FutureWarning warnings.
    """
    if len(segment_lengths) <= 1:
        return df

    # Find row indices of the segment split boundaries
    split_points = []
    current_sum = 0
    for length in segment_lengths[:-1]:
        current_sum += length
        split_points.append(current_sum)

    df_copy = df.copy()

    for split in split_points:
        idx_start = split - W - 1
        idx_end = split + W

        # Safety boundary check
        if idx_start < 0 or idx_end >= len(df_copy):
            continue

        # Interpolate only from column index 2 onwards (ignoring 'frame' and 'time')
        val_start = df_copy.iloc[idx_start, 2:].values.astype(float)
        val_end = df_copy.iloc[idx_end, 2:].values.astype(float)
        num_steps = idx_end - idx_start

        for step in range(1, num_steps):
            alpha = step / num_steps
            interpolated_data = (1.0 - alpha) * val_start + alpha * val_end
            
            # Assign strictly to the floating-point data columns
            df_copy.iloc[idx_start + step, 2:] = interpolated_data

    return df_copy

def export_subject_data(file_path: str, geometry_dir: str):
    """Processes a B3D file, reconstructs split trials, blends boundary transients, and exports."""
    print(f"\nProcessing: {file_path}")
    subject_name = Path(file_path).stem
    
    try:
        subject = nimble.biomechanics.SubjectOnDisk(file_path)
    except Exception as e:
        print(f"Failed to read SubjectOnDisk for {file_path}: {e}")
        return

    # Load the 3rd pass skeleton (index 2)
    try:
        skel = subject.readSkel(2, os.path.abspath(geometry_dir) + '/')
    except Exception as e:
        print(f"Failed to read skeleton from {file_path}: {e}")
        return

    # Identify shape node transforms and build descriptive names
    mesh_paths = {}
    body_shapes = []  # Stores (body_name, shape_index, interpretable_name, shape_node)
    
    for b in range(skel.getNumBodyNodes()):
        body_node = skel.getBodyNode(b)
        body_name = body_node.getName()
        for k in range(body_node.getNumShapeNodes()):
            shape_node = body_node.getShapeNode(k)
            interpretable_name = f"{body_name}_shape_{k}"
            body_shapes.append((body_name, k, interpretable_name, shape_node))
            
            shape = shape_node.getShape()
            mesh_shape = shape.asMeshShape()
            mesh_paths[interpretable_name] = (mesh_shape.getMeshPath(), mesh_shape.getScale())

    # Export mesh metadata inside the rendering subdirectory as Parquet
    mesh_parquet_path = os.path.join(RENDERING_DIR, f"{subject_name}_mesh_paths.parquet")
    mesh_data = []
    for name, (path, scale) in mesh_paths.items():
        mesh_data.append([name, path, scale[0], scale[1], scale[2]])
    
    df_mesh = pd.DataFrame(mesh_data, columns=['name', 'path', 'scale_x', 'scale_y', 'scale_z'])
    df_mesh.to_parquet(mesh_parquet_path, index=False)
    print(f"  -> Exported mesh metadata to {mesh_parquet_path}")

    joints = [skel.getJoint(i) for i in range(skel.getNumJoints())]
    joint_names = [j.getName() for j in joints]
    contact_bodies = subject.getGroundForceBodies()

    try:
        subject.loadAllFrames(doNotStandardizeForcePlateData=True)
    except Exception as e:
        print(f"Failed to load frames for {file_path}: {e}")
        return
    classification_pass(subject)  # Apply classification to populate processingPasses with labels
    raw_header = subject.getHeaderProto()
    raw_trials = list(raw_header.getTrials())

    # Define Headers
    com_header = [
        'frame', 'time',
        'com_pos_x', 'com_pos_y', 'com_pos_z',
        'com_vel_x', 'com_vel_y', 'com_vel_z',
        'com_acc_x', 'com_acc_y', 'com_acc_z'
    ]
    for cb in contact_bodies:
        com_header.extend([
            f"{cb}_force_x", f"{cb}_force_y", f"{cb}_force_z",
            f"{cb}_cop_x", f"{cb}_cop_y", f"{cb}_cop_z",
            f"{cb}_torque_x", f"{cb}_torque_y", f"{cb}_torque_z"
        ])

    trq_header = ['frame', 'time']
    for j_name in joint_names:
        trq_header.extend([
            f"joint_{j_name}_center_pos_x",
            f"joint_{j_name}_center_pos_y",
            f"joint_{j_name}_center_pos_z"
        ])
    for cb in contact_bodies:
        trq_header.extend([
            f"{cb}_force_x", f"{cb}_force_y", f"{cb}_force_z",
            f"{cb}_cop_x", f"{cb}_cop_y", f"{cb}_cop_z",
            f"{cb}_torque_x", f"{cb}_torque_y", f"{cb}_torque_z"
        ])

    render_header = ['frame', 'time']
    for _, _, shape_name, _ in body_shapes:
        render_header.extend([
            f"{shape_name}_pos_x", f"{shape_name}_pos_y", f"{shape_name}_pos_z",
            f"{shape_name}_rot_x", f"{shape_name}_rot_y", f"{shape_name}_rot_z"
        ])

    # Dictionary to store structured frame lists prior to concatenation:
    # base_trial_name -> list of (segment_idx, com_rows, trq_rows, render_rows, dt)
    trial_groups = {}

    # Process individual trials/segments
    for trial in range(subject.getNumTrials()):
        # Constraint 1: Keep only trials that are designated as TREADMILL or STATIC
        trial_type = raw_trials[trial].getBasicTrialType()
        print(f"  Trial {trial}: type is {trial_type}")
        if trial_type != nimble.biomechanics.BasicTrialType.TREADMILL and trial_type != nimble.biomechanics.BasicTrialType.STATIC_TRIAL:
            print(f"  Skipping Trial {trial}: trial type is {trial_type} (expected TREADMILL or STATIC_TRIAL)")
            continue

        # Constraint 2: Keep only trials with exactly 4 processing passes
        num_passes = subject.getTrialNumProcessingPasses(trial)
        if num_passes != 4:
            print(f"  Skipping Trial {trial}: processing passes count is {num_passes} (expected 4)")
            continue

        trial_len = subject.getTrialLength(trial)
        if trial_len == 0:
            continue

        frames = subject.readFrames(trial, 0, trial_len, includeSensorData=False, includeProcessingPasses=True)
        dt = subject.getTrialTimestep(trial)

        # Constraint 3: Verify that all frames have valid GRF data
        has_missing_grf = False
        for frame in frames:
            if frame.missingGRFReason != nimble.biomechanics.MissingGRFReason.notMissingGRF:
                has_missing_grf = True
                break

        if has_missing_grf:
            print(f"  Skipping Trial {trial}: missing GRF elements detected in frame sequence")
            continue

        # Constraint 4: Skip trials with no meaningful horizontal GRF measurements (X or Z axes)
        # We require at least one frame in the trial to have absolute force > 1.0 N on a horizontal axis.
        has_horizontal_grf = False
        for frame in frames:
            raw_pass = frame.processingPasses[0]
            forces = raw_pass.groundContactForce
            for i in range(len(contact_bodies)):
                idx = i * 3
                if idx + 2 < len(forces):
                    fx = forces[idx]       # Anteroposterior force (X)
                    fz = forces[idx + 2]   # Mediolateral force (Z)
                    if abs(fx) > 1.0 or abs(fz) > 1.0:
                        has_horizontal_grf = True
                        break
            if has_horizontal_grf:
                break

        if not has_horizontal_grf:
            print(f"  Skipping Trial {trial}: no meaningful horizontal GRF measurements detected (X or Z axes).")
            continue

        trial_name = subject.getTrialName(trial) or f"trial_{trial}"
        base_name, segment_idx = parse_trial_segment(trial_name)
        print(f"  Trial {trial} ({trial_name}): passed initial criteria. Extracting (Base: '{base_name}', Segment: {segment_idx})...")

        com_rows = []
        trq_rows = []
        render_rows = []

        # Extract frame metrics sequentially (all frames are exported; edge smoothing is handled at consolidation)
        for t in range(len(frames)):
            frame_data = frames[t]
            
            # Select the last processing pass
            smoothed_pass = frame_data.processingPasses[-1]
            raw_pass = frame_data.processingPasses[0]
            skel.setPositions(smoothed_pass.pos)
            skel.setVelocities(smoothed_pass.vel)
            skel.setAccelerations(smoothed_pass.acc)

            com_pos = skel.getCOM()
            com_vel = skel.getCOMLinearVelocity()
            com_acc = skel.getCOMLinearAcceleration()

            forces = raw_pass.groundContactForce
            cops = raw_pass.groundContactCenterOfPressure
            torques = raw_pass.groundContactTorque

            grf_data = []
            for i in range(len(contact_bodies)):
                idx = i * 3
                body_force = forces[idx:idx+3] if idx < len(forces) else np.zeros(3)
                body_cop = cops[idx:idx+3] if idx < len(cops) else np.zeros(3)
                body_torque = torques[idx:idx+3] if idx < len(torques) else np.zeros(3)
                grf_data.extend([
                    body_force[0], body_force[1], body_force[2],
                    body_cop[0], body_cop[1], body_cop[2],
                    body_torque[0], body_torque[1], body_torque[2]
                ])

            com_row = [
                0, 0.0,
                com_pos[0], com_pos[1], com_pos[2],
                com_vel[0], com_vel[1], com_vel[2],
                com_acc[0], com_acc[1], com_acc[2]
            ] + grf_data
            com_rows.append(com_row)

            trq_row = [0, 0.0]
            for joint in joints:
                joint_world_pos = skel.getJointWorldPositions([joint])
                joint_pos_vector = joint_world_pos[0] if isinstance(joint_world_pos, list) else joint_world_pos[:3]
                trq_row.extend([joint_pos_vector[0], joint_pos_vector[1], joint_pos_vector[2]])
            trq_row.extend(grf_data)
            trq_rows.append(trq_row)

            render_row = [0, 0.0]
            for _, _, _, shape_node in body_shapes:
                world_transform = shape_node.getWorldTransform()
                pos = world_transform.translation()
                rot_matrix = world_transform.rotation()
                euler_angles = nimble.math.matrixToEulerXYZ(rot_matrix)
                
                render_row.extend([pos[0], pos[1], pos[2], euler_angles[0], euler_angles[1], euler_angles[2]])
            render_rows.append(render_row)

        if base_name not in trial_groups:
            trial_groups[base_name] = []
        trial_groups[base_name].append((segment_idx, com_rows, trq_rows, render_rows, dt))

    # Consolidation, boundary blending, and writing of aggregated trials
    for base_name, segments in trial_groups.items():
        # Sort segments based on segment_idx to guarantee chronological order
        segments.sort(key=lambda x: x[0])

        # Verify segments are long enough to undergo boundary interpolation
        valid_group = True
        for _, com_rows, _, _, _ in segments:
            if len(com_rows) < 2 * EDGE_TRIM_FRAMES + 2:
                valid_group = False
                break
        if not valid_group:
            print(f"  Skipping unified trial '{base_name}': one or more segments are too short for boundary blending.")
            continue

        combined_com = []
        combined_trq = []
        combined_render = []
        segment_lengths = []

        total_frames = 0
        cumulative_time = 0.0

        for segment_idx, com_rows, trq_rows, render_rows, dt in segments:
            segment_lengths.append(len(com_rows))
            for idx in range(len(com_rows)):
                r_com = com_rows[idx].copy()
                r_com[0] = total_frames
                r_com[1] = cumulative_time
                combined_com.append(r_com)

                r_trq = trq_rows[idx].copy()
                r_trq[0] = total_frames
                r_trq[1] = cumulative_time
                combined_trq.append(r_trq)

                r_render = render_rows[idx].copy()
                r_render[0] = total_frames
                r_render[1] = cumulative_time
                combined_render.append(r_render)

                total_frames += 1
                cumulative_time += dt

        # Target constraint: Skip and discard trials below 4000 total frames
        if total_frames < 4000:
            print(f"  Skipping unified trial '{base_name}': total frames ({total_frames}) is less than 4000 frame limit.")
            continue

        com_val_path = os.path.join(COM_VAL_DIR, f"{subject_name}_{base_name}_com_validation.parquet")
        torque_val_path = os.path.join(TORQUE_VAL_DIR, f"{subject_name}_{base_name}_torque_validation.parquet")
        rendering_path = os.path.join(RENDERING_DIR, f"{subject_name}_{base_name}_rendering.parquet")

        df_com = pd.DataFrame(combined_com, columns=com_header)
        df_trq = pd.DataFrame(combined_trq, columns=trq_header)
        df_render = pd.DataFrame(combined_render, columns=render_header)

        # 1. Linearly interpolate the inner segment boundary transitions to clean up artifacts
        df_com = interpolate_boundaries(df_com, segment_lengths, EDGE_TRIM_FRAMES)
        df_trq = interpolate_boundaries(df_trq, segment_lengths, EDGE_TRIM_FRAMES)
        df_render = interpolate_boundaries(df_render, segment_lengths, EDGE_TRIM_FRAMES)

        # 2. Trim the absolute outer boundaries of the consolidated dataset to drop start/end filter transients
        df_com = df_com.iloc[EDGE_TRIM_FRAMES:-EDGE_TRIM_FRAMES].reset_index(drop=True)
        df_trq = df_trq.iloc[EDGE_TRIM_FRAMES:-EDGE_TRIM_FRAMES].reset_index(drop=True)
        df_render = df_render.iloc[EDGE_TRIM_FRAMES:-EDGE_TRIM_FRAMES].reset_index(drop=True)

        # 2b. De-bias COM velocity values
        # Since these are treadmill or static trials, the net average velocity is physically zero.
        # This removes small numerical drift/offsets from the biomechanics pipeline.
        for col in ['com_vel_x', 'com_vel_y', 'com_vel_z']:
            if col in df_com.columns:
                df_com[col] = df_com[col] - df_com[col].mean()

        # 3. Correct time and frame sequences to be perfectly sequential
        dt = segments[0][4]
        for df in [df_com, df_trq, df_render]:
            df['frame'] = np.arange(len(df))
            df['time'] = np.arange(len(df)) * dt

        df_com.to_parquet(com_val_path, index=False)
        df_trq.to_parquet(torque_val_path, index=False)
        df_render.to_parquet(rendering_path, index=False)

        print(f"    Saved consolidated & blended trial '{base_name}' ({len(df_com)} frames) ->")
        print(f"      {com_val_path}")
        print(f"      {torque_val_path}")
        print(f"      {rendering_path}")


def main():
    if not download_geometry_if_needed():
        return

    print(f"Scanning directory: {BASE_DIR}")
    b3d_files = get_b3d_files(BASE_DIR)
    
    if not b3d_files:
        print("No B3D files found. Check your BASE_DIR path.")
        return

    print(f"Found {len(b3d_files)} B3D files. Starting export...")
    for b3d_file in b3d_files:
        if 'Fregly' in b3d_file:
            print(f"  Skipping {b3d_file}: identified as Fregly dataset (not compatible with current export process).")
            continue
        if 'Carter' in b3d_file:
            print(f"  Skipping {b3d_file}: identified as Carter dataset (no horizontal GRF).")
            continue
        if 'vanderZee' in b3d_file:
            print(f"  Skipping {b3d_file}: identified as VanderZee dataset (strange horizontal GRF bugs).")
            continue
        export_subject_data(b3d_file, GEOMETRY_FOLDER)

    print("\nProcess completed.")

if __name__ == "__main__":
    main()