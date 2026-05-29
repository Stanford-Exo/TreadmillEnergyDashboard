import os
import sys
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

def export_subject_data(file_path: str, geometry_dir: str):
    """Processes a B3D file and exports filtered splits as Parquet files."""
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

    # Process trials within the subject file
    for trial in range(subject.getNumTrials()):
        # Constraint 1: Keep only trials that are designated as TREADMILL
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
        print(f"  Trial {trial}: passed all constraints, exporting data...")

        trial_name = subject.getTrialName(trial) or f"trial_{trial}"
        print(f"  Processing Trial: {trial_name} ({trial_len} frames)")

        # Target Parquet output paths
        com_val_path = os.path.join(COM_VAL_DIR, f"{subject_name}_{trial_name}_com_validation.parquet")
        torque_val_path = os.path.join(TORQUE_VAL_DIR, f"{subject_name}_{trial_name}_torque_validation.parquet")
        rendering_path = os.path.join(RENDERING_DIR, f"{subject_name}_{trial_name}_rendering.parquet")

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

        # Data rows buffers
        com_rows = []
        trq_rows = []
        render_rows = []

        # Iterate through frame metrics
        for t in range(len(frames)):
            frame_data = frames[t]
            time_val = t * dt
            
            # Select the last processing pass (index 2)
            smoothed_pass = frame_data.processingPasses[-1]
            skel.setPositions(smoothed_pass.pos)
            skel.setVelocities(smoothed_pass.vel)
            skel.setAccelerations(smoothed_pass.acc)

            # Retrieve untransformed kinematic metrics
            # com_pos = skel.getCOM()
            # com_vel = skel.getCOMLinearVelocity()
            com_acc = skel.getCOMLinearAcceleration()

            com_pos = smoothed_pass.comPos
            com_vel = smoothed_pass.comVel

            forces = smoothed_pass.groundContactForce
            cops = smoothed_pass.groundContactCenterOfPressure
            torques = smoothed_pass.groundContactTorque

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

            # --- Row 1: COM Validation ---
            com_row = [
                t, time_val,
                com_pos[0], com_pos[1], com_pos[2],
                com_vel[0], com_vel[1], com_vel[2],
                com_acc[0], com_acc[1], com_acc[2]
            ] + grf_data
            com_rows.append(com_row)

            # --- Row 2: Torque Validation ---
            trq_row = [t, time_val]
            for joint in joints:
                joint_world_pos = skel.getJointWorldPositions([joint])
                joint_pos_vector = joint_world_pos[0] if isinstance(joint_world_pos, list) else joint_world_pos[:3]
                trq_row.extend([joint_pos_vector[0], joint_pos_vector[1], joint_pos_vector[2]])
            trq_row.extend(grf_data)
            trq_rows.append(trq_row)

            # --- Row 3: Rendering ---
            render_row = [t, time_val]
            for _, _, _, shape_node in body_shapes:
                world_transform = shape_node.getWorldTransform()
                pos = world_transform.translation()
                rot_matrix = world_transform.rotation()
                euler_angles = nimble.math.matrixToEulerXYZ(rot_matrix)
                
                render_row.extend([pos[0], pos[1], pos[2], euler_angles[0], euler_angles[1], euler_angles[2]])
            render_rows.append(render_row)

        # Convert buffers to DataFrames and save as Parquet files
        df_com = pd.DataFrame(com_rows, columns=com_header)
        df_com.to_parquet(com_val_path, index=False)

        df_trq = pd.DataFrame(trq_rows, columns=trq_header)
        df_trq.to_parquet(torque_val_path, index=False)

        df_render = pd.DataFrame(render_rows, columns=render_header)
        df_render.to_parquet(rendering_path, index=False)

        print(f"    Saved -> {com_val_path}")
        print(f"    Saved -> {torque_val_path}")
        print(f"    Saved -> {rendering_path}")


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
        export_subject_data(b3d_file, GEOMETRY_FOLDER)

    print("\nProcess completed.")

if __name__ == "__main__":
    main()