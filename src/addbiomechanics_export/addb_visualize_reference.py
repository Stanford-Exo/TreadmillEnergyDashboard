import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import List
import nimblephysics as nimble
import random
import numpy as np
import asyncio
import signal
# Import necessary libraries for CSV writing and path manipulation
import csv
from pathlib import Path

gui = nimble.NimbleGUI()
keep_running = True


async def process_grf_data():
    global keep_running
    await asyncio.sleep(1.0)

    # Define the dataset and data loader
    base_dir = '/Users/keenonwerling/Desktop/data/addb_dataset_publication/'

    geometry_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Geometry') + '/'

    # List all the B3D files under the standardized folder
    raw_files: List[str] = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_name = os.path.join(root, file)
            print(file_name)
            # if file.endswith('.b3d') and 'Camargo' in file_name and 'split5' in file_name:
            if file.endswith('.b3d'):
                # If the file is not empty
                if os.path.getsize(file_name) > 0:
                    raw_files.append(file_name)

    # Load the files
    print(f'Found {len(raw_files)} files.')

    # random.seed(42)
    # random.shuffle(raw_files)
    # raw_files = raw_files[:10]

    gui_port: int = 8080
    osim = nimble.RajagopalHumanBodyModel()
    skel = osim.skeleton
    gui_port = gui_port
    # Make the GUI
    gui.nativeAPI().renderSkeleton(skel)

    gui.serve(gui_port)

    gui.nativeAPI().createLine('global_x', [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
                               np.array([1.0, 0.0, 0.0, 1.0]))
    gui.nativeAPI().createLine('global_y', [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
                               np.array([0.0, 1.0, 0.0, 1.0]))
    gui.nativeAPI().createLine('global_z', [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])],
                               np.array([0.0, 0.0, 1.0, 1.0]))

    for f, file_path in enumerate(raw_files):
        if not keep_running:
            break

        print(f'Processing file {f}/{len(raw_files)}')
        print(file_path)
        subject = nimble.biomechanics.SubjectOnDisk(file_path)

        if f == 0:
            print(f'Reading skel')
            skel = subject.readSkel(subject.getNumProcessingPasses() - 1, ignoreGeometry=False,
                                    geometryFolder=geometry_folder)

        left_foot_index = subject.getGroundForceBodies().index('calcn_l')
        left_foot_body = skel.getBodyNode('calcn_l')
        right_foot_index = subject.getGroundForceBodies().index('calcn_r')
        right_foot_body = skel.getBodyNode('calcn_r')

        for i in range(skel.getNumBodyNodes()):
            body = skel.getBodyNode(i)
            print(f'Body {i}: {body.getName()}')

        torso = skel.getBodyNode('torso')
        left_femur = skel.getBodyNode('femur_l')
        right_femur = skel.getBodyNode('femur_r')
        left_tibia = skel.getBodyNode('tibia_l')
        right_tibia = skel.getBodyNode('tibia_r')

        for i in range(skel.getNumJoints()):
            joint = skel.getJoint(i)
            print(f'Joint {i}: {joint.getName()}')

        left_knee = skel.getJoint('walker_knee_l')

        for trial in range(subject.getNumTrials()):
            if not keep_running:
                break

            print(f'Processing trial {trial}/{subject.getNumTrials()}')
            if subject.getTrialNumProcessingPasses(trial) == 0:
                continue
            frames = subject.readFrames(trial, 0, subject.getTrialLength(trial), includeSensorData=False,
                                        includeProcessingPasses=True)

            dt = subject.getTrialTimestep(trial)
            for t in range(len(frames)):
                if not keep_running:
                    break

                frame: nimble.biomechanics.Frame = frames[t]

                pos = frame.processingPasses[1].pos.copy()
                # pos[:] = 0.0
                pos[3:6] = 0.0
                skel.setPositions(pos)
                gui.nativeAPI().renderSkeleton(skel)

                com = skel.getCOM()
                com_velocity = skel.getCOMLinearVelocity()
                com_acceleration = skel.getCOMLinearAcceleration()

                left_joint_pos = skel.getJointWorldPositions([left_knee])

                gui.nativeAPI().createSphere('com', 0.05 * np.ones(3), com, np.array([1.0, 1.0, 0.0, 1.0]))
                gui.nativeAPI().createSphere('left_knee', 0.05 * np.ones(3), left_joint_pos, np.array([1.0, 0.0, 1.0, 1.0]))

                smoothed_pass = frame.processingPasses[1]
                forces = smoothed_pass.groundContactForce
                body_form_norms = []
                for i in range(int(len(forces) / 3)):
                    body_form_norms.append(np.linalg.norm(forces[i * 3:i * 3 + 3]))

                if sum(body_form_norms) > 0:
                    left_foot_percentage_of_total = body_form_norms[left_foot_index] / sum(body_form_norms)
                else:
                    left_foot_percentage_of_total = 0.5

                gui.nativeAPI().createSphere('left_foot_contact', left_foot_percentage_of_total * 0.1 * np.ones(3),
                                                left_foot_body.getWorldTransform().translation(),
                                                np.array([1.0, 0.0, 0.0, 0.2]))
                gui.nativeAPI().createSphere('right_foot_contact',
                                                (1.0 - left_foot_percentage_of_total) * 0.1 * np.ones(3),
                                                right_foot_body.getWorldTransform().translation(),
                                                np.array([1.0, 0.0, 0.0, 0.2]))

                left_femur_transform = left_femur.getWorldTransform()
                left_femur_pos = left_femur_transform.translation()
                left_femur_mixamo_offset = nimble.math.eulerXYZToMatrix([0.0, np.pi / 2.0, np.pi])
                left_femur_mixamo_rotation = left_femur_transform.rotation() @ left_femur_mixamo_offset
                # left_femur_euler = nimble.math.matrixToEulerXYZ(left_femur_transform.rotation())
                # gui.nativeAPI().createBox('left_femur_box', 0.22 * np.ones(3), left_femur_pos, left_femur_euler, np.array([0.0, 0.0, 1.0, 0.2]))
                left_femur_unit_x = left_femur_pos + left_femur_mixamo_rotation @ np.array([1.0, 0.0, 0.0])
                gui.nativeAPI().createLine('left_femur_x', [left_femur_pos, left_femur_unit_x],
                                            np.array([1.0, 0.0, 0.0, 0.5]))
                left_femur_unit_y = left_femur_pos + left_femur_mixamo_rotation @ np.array([0.0, 1.0, 0.0])
                gui.nativeAPI().createLine('left_femur_y', [left_femur_pos, left_femur_unit_y],
                                            np.array([0.0, 1.0, 0.0, 0.5]))
                left_femur_unit_z = left_femur_pos + left_femur_mixamo_rotation @ np.array([0.0, 0.0, 1.0])
                gui.nativeAPI().createLine('left_femur_z', [left_femur_pos, left_femur_unit_z],
                                            np.array([0.0, 0.0, 1.0, 0.5]))

                left_tibia_transform = left_tibia.getWorldTransform()
                left_tibia_pos = left_tibia_transform.translation()
                left_tibia_mixamo_offset = nimble.math.eulerXYZToMatrix([0.0, np.pi / 2.0, np.pi])
                left_tibia_mixamo_rotation = left_tibia_transform.rotation() @ left_tibia_mixamo_offset
                # left_tibia_euler = nimble.math.matrixToEulerXYZ(left_tibia_transform.rotation())
                # gui.nativeAPI().createBox('left_tibia_box', 0.22 * np.ones(3), left_tibia_pos, left_tibia_euler, np.array([0.0, 1.0, 0.0, 0.2]))
                left_tibia_unit_x = left_tibia_pos + left_tibia_mixamo_rotation @ np.array([1.0, 0.0, 0.0])
                gui.nativeAPI().createLine('left_tibia_x', [left_tibia_pos, left_tibia_unit_x],
                                            np.array([1.0, 0.0, 0.0, 0.5]))
                left_tibia_unit_y = left_tibia_pos + left_tibia_mixamo_rotation @ np.array([0.0, 1.0, 0.0])
                gui.nativeAPI().createLine('left_tibia_y', [left_tibia_pos, left_tibia_unit_y],
                                            np.array([0.0, 1.0, 0.0, 0.5]))
                left_tibia_unit_z = left_tibia_pos + left_tibia_mixamo_rotation @ np.array([0.0, 0.0, 1.0])
                gui.nativeAPI().createLine('left_tibia_z', [left_tibia_pos, left_tibia_unit_z],
                                            np.array([0.0, 0.0, 1.0, 0.5]))

                torso_transform = torso.getWorldTransform()
                torso_pos = torso_transform.translation()
                torso_offset = nimble.math.eulerXYZToMatrix([0.0, np.pi / 2.0, 0.0])
                torso_mixamo_rotation = torso_transform.rotation() @ torso_offset
                # gui.nativeAPI().createBox('torso_box', 0.3 * np.ones(3), torso_pos, torso_euler, np.array([1.0, 0.0, 0.0, 0.2]))
                torso_unit_x = torso_pos + torso_mixamo_rotation @ np.array([1.0, 0.0, 0.0])
                gui.nativeAPI().createLine('torso_x', [torso_pos, torso_unit_x], np.array([1.0, 0.0, 0.0, 0.5]))
                torso_unit_y = torso_pos + torso_mixamo_rotation @ np.array([0.0, 1.0, 0.0])
                gui.nativeAPI().createLine('torso_y', [torso_pos, torso_unit_y], np.array([0.0, 1.0, 0.0, 0.5]))
                torso_unit_z = torso_pos + torso_mixamo_rotation @ np.array([0.0, 0.0, 1.0])
                gui.nativeAPI().createLine('torso_z', [torso_pos, torso_unit_z], np.array([0.0, 0.0, 1.0, 0.5]))

                csv_row = {}
                csv_row['trial'] = trial
                csv_row['time'] = t * dt
                csv_row['left_foot_grf'] = body_form_norms[left_foot_index]
                csv_row['right_foot_grf'] = body_form_norms[right_foot_index]

                await asyncio.sleep(subject.getTrialTimestep(trial))


async def shutdown():
    global keep_running
    print("Shutting down...")
    keep_running = False
    gui.stopServing()


for sig in (signal.SIGINT, signal.SIGTERM):
    asyncio.get_event_loop().add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

asyncio.get_event_loop().run_until_complete(asyncio.gather(
    process_grf_data()
))