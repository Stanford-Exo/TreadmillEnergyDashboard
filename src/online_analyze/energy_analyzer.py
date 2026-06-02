# File: src/online_analyze/energy_analyzer.py
import numpy as np

from online_analyze.com_kf import ComKalmanFilter
from online_analyze.stride_analyzer import StrideAnalyzer


class EnergyAnalyzer:
    """
    Computes real-time overground powers and symmetrically aggregates
    energetics across full stride cycles for both legs, relying on
    external cleanliness designations.
    """

    def __init__(
        self,
        initial_mass=70.0,
        contact_threshold=30.0,
        foot_roll_length=0.254,
        num_gait_points=100,
        override_belt_speed=None,
    ):
        self.kf = ComKalmanFilter(initial_mass=initial_mass)
        self.stride_analyzer = StrideAnalyzer(
            contact_threshold=contact_threshold,
            foot_roll_length=foot_roll_length,
            override_belt_speed=override_belt_speed,
        )
        self.num_gait_points = num_gait_points

        self.active_strides = {"left": None, "right": None}
        self.stride_profiles = {
            "ref_sys": [],
            "contra_sys": [],
            "ref_exo": [],
            "contra_exo": [],
            "ref_hum": [],
            "contra_hum": [],
            "com_x": [],
            "com_y": [],
            "com_z": [],
        }

        # API properties for compatibility with visualization server
        self.first_l_strike_seen = False
        self.gait_cycle_buffer = {"time": []}

    def update(
        self,
        time,
        forces,
        cops,
        dt,
        exo_power_left=0.0,
        exo_power_right=0.0,
        is_clean=True,
    ):
        f_total = forces["left"] + forces["right"]
        self.kf.update(f_total, dt)

        was_active = {
            "left": self.stride_analyzer.contact_states["left"],
            "right": self.stride_analyzer.contact_states["right"],
        }
        self.stride_analyzer.update(time, forces, cops, is_clean=is_clean)
        is_active = {
            "left": self.stride_analyzer.contact_states["left"],
            "right": self.stride_analyzer.contact_states["right"],
        }

        # Progress tracking for client server visualization
        if is_active["left"] and not was_active["left"]:
            self.first_l_strike_seen = True
            self.gait_cycle_buffer["time"] = [time]
        elif self.first_l_strike_seen:
            self.gait_cycle_buffer["time"].append(time)

        # 1. Transform CoM Velocity to Overground Frame
        v_com_lab = self.kf.com_velocity
        v_belt = self.stride_analyzer.belt_velocity
        v_com_overground = v_com_lab - v_belt

        # 2. Compute Instantaneous Powers
        p_sys_l = float(np.dot(forces["left"], v_com_overground))
        p_sys_r = float(np.dot(forces["right"], v_com_overground))
        p_hum_l = p_sys_l - exo_power_left
        p_hum_r = p_sys_r - exo_power_right

        # 3. Process Symmetric Full-Stride Buffers
        for foot in ["left", "right"]:
            # Heel Strike: Close previous stride and start a new one
            if is_active[foot] and not was_active[foot]:
                buf = self.active_strides[foot]
                if buf is not None and len(buf["time"]) > 15:
                    t_arr = np.array(buf["time"])
                    stride_dur = t_arr[-1] - t_arr[0]

                    is_valid_stride = buf.get("is_clean", True)
                    if is_valid_stride and (stride_dur < 0.4 or stride_dur > 1.5):
                        is_valid_stride = False

                    if is_valid_stride:
                        for k in self.stride_profiles.keys():
                            resampled = self._resample_array(t_arr, np.array(buf[k]))
                            self.stride_profiles[k].append(resampled)

                self.active_strides[foot] = {
                    "time": [],
                    "ref_sys": [],
                    "contra_sys": [],
                    "ref_exo": [],
                    "contra_exo": [],
                    "ref_hum": [],
                    "contra_hum": [],
                    "com_x": [],
                    "com_y": [],
                    "com_z": [],
                    "is_clean": True,
                }

            # Record states continuously and run frame-by-frame validations
            if self.active_strides[foot] is not None:
                buf = self.active_strides[foot]
                if not is_clean:
                    buf["is_clean"] = False

                ref_sys = p_sys_l if foot == "left" else p_sys_r
                contra_sys = p_sys_r if foot == "left" else p_sys_l
                ref_exo = exo_power_left if foot == "left" else exo_power_right
                contra_exo = exo_power_right if foot == "left" else exo_power_left
                ref_hum = p_hum_l if foot == "left" else p_hum_r
                contra_hum = p_hum_r if foot == "left" else p_hum_l

                com_excursion = self.kf.com_excursion
                buf["time"].append(time)
                buf["ref_sys"].append(ref_sys)
                buf["contra_sys"].append(contra_sys)
                buf["ref_exo"].append(ref_exo)
                buf["contra_exo"].append(contra_exo)
                buf["ref_hum"].append(ref_hum)
                buf["contra_hum"].append(contra_hum)
                buf["com_x"].append(com_excursion[0])
                buf["com_y"].append(com_excursion[1])
                buf["com_z"].append(com_excursion[2])

        return {
            "com_pos": self.kf.com_excursion,
            "com_vel_lab": v_com_lab,
            "com_vel_overground": v_com_overground,
            "power_left": p_sys_l,
            "power_right": p_sys_r,
            "power_total": p_sys_l + p_sys_r,
            "sys_left": p_sys_l,
            "sys_right": p_sys_r,
            "exo_left": exo_power_left,
            "exo_right": exo_power_right,
            "hum_left": p_hum_l,
            "hum_right": p_hum_r,
            "mass": self.kf.mass,
            "tilt": self.kf.tilt_angles,
        }

    def _resample_array(self, times, values):
        if len(times) < 2 or len(values) < 2:
            return np.zeros(self.num_gait_points)
        t_start, t_end = times[0], times[-1]
        if t_end == t_start:
            return np.zeros(self.num_gait_points)

        norm_times = (times - t_start) / (t_end - t_start)
        norm_times = norm_times[: len(values)]
        target_times = np.linspace(0.0, 1.0, self.num_gait_points)
        return np.interp(target_times, norm_times, values)

    def get_stride_aggregates(self):
        """Returns standard arrays representing 0-100% of the Stride Phase."""
        aggregates = {}
        for key, profiles in self.stride_profiles.items():
            if len(profiles) > 0:
                profiles_arr = np.array(profiles)
                aggregates[f"{key}_mean"] = np.mean(profiles_arr, axis=0).tolist()
                aggregates[f"{key}_std"] = np.std(profiles_arr, axis=0).tolist()
            else:
                aggregates[f"{key}_mean"] = np.zeros(self.num_gait_points).tolist()
                aggregates[f"{key}_std"] = np.zeros(self.num_gait_points).tolist()
        return aggregates

    def get_aggregate_profiles(self):
        """Bridge interface mapping ref/contra summaries to Left/Right profiles for visualization."""
        aggs = self.get_stride_aggregates()
        return {
            "left_mean": aggs.get("ref_sys_mean", [0.0] * self.num_gait_points),
            "left_std": aggs.get("ref_sys_std", [0.0] * self.num_gait_points),
            "right_mean": aggs.get("contra_sys_mean", [0.0] * self.num_gait_points),
            "right_std": aggs.get("contra_sys_std", [0.0] * self.num_gait_points),
            "total_mean": [
                l + r
                for l, r in zip(
                    aggs.get("ref_sys_mean", [0.0] * self.num_gait_points),
                    aggs.get("contra_sys_mean", [0.0] * self.num_gait_points),
                )
            ],
            "total_std": aggs.get("ref_sys_std", [0.0] * self.num_gait_points),
        }
