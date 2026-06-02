# File: src/online_analyze/energy_analyzer.py
import numpy as np

from online_analyze.com_kf import ComKalmanFilter
from online_analyze.stride_analyzer import StrideAnalyzer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class EnergyAnalyzer:
    """
    Computes real-time overground powers and symmetrically aggregates
    energetics across full stride cycles for both legs, rejecting strides
    polluted by belt-crossing or force plate sharing.
    """

    def __init__(
        self,
        initial_mass=70.0,
        contact_threshold=30.0,
        foot_roll_length=0.254,
        num_gait_points=100,
        override_belt_speed=None,
        enable_diagnostics=True,
    ):
        self.kf = ComKalmanFilter(initial_mass=initial_mass)
        self.stride_analyzer = StrideAnalyzer(
            contact_threshold=contact_threshold,
            foot_roll_length=foot_roll_length,
            override_belt_speed=override_belt_speed,
        )
        self.num_gait_points = num_gait_points
        self.enable_diagnostics = enable_diagnostics

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

        # --- Belt Crossing Rejection Thresholds ---
        self.max_cop_velocity = (
            6.0  # m/s (catches sudden contact shifts on the same plate)
        )
        self.cop_stability_force_threshold = (
            150.0  # N (CoP calculations are highly noisy below this threshold)
        )
        self.max_normalized_force = (
            1.35  # times body weight (catches dual weight on a single plate)
        )

        self.last_cop = {"left": None, "right": None}
        self.last_fy = {"left": 0.0, "right": 0.0}

    def update(self, time, forces, cops, dt, exo_power_left=0.0, exo_power_right=0.0):
        f_total = forces["left"] + forces["right"]
        self.kf.update(f_total, dt)

        was_active = {
            "left": self.stride_analyzer.contact_states["left"],
            "right": self.stride_analyzer.contact_states["right"],
        }
        self.stride_analyzer.update(time, forces, cops)
        is_active = {
            "left": self.stride_analyzer.contact_states["left"],
            "right": self.stride_analyzer.contact_states["right"],
        }

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

                    # --- STRIDE VALIDATION FILTERS ---
                    stride_dur = t_arr[-1] - t_arr[0]
                    v_belt_val = self.stride_analyzer.current_belt_speed
                    stride_len = v_belt_val * stride_dur

                    is_valid_stride = buf.get("is_clean", True)
                    reject_reason = buf.get("reject_reason", None)

                    # Filter A: Biomechanically unrealistic stride length
                    if is_valid_stride and stride_len > 2.0:
                        reject_reason = f"weirdly long stride ({stride_len:.2f} m)"
                        is_valid_stride = False

                    # Filter B: Treadmill belt speed deviation check (if override is set)
                    if (
                        is_valid_stride
                        and self.stride_analyzer.override_belt_speed is not None
                    ):
                        if self.stride_analyzer.belt_speeds:
                            last_measured_speed = self.stride_analyzer.belt_speeds[-1][
                                1
                            ]
                            speed_deviation = abs(
                                last_measured_speed
                                - self.stride_analyzer.override_belt_speed
                            )
                            if speed_deviation > 1.0:
                                reject_reason = f"high speed deviation ({last_measured_speed:.2f} m/s)"
                                is_valid_stride = False

                    # Filter C: Duration bounds check
                    if is_valid_stride and (stride_dur < 0.4 or stride_dur > 1.5):
                        reject_reason = (
                            f"unrealistic temporal duration ({stride_dur:.2f} s)"
                        )
                        is_valid_stride = False

                    # Store only if the stride passes all filters
                    if is_valid_stride:
                        for k in self.stride_profiles.keys():
                            resampled = self._resample_array(t_arr, np.array(buf[k]))
                            self.stride_profiles[k].append(resampled)
                    else:
                        print(
                            f"⚠️ Discarded {foot} stride: {reject_reason or 'unknown artifact'}."
                        )
                        if self.enable_diagnostics:
                            self._plot_stride_diagnostic(foot, buf, reject_reason)

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
                    "reject_reason": None,
                    # --- Diagnostic buffers ---
                    "forces": [],
                    "cops": [],
                    "cop_vels": [],
                }

            # Record states continuously and run frame-by-frame safety validations
            if self.active_strides[foot] is not None:
                buf = self.active_strides[foot]
                fy = forces[foot][1]

                # Verify spatial and force safety constraints during active contact
                if is_active[foot]:
                    # Check 1: Single plate overload (indicates shared weight of both limbs)
                    est_mass = self.kf.mass
                    if est_mass > 0 and est_mass != float("inf"):
                        norm_force = fy / (est_mass * 9.81)
                        if norm_force > self.max_normalized_force:
                            buf["is_clean"] = False
                            buf["reject_reason"] = (
                                f"Force plate overload ({norm_force:.2f} BW)"
                            )

                    # Check 2: Sudden CoP Velocity jump (gated by stable force thresholds)
                    cop_vel = 0.0
                    if self.last_cop[foot] is not None and dt > 0:
                        cop_vel = np.linalg.norm(cops[foot] - self.last_cop[foot]) / dt
                        if (
                            fy > self.cop_stability_force_threshold
                            and self.last_fy[foot] > self.cop_stability_force_threshold
                        ):
                            if cop_vel > self.max_cop_velocity and was_active[foot]:
                                buf["is_clean"] = False
                                buf["reject_reason"] = (
                                    f"CoP velocity jump ({cop_vel:.2f} m/s) under stable load ({fy:.1f} N)"
                                )

                    # Record diagnostic values
                    buf["forces"].append(forces[foot].copy())
                    buf["cops"].append(cops[foot].copy())
                    buf["cop_vels"].append(cop_vel)

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

            # Keep trace of CoP and Force for the next frame
            self.last_cop[foot] = cops[foot].copy()
            self.last_fy[foot] = forces[foot][1]

        return {
            "com_pos": self.kf.com_excursion,
            "com_vel_lab": v_com_lab,
            "com_vel_overground": v_com_overground,
            "sys_left": p_sys_l,
            "sys_right": p_sys_r,
            "exo_left": exo_power_left,
            "exo_right": exo_power_right,
            "hum_left": p_hum_l,
            "hum_right": p_hum_r,
        }

    def _plot_stride_diagnostic(self, foot, buf, reason):
        if plt is None:
            print("Warning: matplotlib is not installed. Cannot show diagnostic plot.")
            return

        # Guard: Ensure we have populated diagnostic lists to plot
        if not buf["forces"] or not buf["cops"] or not buf["cop_vels"]:
            return

        times = np.array(buf["time"][: len(buf["forces"])])
        times = times - times[0]  # Normalize time start to 0.0s

        forces = np.array(buf["forces"])
        cops = np.array(buf["cops"])
        cop_vels = np.array(buf["cop_vels"])

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(
            f"Rejected {foot.upper()} Stride Diagnostic\nReason: {reason}",
            fontsize=11,
            fontweight="bold",
        )

        # Subplot 1: Vertical Force (Fy)
        axs[0].plot(times, forces[:, 1], "r-", label=f"{foot} Vertical Force (Fy)")
        axs[0].axhline(
            self.cop_stability_force_threshold,
            color="gray",
            linestyle="--",
            label="Stability Force Threshold",
        )
        axs[0].set_ylabel("Vertical Force (N)")
        axs[0].legend(loc="upper right", fontsize=8)
        axs[0].grid(True, linestyle=":", alpha=0.6)

        # Subplot 2: Mediolateral Position (Z)
        axs[1].plot(times, cops[:, 2], "b-", label=f"{foot} CoP Z")
        axs[1].axhline(
            0.0,
            color="black",
            linestyle="-",
            alpha=0.5,
            label="Physical Split Divide (0.0)",
        )
        axs[1].set_ylabel("Lateral Position Z (m)")
        axs[1].legend(loc="upper right", fontsize=8)
        axs[1].grid(True, linestyle=":", alpha=0.6)

        # Subplot 3: Instantaneous CoP Velocity
        axs[2].plot(times, cop_vels, "g-", label="Calculated CoP Velocity")
        axs[2].axhline(
            self.max_cop_velocity,
            color="red",
            linestyle="--",
            label="Max Allowed Velocity Threshold",
        )

        # Shade the stable force window region
        stable_region = forces[:, 1] > self.cop_stability_force_threshold
        axs[2].fill_between(
            times,
            0,
            cop_vels,
            where=stable_region,
            color="green",
            alpha=0.1,
            label="Stable Force Zone",
        )

        axs[2].set_ylabel("CoP Velocity (m/s)")
        axs[2].set_xlabel("Time from Stride Start (s)")
        axs[2].legend(loc="upper right", fontsize=8)
        axs[2].grid(True, linestyle=":", alpha=0.6)

        # plt.tight_layout()
        # plt.show()

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
