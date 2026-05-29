import numpy as np
from online_analyze.com_kf import ComKalmanFilter
from online_analyze.stride_analyzer import StrideAnalyzer

class EnergyAnalyzer:
    """
    Coordinates state updates and manages real-time power calculations and
    gait cycle normalization.
    """
    def __init__(self, initial_mass=70.0, contact_threshold=30.0, foot_roll_length=0.18, num_gait_points=100):
        self.kf = ComKalmanFilter(initial_mass=initial_mass)
        self.stride_analyzer = StrideAnalyzer(contact_threshold=contact_threshold, foot_roll_length=foot_roll_length)
        self.num_gait_points = num_gait_points
        
        # Buffer to aggregate values during active Left-foot gait cycles
        self.gait_cycle_buffer = {
            'time': [],
            'power_left': [],
            'power_right': [],
            'power_total': []
        }
        self.first_l_strike_seen = False
        
        # Collection of resampled curves
        self.normalized_profiles = {
            'left': [],
            'right': [],
            'total': []
        }

    def update(self, time, forces, cops, dt):
        """
        Processes a single multi-channel data frame.
        
        forces: dict {'left': np.ndarray, 'right': np.ndarray}
        cops: dict {'left': np.ndarray, 'right': np.ndarray}
        dt: float
        """
        # 1. Update COM Kalman filter using composite bilateral force vectors
        f_total = forces['left'] + forces['right']
        self.kf.update(f_total, dt)
        v_com = self.kf.com_velocity
        
        # 2. Segment strides and track steps
        was_l_active = self.stride_analyzer.contact_states['left']
        self.stride_analyzer.update(time, forces, cops)
        is_l_active = self.stride_analyzer.contact_states['left']
        
        # 3. Compute instant powers
        p_left = float(np.dot(forces['left'], v_com))
        p_right = float(np.dot(forces['right'], v_com))
        p_total = p_left + p_right
        
        # 4. Perform Left-Foot strike normalization
        l_strike_detected = is_l_active and (was_l_active is False)
        
        if l_strike_detected:
            if self.first_l_strike_seen:
                t_arr = np.array(self.gait_cycle_buffer['time'])
                p_l_arr = np.array(self.gait_cycle_buffer['power_left'])
                p_r_arr = np.array(self.gait_cycle_buffer['power_right'])
                p_t_arr = np.array(self.gait_cycle_buffer['power_total'])
                
                # Only accept cycles with sufficient temporal samples
                if len(t_arr) > 15:
                    self.normalized_profiles['left'].append(self._resample_gait_cycle(t_arr, p_l_arr))
                    self.normalized_profiles['right'].append(self._resample_gait_cycle(t_arr, p_r_arr))
                    self.normalized_profiles['total'].append(self._resample_gait_cycle(t_arr, p_t_arr))
            
            # Restart accumulation sequence
            self.first_l_strike_seen = True
            self.gait_cycle_buffer = {
                'time': [time],
                'power_left': [p_left],
                'power_right': [p_right],
                'power_total': [p_total]
            }
        else:
            if self.first_l_strike_seen:
                self.gait_cycle_buffer['time'].append(time)
                self.gait_cycle_buffer['power_left'].append(p_left)
                self.gait_cycle_buffer['power_right'].append(p_right)
                self.gait_cycle_buffer['power_total'].append(p_total)
                
        return {
            'com_pos': self.kf.com_excursion,
            'com_vel': v_com,
            'power_left': p_left,
            'power_right': p_right,
            'power_total': p_total,
            'mass': self.kf.mass,
            'tilt': self.kf.tilt_angles
        }

    def _resample_gait_cycle(self, times, values):
        """Linearly interpolates time-series vectors over a fixed grid size."""
        if len(times) < 2:
            return np.zeros(self.num_gait_points)
        t_start, t_end = times[0], times[-1]
        if t_end == t_start:
            return np.zeros(self.num_gait_points)
            
        norm_times = (times - t_start) / (t_end - t_start)
        target_times = np.linspace(0.0, 1.0, self.num_gait_points)
        return np.interp(target_times, norm_times, values)

    def get_aggregate_profiles(self):
        """Compiles mean curves and standard deviation ribbons over all completed strides."""
        aggregates = {}
        for key in ['left', 'right', 'total']:
            profiles = self.normalized_profiles[key]
            if len(profiles) > 0:
                profiles_arr = np.array(profiles)
                aggregates[f"{key}_mean"] = np.mean(profiles_arr, axis=0).tolist()
                aggregates[f"{key}_std"] = np.std(profiles_arr, axis=0).tolist()
            else:
                aggregates[f"{key}_mean"] = np.zeros(self.num_gait_points).tolist()
                aggregates[f"{key}_std"] = np.zeros(self.num_gait_points).tolist()
        return aggregates