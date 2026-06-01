# File: src/online_analyze/energy_analyzer.py
import numpy as np
from online_analyze.com_kf import ComKalmanFilter
from online_analyze.stride_analyzer import StrideAnalyzer

class EnergyAnalyzer:
    """
    Computes real-time overground powers and symmetrically aggregates 
    energetics across full stride cycles for both legs.
    """
    def __init__(self, initial_mass=70.0, contact_threshold=30.0, foot_roll_length=0.254, num_gait_points=100, override_belt_speed=None):
        self.kf = ComKalmanFilter(initial_mass=initial_mass)
        self.stride_analyzer = StrideAnalyzer(contact_threshold=contact_threshold, foot_roll_length=foot_roll_length, override_belt_speed=override_belt_speed)
        self.num_gait_points = num_gait_points
        
        self.active_strides = {'left': None, 'right': None}
        self.stride_profiles = {
            'ref_sys': [], 'contra_sys': [],
            'ref_exo': [], 'contra_exo': [],
            'ref_hum': [], 'contra_hum': [],
            'com_x': [], 'com_y': [], 'com_z': []
        }

    def update(self, time, forces, cops, dt, exo_power_left=0.0, exo_power_right=0.0):
        f_total = forces['left'] + forces['right']
        self.kf.update(f_total, dt)
        
        was_active = {'left': self.stride_analyzer.contact_states['left'], 
                      'right': self.stride_analyzer.contact_states['right']}
        self.stride_analyzer.update(time, forces, cops)
        is_active = {'left': self.stride_analyzer.contact_states['left'], 
                     'right': self.stride_analyzer.contact_states['right']}
        
        # 1. Transform CoM Velocity to Overground Frame
        v_com_lab = self.kf.com_velocity
        v_belt = self.stride_analyzer.belt_velocity
        v_com_overground = v_com_lab - v_belt 
        
        # 2. Compute Instantaneous Powers
        p_sys_l = float(np.dot(forces['left'], v_com_overground))
        p_sys_r = float(np.dot(forces['right'], v_com_overground))
        p_hum_l = p_sys_l - exo_power_left
        p_hum_r = p_sys_r - exo_power_right
        
        # 3. Process Symmetric Full-Stride Buffers
        for foot in ['left', 'right']:
            # Heel Strike: Close previous stride and start a new one
            if is_active[foot] and not was_active[foot]:
                buf = self.active_strides[foot]
                if buf is not None and len(buf['time']) > 15:
                    t_arr = np.array(buf['time'])
                    for k in self.stride_profiles.keys():
                        resampled = self._resample_array(t_arr, np.array(buf[k]))
                        self.stride_profiles[k].append(resampled)
                        
                self.active_strides[foot] = {
                    'time': [],
                    'ref_sys': [], 'contra_sys': [],
                    'ref_exo': [], 'contra_exo': [],
                    'ref_hum': [], 'contra_hum': [],
                    'com_x': [], 'com_y': [], 'com_z': []
                }
                
            # Record states continuously during the stride
            if self.active_strides[foot] is not None:
                ref_sys = p_sys_l if foot == 'left' else p_sys_r
                contra_sys = p_sys_r if foot == 'left' else p_sys_l
                ref_exo = exo_power_left if foot == 'left' else exo_power_right
                contra_exo = exo_power_right if foot == 'left' else exo_power_left
                ref_hum = p_hum_l if foot == 'left' else p_hum_r
                contra_hum = p_hum_r if foot == 'left' else p_hum_l
                
                com_excursion = self.kf.com_excursion
                buf = self.active_strides[foot]
                buf['time'].append(time)
                buf['ref_sys'].append(ref_sys)
                buf['contra_sys'].append(contra_sys)
                buf['ref_exo'].append(ref_exo)
                buf['contra_exo'].append(contra_exo)
                buf['ref_hum'].append(ref_hum)
                buf['contra_hum'].append(contra_hum)
                buf['com_x'].append(com_excursion[0])
                buf['com_y'].append(com_excursion[1])
                buf['com_z'].append(com_excursion[2])
                
        return {
            'com_pos': self.kf.com_excursion,
            'com_vel_lab': v_com_lab,
            'com_vel_overground': v_com_overground,
            'sys_left': p_sys_l, 'sys_right': p_sys_r, 
            'exo_left': exo_power_left, 'exo_right': exo_power_right,
            'hum_left': p_hum_l, 'hum_right': p_hum_r
        }

    def _resample_array(self, times, values):
        if len(times) < 2 or len(values) < 2:
            return np.zeros(self.num_gait_points)
        t_start, t_end = times[0], times[-1]
        if t_end == t_start:
            return np.zeros(self.num_gait_points)
        
        norm_times = (times - t_start) / (t_end - t_start)
        norm_times = norm_times[:len(values)] 
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