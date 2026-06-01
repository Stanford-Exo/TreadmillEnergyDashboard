# File: src/online_analyze/stride_analyzer.py
import numpy as np

class StrideAnalyzer:
    """
    Analyzes bilateral force and center of pressure (CoP) data streams to segment
    gait cycles, estimate belt speed, and compute spatial/temporal parameters.
    """
    def __init__(self, contact_threshold=30.0, foot_roll_length=0.254):
        self.contact_threshold = contact_threshold
        self.foot_roll_length = foot_roll_length  # Default 10 inches
        
        self.contact_states = {'left': None, 'right': None}
        self.heel_strike_times = {'left': [], 'right': []}
        self.toe_off_times = {'left': [], 'right': []}
        self.heel_strike_cops = {'left': [], 'right': []}
        self.toe_off_cops = {'left': [], 'right': []}
        
        self.active_stance_start_time = {'left': None, 'right': None}
        self.active_stance_start_cop = {'left': None, 'right': None}
        
        self.last_strike_foot = None
        self.last_strike_time = None
        self.last_strike_cop = None
        
        self.belt_speeds = []
        self.current_belt_speed = 1.25  
        self.belt_velocity = np.array([-1.25, 0.0, 0.0]) # 3D vector for frame translation
        
        self.metrics = {
            'stride_duration': [], 'stride_frequency': [], 'stance_duration': [],
            'swing_duration': [], 'duty_factor': [], 'step_length': [],
            'step_width': [], 'stride_length': []
        }

    def update(self, time, forces, cops):
        for foot in ['left', 'right']:
            fy = forces[foot][1]  
            is_active = fy > self.contact_threshold
            
            if self.contact_states[foot] is None:
                self.contact_states[foot] = is_active
                if is_active:
                    self.active_stance_start_time[foot] = time
                    self.active_stance_start_cop[foot] = cops[foot].copy()
                continue
                
            was_active = self.contact_states[foot]
            
            # Transition: Heel Strike
            if is_active and not was_active:
                self.contact_states[foot] = True
                if self.active_stance_start_time[foot] is not None:
                    stride_dur = time - self.active_stance_start_time[foot]
                    if stride_dur > 0.4:
                        self.metrics['stride_duration'].append(stride_dur)
                        self.metrics['stride_frequency'].append(1.0 / stride_dur)
                        if len(self.toe_off_times[foot]) > 0:
                            last_toe_off = self.toe_off_times[foot][-1]
                            if last_toe_off > self.active_stance_start_time[foot]:
                                stance_dur = last_toe_off - self.active_stance_start_time[foot]
                                self.metrics['stance_duration'].append(stance_dur)
                                self.metrics['swing_duration'].append(time - last_toe_off)
                                self.metrics['duty_factor'].append(stance_dur / stride_dur)
                
                self.active_stance_start_time[foot] = time
                self.active_stance_start_cop[foot] = cops[foot].copy()
                self.heel_strike_times[foot].append(time)
                self.heel_strike_cops[foot].append(cops[foot].copy())
                self.last_strike_foot = foot
                self.last_strike_time = time
                self.last_strike_cop = cops[foot].copy()
                
            # Transition: Toe Off
            elif not is_active and was_active:
                self.contact_states[foot] = False
                self.toe_off_times[foot].append(time)
                self.toe_off_cops[foot].append(cops[foot].copy())
                
                if self.active_stance_start_time[foot] is not None:
                    stance_dur = time - self.active_stance_start_time[foot]
                    if stance_dur > 0.1:
                        cop_start = self.active_stance_start_cop[foot]
                        cop_disp_ap = cops[foot][0] - cop_start[0]
                        
                        # Belt moves backward relative to walking direction
                        d_walk = -np.sign(cop_disp_ap) if cop_disp_ap != 0 else 1.0
                        v_belt_inst = (self.foot_roll_length - (cop_disp_ap * d_walk)) / stance_dur
                        
                        if 0.15 < v_belt_inst < 4.0:
                            self.belt_speeds.append((time, v_belt_inst))
                            recent = [s[1] for s in self.belt_speeds[-10:]]
                            self.current_belt_speed = float(np.mean(recent))
                            # Update 3D vector for downstream power tracking
                            self.belt_velocity = np.array([-d_walk * self.current_belt_speed, 0.0, 0.0])

    def get_metrics_summary(self):
        summary = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_count"] = len(values)
            else:
                summary[f"{key}_mean"] = 0.0
                summary[f"{key}_std"] = 0.0
                summary[f"{key}_count"] = 0
        summary['estimated_belt_speed'] = self.current_belt_speed
        return summary