import numpy as np

class StrideAnalyzer:
    """
    Analyzes bilateral force and center of pressure (CoP) data streams to segment
    gait cycles and compute spatial/temporal parameters.
    """
    def __init__(self, contact_threshold=30.0, foot_roll_length=0.18):
        self.contact_threshold = contact_threshold
        self.foot_roll_length = foot_roll_length
        
        # Current contact states (initialized dynamically)
        self.contact_states = {'left': None, 'right': None}
        
        # Event historical timestamps
        self.heel_strike_times = {'left': [], 'right': []}
        self.toe_off_times = {'left': [], 'right': []}
        
        # Spatial endpoints
        self.heel_strike_cops = {'left': [], 'right': []}
        self.toe_off_cops = {'left': [], 'right': []}
        
        # Running segment references
        self.active_stance_start_time = {'left': None, 'right': None}
        self.active_stance_start_cop = {'left': None, 'right': None}
        
        # Inter-stride references for step calculations
        self.last_strike_foot = None
        self.last_strike_time = None
        self.last_strike_cop = None
        
        # Running belt speed estimations
        self.belt_speeds = []
        self.current_belt_speed = 1.25  # Sensible fallback speed
        
        # Target gait metrics
        self.metrics = {
            'stride_duration': [],
            'stride_frequency': [],
            'stance_duration': [],
            'swing_duration': [],
            'duty_factor': [],
            'step_length': [],
            'step_width': [],
            'stride_length': []
        }

    def update(self, time, forces, cops):
        """
        Updates trackers with high frequency force and Center of Pressure coordinates.
        
        forces: dict {'left': np.ndarray([Fx, Fy, Fz]), 'right': np.ndarray([Fx, Fy, Fz])}
        cops: dict {'left': np.ndarray([Cx, Cy, Cz]), 'right': np.ndarray([Cx, Cy, Cz])}
        """
        for foot in ['left', 'right']:
            fy = forces[foot][1]  # Vertical component (Y axis)
            is_active = fy > self.contact_threshold
            
            # Initialize state on first frame
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
                
                # Retrieve previous heel strike of same foot to finalize stride metrics
                if self.active_stance_start_time[foot] is not None:
                    stride_dur = time - self.active_stance_start_time[foot]
                    if stride_dur > 0.4:  # Plausibility threshold to ignore chatter
                        self.metrics['stride_duration'].append(stride_dur)
                        self.metrics['stride_frequency'].append(1.0 / stride_dur)
                        
                        # Calculate split parameters if stance offset was captured
                        if len(self.toe_off_times[foot]) > 0:
                            last_toe_off = self.toe_off_times[foot][-1]
                            if last_toe_off > self.active_stance_start_time[foot]:
                                stance_dur = last_toe_off - self.active_stance_start_time[foot]
                                swing_dur = time - last_toe_off
                                
                                self.metrics['stance_duration'].append(stance_dur)
                                self.metrics['swing_duration'].append(swing_dur)
                                self.metrics['duty_factor'].append(stance_dur / stride_dur)
                                
                                # Stride length estimation factoring in belt translation
                                last_landing_cop = self.active_stance_start_cop[foot]
                                d_cop_ap = cops[foot][0] - last_landing_cop[0]
                                
                                d_walk = 1.0
                                if len(self.toe_off_cops[foot]) > 0 and len(self.heel_strike_cops[foot]) > 0:
                                    last_ap_disp = self.toe_off_cops[foot][-1][0] - self.heel_strike_cops[foot][-1][0]
                                    d_walk = -np.sign(last_ap_disp) if last_ap_disp != 0 else 1.0
                                    
                                stride_len = self.current_belt_speed * stride_dur + d_cop_ap * d_walk
                                self.metrics['stride_length'].append(abs(stride_len))
                
                # Cross-foot metrics
                if self.last_strike_foot is not None and self.last_strike_foot != foot:
                    dt = time - self.last_strike_time
                    if dt > 0.1:
                        cop_diff_ap = cops[foot][0] - self.last_strike_cop[0]
                        cop_diff_ml = cops[foot][2] - self.last_strike_cop[2]
                        
                        d_walk = 1.0
                        if len(self.toe_off_cops[foot]) > 0 and len(self.heel_strike_cops[foot]) > 0:
                            last_ap_disp = self.toe_off_cops[foot][-1][0] - self.heel_strike_cops[foot][-1][0]
                            d_walk = -np.sign(last_ap_disp) if last_ap_disp != 0 else 1.0
                            
                        step_len = abs(cop_diff_ap + d_walk * self.current_belt_speed * dt)
                        step_wid = abs(cop_diff_ml)
                        
                        self.metrics['step_length'].append(step_len)
                        self.metrics['step_width'].append(step_wid)
                
                # Reset contact boundaries
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
                
                # Process belt speed updates
                if self.active_stance_start_time[foot] is not None:
                    stance_dur = time - self.active_stance_start_time[foot]
                    if stance_dur > 0.1:
                        cop_start = self.active_stance_start_cop[foot]
                        cop_disp_ap = cops[foot][0] - cop_start[0]
                        
                        d_walk = -np.sign(cop_disp_ap) if cop_disp_ap != 0 else 1.0
                        v_belt_inst = (self.foot_roll_length - (cop_disp_ap * d_walk)) / stance_dur
                        
                        # Validate and add to rolling filter
                        if 0.15 < v_belt_inst < 4.0:
                            self.belt_speeds.append((time, v_belt_inst))
                            recent = [s[1] for s in self.belt_speeds[-10:]]
                            self.current_belt_speed = float(np.mean(recent))

    def get_metrics_summary(self):
        """Compiles running means, standard deviations, and step counts."""
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