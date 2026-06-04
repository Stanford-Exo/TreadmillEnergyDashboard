use nalgebra::Vector3;

#[derive(Clone, Default)]
pub struct FootState {
    pub is_active: Option<bool>,
    pub active_stance_start_time: Option<f64>,
    pub active_stance_start_cop: Option<Vector3<f64>>,
    pub toe_off_times: Vec<f64>,
}

pub struct StrideAnalyzer {
    pub contact_threshold: f64,
    pub foot_roll_length: f64,
    pub override_belt_speed: Option<f64>,
    pub current_belt_speed: f64,
    pub left: FootState,
    pub right: FootState,
    pub belt_speeds: Vec<f64>,
}

impl StrideAnalyzer {
    pub fn new(
        contact_threshold: f64,
        foot_roll_length: f64,
        override_belt_speed: Option<f64>,
    ) -> Self {
        Self {
            contact_threshold,
            foot_roll_length,
            override_belt_speed,
            current_belt_speed: override_belt_speed.unwrap_or(1.25),
            left: FootState::default(),
            right: FootState::default(),
            belt_speeds: Vec::new(),
        }
    }

    pub fn belt_velocity(&self) -> Vector3<f64> {
        Vector3::new(-self.current_belt_speed, 0.0, 0.0)
    }

    pub fn update(
        &mut self,
        time: f64,
        forces_l: Vector3<f64>,
        forces_r: Vector3<f64>,
        cops_l: Vector3<f64>,
        cops_r: Vector3<f64>,
    ) {
        self.process_foot("left", time, forces_l, cops_l);
        self.process_foot("right", time, forces_r, cops_r);
    }

    fn process_foot(&mut self, side: &str, time: f64, force: Vector3<f64>, cop: Vector3<f64>) {
        let is_active = force.y > self.contact_threshold;
        let foot = if side == "left" {
            &mut self.left
        } else {
            &mut self.right
        };

        if foot.is_active.is_none() {
            foot.is_active = Some(is_active);
            if is_active {
                foot.active_stance_start_time = Some(time);
                foot.active_stance_start_cop = Some(cop);
            }
            return;
        }

        let was_active = foot.is_active.unwrap();

        if is_active && !was_active {
            foot.is_active = Some(true);
            foot.active_stance_start_time = Some(time);
            foot.active_stance_start_cop = Some(cop);
        } else if !is_active && was_active {
            foot.is_active = Some(false);
            foot.toe_off_times.push(time);

            if let (Some(start_time), Some(start_cop)) =
                (foot.active_stance_start_time, foot.active_stance_start_cop)
            {
                let stance_dur = time - start_time;
                if stance_dur > 0.1 {
                    let cop_disp_ap = cop.x - start_cop.x;
                    let d_walk = if cop_disp_ap != 0.0 {
                        -cop_disp_ap.signum()
                    } else {
                        1.0
                    };
                    let v_belt_inst = (self.foot_roll_length - (cop_disp_ap * d_walk)) / stance_dur;

                    if v_belt_inst > 0.15 && v_belt_inst < 4.0 {
                        self.belt_speeds.push(v_belt_inst);
                        if let Some(over) = self.override_belt_speed {
                            self.current_belt_speed = over;
                        } else {
                            let recent: Vec<f64> =
                                self.belt_speeds.iter().rev().take(10).cloned().collect();
                            self.current_belt_speed =
                                recent.iter().sum::<f64>() / recent.len() as f64;
                        }
                    }
                }
            }
        }
    }
}
