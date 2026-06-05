use crate::com_kf::ComKalmanFilter;
use crate::stride_analyzer::StrideAnalyzer;
use nalgebra::Vector3;

pub struct EnergyAnalyzer {
    pub kf: ComKalmanFilter,
    pub stride_analyzer: StrideAnalyzer,
}

pub struct EnergyResult {
    pub power_left: f64,
    pub power_right: f64,
    pub power_sys_x: f64,
    pub power_sys_y: f64,
    pub power_sys_z: f64,
    pub com_pos: Vector3<f64>,
    pub mass: f64,
}

impl EnergyAnalyzer {
    pub fn new(
        initial_mass: f64,
        contact_threshold: f64,
        foot_roll_length: f64,
        override_belt_speed: Option<f64>,
    ) -> Self {
        Self {
            kf: ComKalmanFilter::new(
                initial_mass,
                0.1,
                0.1,
                10.0,
                0.003,
                0.005,
                1e-5,
                1e-4,
                1e-10,
                1e-11,
            ),
            stride_analyzer: StrideAnalyzer::new(
                contact_threshold,
                foot_roll_length,
                override_belt_speed,
            ),
        }
    }

    pub fn update(
        &mut self,
        time: f64,
        force_l: Vector3<f64>,
        force_r: Vector3<f64>,
        cop_l: Vector3<f64>,
        cop_r: Vector3<f64>,
        dt: f64,
    ) -> EnergyResult {
        let f_total = force_l + force_r;
        self.kf.update(f_total, dt);
        self.stride_analyzer
            .update(time, force_l, force_r, cop_l, cop_r);

        let v_com_lab = self.kf.com_velocity();
        let v_belt = self.stride_analyzer.belt_velocity();
        let v_com_overground = v_com_lab - v_belt;

        EnergyResult {
            power_left: force_l.dot(&v_com_overground),
            power_right: force_r.dot(&v_com_overground),
            power_sys_x: (force_l.x + force_r.x) * v_com_overground.x,
            power_sys_y: (force_l.y + force_r.y) * v_com_overground.y,
            power_sys_z: (force_l.z + force_r.z) * v_com_overground.z,
            com_pos: self.kf.com_excursion(),
            mass: self.kf.mass(),
        }
    }
}