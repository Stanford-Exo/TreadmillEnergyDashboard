use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

pub struct ComKalmanFilter {
    pub x: SVector<f64, 9>,
    pub p: SMatrix<f64, 9, 9>,
    q: SMatrix<f64, 9, 9>,
    r: Matrix3<f64>,
    g: Vector3<f64>,
}

impl ComKalmanFilter {
    pub fn new(
        initial_mass: f64,
        pos_std: f64,
        vel_std: f64,
        mass_std: f64,
        tilt_std_rad: f64,
        meas_std: f64,
        proc_noise_pos: f64,
        proc_noise_vel: f64,
        proc_noise_w: f64,
        proc_noise_phi: f64,
    ) -> Self {
        let mut x = SVector::<f64, 9>::zeros();
        x[6] = 1.0 / initial_mass;

        let mut p = SMatrix::<f64, 9, 9>::zeros();
        for i in 0..3 {
            p[(i, i)] = pos_std.powi(2);
        }
        for i in 3..6 {
            p[(i, i)] = vel_std.powi(2);
        }

        let w_var = (1.0 / initial_mass.powi(2)).powi(2) * mass_std.powi(2);
        p[(6, 6)] = w_var;

        let phi_var = (x[6] * tilt_std_rad).powi(2);
        p[(7, 7)] = phi_var;
        p[(8, 8)] = phi_var;

        let mut q = SMatrix::<f64, 9, 9>::zeros();
        for i in 0..3 {
            q[(i, i)] = proc_noise_pos;
        }
        for i in 3..6 {
            q[(i, i)] = proc_noise_vel;
        }
        q[(6, 6)] = proc_noise_w;
        q[(7, 7)] = proc_noise_phi;
        q[(8, 8)] = proc_noise_phi;

        let mut r = Matrix3::<f64>::zeros();
        for i in 0..3 {
            r[(i, i)] = meas_std.powi(2);
        }

        Self {
            x,
            p,
            q,
            r,
            g: Vector3::new(0.0, -9.81, 0.0),
        }
    }

    pub fn update(&mut self, f_m: Vector3<f64>, dt: f64) {
        let mut phi = SMatrix::<f64, 9, 9>::identity();
        for i in 0..3 {
            phi[(i, i + 3)] = dt;
        }

        let half_dt2 = 0.5 * dt.powi(2);
        for i in 0..3 {
            phi[(i, 6)] = half_dt2 * f_m[i];
            phi[(i + 3, 6)] = dt * f_m[i];
        }

        let j = SMatrix::<f64, 3, 2>::new(0.0, -f_m.y, -f_m.z, f_m.x, f_m.y, 0.0);

        phi.fixed_view_mut::<3, 2>(0, 7).copy_from(&(j * half_dt2));
        phi.fixed_view_mut::<3, 2>(3, 7).copy_from(&(j * dt));

        let mut u = SVector::<f64, 9>::zeros();
        u.fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(self.g * half_dt2));
        u.fixed_view_mut::<3, 1>(3, 0).copy_from(&(self.g * dt));

        let x_pred = phi * self.x + u;
        let p_pred = phi * self.p * phi.transpose() + self.q;

        let y = -x_pred.fixed_rows::<3>(0);
        let s = p_pred.fixed_view::<3, 3>(0, 0) + self.r;
        let s_inv = s.try_inverse().unwrap_or_else(Matrix3::zeros);

        let p_pred_h_t = p_pred.fixed_columns::<3>(0);
        let k = p_pred_h_t * s_inv;

        self.x = x_pred + k * y;
        self.p = p_pred - k * p_pred.fixed_rows::<3>(0);
    }

    pub fn com_velocity(&self) -> Vector3<f64> {
        self.x.fixed_rows::<3>(3).into()
    }
    pub fn mass(&self) -> f64 {
        if self.x[6] == 0.0 {
            f64::INFINITY
        } else {
            1.0 / self.x[6]
        }
    }
}
