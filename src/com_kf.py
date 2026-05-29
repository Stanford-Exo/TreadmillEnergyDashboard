import numpy as np

class ComKalmanFilter:
    """
    A 9D Kalman Filter to estimate Center of Mass (COM) excursion, velocity,
    inverse mass, and 2D force plate tilt angles (pitch and roll).
    """
    def __init__(self, 
                 initial_mass=70.0, 
                 pos_std=0.1, 
                 vel_std=0.1, 
                 mass_std=10.0, 
                 tilt_std_rad=0.003,
                 meas_std=0.005,
                 proc_noise_pos=1e-5,
                 proc_noise_vel=1e-4,
                 proc_noise_w=1e-10,
                 proc_noise_phi=1e-11):
        
        self.x = np.zeros(9)
        self.x[6] = 1.0 / initial_mass  # w = 1/m

        self.P = np.zeros((9, 9))
        self.P[0:3, 0:3] = np.eye(3) * (pos_std ** 2)
        self.P[3:6, 3:6] = np.eye(3) * (vel_std ** 2)
        
        w_var = (1.0 / (initial_mass ** 2)) ** 2 * (mass_std ** 2)
        self.P[6, 6] = w_var
        
        phi_var = (self.x[6] * tilt_std_rad) ** 2
        self.P[7:9, 7:9] = np.eye(2) * phi_var

        self.Q = np.zeros((9, 9))
        self.Q[0:3, 0:3] = np.eye(3) * proc_noise_pos
        self.Q[3:6, 3:6] = np.eye(3) * proc_noise_vel
        self.Q[6, 6] = proc_noise_w
        self.Q[7:9, 7:9] = np.eye(2) * proc_noise_phi

        self.R = np.eye(3) * (meas_std ** 2)
        self.g = np.array([0.0, -9.81, 0.0])

    def _get_tilt_jacobian(self, F_m):
        F_mx, F_my, F_mz = F_m
        return np.array([
            [0.0, F_mz],
            [-F_mz, 0.0],
            [F_my, -F_mx]
        ])

    def update(self, F_m, dt):
        F_m = np.asarray(F_m, dtype=float)
        
        Phi = np.eye(9)
        Phi[0:3, 3:6] = np.eye(3) * dt
        Phi[0:3, 6] = 0.5 * (dt ** 2) * F_m
        Phi[3:6, 6] = dt * F_m
        
        J = self._get_tilt_jacobian(F_m)
        Phi[0:3, 7:9] = 0.5 * (dt ** 2) * J
        Phi[3:6, 7:9] = dt * J
        
        u = np.zeros(9)
        u[0:3] = 0.5 * self.g * (dt ** 2)
        u[3:6] = self.g * dt

        x_pred = Phi @ self.x + u
        P_pred = Phi @ self.P @ Phi.T + self.Q

        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        
        z = np.zeros(3)
        y = z - H @ x_pred
        
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.P = (np.eye(9) - K @ H) @ P_pred

    @property
    def com_excursion(self):
        return self.x[0:3].copy()

    @property
    def com_velocity(self):
        return self.x[3:6].copy()

    @property
    def mass(self):
        w = self.x[6]
        return 1.0 / w if w != 0 else float('inf')

    @property
    def tilt_angles(self):
        w = self.x[6]
        if w == 0:
            return np.zeros(2)
        phi_x, phi_y = self.x[7], self.x[8]
        return np.array([phi_x / w, phi_y / w])

    @property
    def com_excursion_covariance(self):
        return self.P[0:3, 0:3].copy()

    @property
    def com_velocity_covariance(self):
        return self.P[3:6, 3:6].copy()

    @property
    def mass_variance(self):
        w = self.x[6]
        if w <= 0:
            return float('inf')
        var_w = self.P[6, 6]
        return (1.0 / (w ** 4)) * var_w

    @property
    def tilt_angles_covariance(self):
        w = self.x[6]
        if w <= 0:
            return np.eye(2) * float('inf')
        
        phi_x, phi_y = self.x[7], self.x[8]
        P_sub = self.P[6:9, 6:9]
        
        J = np.array([
            [-phi_x / (w ** 2), 1.0 / w, 0.0],
            [-phi_y / (w ** 2), 0.0, 1.0 / w]
        ])
        
        return J @ P_sub @ J.T