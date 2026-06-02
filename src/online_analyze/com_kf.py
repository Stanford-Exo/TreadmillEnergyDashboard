import numpy as np


class ComKalmanFilter:
    """
    A highly optimized 9D Kalman Filter to estimate Center of Mass (COM) excursion, velocity,
    inverse mass, and 2D force plate tilt angles (pitch and roll) with gravity oriented on the Y-axis.
    """

    def __init__(
        self,
        initial_mass=70.0,
        pos_std=0.1,
        vel_std=0.1,
        mass_std=10.0,
        tilt_std_rad=0.003,
        meas_std=0.005,
        proc_noise_pos=1e-5,
        proc_noise_vel=1e-4,
        proc_noise_w=1e-10,
        proc_noise_phi=1e-11,
    ):

        self.x = np.zeros(9)
        self.x[6] = 1.0 / initial_mass  # w = 1/m

        self.P = np.zeros((9, 9))
        self.P[0:3, 0:3] = np.eye(3) * (pos_std**2)
        self.P[3:6, 3:6] = np.eye(3) * (vel_std**2)

        w_var = (1.0 / (initial_mass**2)) ** 2 * (mass_std**2)
        self.P[6, 6] = w_var

        phi_var = (self.x[6] * tilt_std_rad) ** 2
        self.P[7:9, 7:9] = np.eye(2) * phi_var

        self.Q = np.zeros((9, 9))
        self.Q[0:3, 0:3] = np.eye(3) * proc_noise_pos
        self.Q[3:6, 3:6] = np.eye(3) * proc_noise_vel
        self.Q[6, 6] = proc_noise_w
        self.Q[7:9, 7:9] = np.eye(2) * proc_noise_phi

        self.R = np.eye(3) * (meas_std**2)
        self.g = np.array([0.0, -9.81, 0.0])

        # --- Pre-allocated helper matrices (Zero-allocation runtime) ---
        self._I3 = np.eye(3)
        self._u_temp = np.zeros(9)
        self._S_inv_buf = np.empty((3, 3))

        # Keep a single persistent Phi matrix to avoid 9.78M copy calls
        self.Phi = np.eye(9)

    def _get_tilt_jacobian(self, F_m):
        F_mx, F_my, F_mz = F_m
        return np.array([[0.0, -F_my], [-F_mz, F_mx], [F_my, 0.0]])

    def _inv3x3(self, A):
        """Analytical 3x3 matrix inversion optimized with flat unpacking and a static write buffer."""
        # Unpack all 9 elements into local variables to avoid slow 2D subscription overhead
        a, b, c, d, e, f, g, h, i = A.flat

        # Compute cofactor determinants
        co_00 = e * i - f * h
        co_10 = f * g - d * i
        co_20 = d * h - e * g

        det = a * co_00 + b * co_10 + c * co_20
        if abs(det) < 1e-15:
            return np.linalg.inv(A)

        invdet = 1.0 / det
        invA = self._S_inv_buf

        # Write directly to the persistent buffer
        invA[0, 0] = co_00 * invdet
        invA[0, 1] = (c * h - b * i) * invdet
        invA[0, 2] = (b * f - c * e) * invdet
        invA[1, 0] = co_10 * invdet
        invA[1, 1] = (a * i - c * g) * invdet
        invA[1, 2] = (c * d - a * f) * invdet
        invA[2, 0] = co_20 * invdet
        invA[2, 1] = (b * g - a * h) * invdet
        invA[2, 2] = (a * e - b * d) * invdet
        return invA

    def update(self, F_m, dt):
        F_m = np.asarray(F_m, dtype=float)

        # Use the persistent self.Phi matrix (removes 9.78 million allocations)
        Phi = self.Phi

        dt_I3 = self._I3 * dt
        Phi[0:3, 3:6] = dt_I3

        half_dt2 = 0.5 * (dt**2)
        Phi[0:3, 6] = half_dt2 * F_m
        Phi[3:6, 6] = dt * F_m

        J = self._get_tilt_jacobian(F_m)
        Phi[0:3, 7:9] = half_dt2 * J
        Phi[3:6, 7:9] = dt * J

        u = self._u_temp
        u[0:3] = 0.5 * self.g * (dt**2)
        u[3:6] = self.g * dt

        # State and covariance propagation
        x_pred = Phi @ self.x + u
        P_pred = Phi @ self.P @ Phi.T + self.Q

        # Reset dynamic components of Phi back to 0.0 in-place for the next frame
        # (This avoids memory allocation and is faster than copying)
        Phi[0:3, 3:6] = 0.0
        Phi[0:3, 6] = 0.0
        Phi[3:6, 6] = 0.0
        Phi[0:3, 7:9] = 0.0
        Phi[3:6, 7:9] = 0.0

        # Measurement update
        y = -x_pred[0:3]
        S = P_pred[0:3, 0:3] + self.R
        S_inv = self._inv3x3(S)

        P_pred_H_T = P_pred[:, 0:3]
        K = P_pred_H_T @ S_inv

        self.x = x_pred + K @ y
        self.P = P_pred - K @ P_pred[0:3, :]

    @property
    def com_excursion(self):
        return self.x[0:3]

    @property
    def com_velocity(self):
        return self.x[3:6]

    @property
    def mass(self):
        w = self.x[6]
        return 1.0 / w if w != 0 else float("inf")

    @property
    def tilt_angles(self):
        w = self.x[6]
        if w == 0:
            return np.zeros(2)
        phi_x, phi_z = self.x[7], self.x[8]
        return np.array([phi_x / w, phi_z / w])

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
            return float("inf")
        var_w = self.P[6, 6]
        return (1.0 / (w**4)) * var_w

    @property
    def tilt_angles_covariance(self):
        w = self.x[6]
        if w <= 0:
            return np.eye(2) * float("inf")

        phi_x, phi_z = self.x[7], self.x[8]
        P_sub = self.P[6:9, 6:9]

        J = np.array([[-phi_x / (w**2), 1.0 / w, 0.0], [-phi_z / (w**2), 0.0, 1.0 / w]])

        return J @ P_sub @ J.T
