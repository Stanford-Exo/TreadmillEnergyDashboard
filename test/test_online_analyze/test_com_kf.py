import unittest
import numpy as np
from online_analyze.com_kf import ComKalmanFilter

class TestComKalmanFilter(unittest.TestCase):

    def test_initialization(self):
        """Verify initialization parameters map cleanly to states."""
        kf = ComKalmanFilter(initial_mass=80.0, pos_std=0.2)
        self.assertAlmostEqual(kf.mass, 80.0)
        np.testing.assert_almost_equal(kf.com_excursion, np.zeros(3))
        np.testing.assert_almost_equal(kf.com_velocity, np.zeros(3))
        self.assertAlmostEqual(kf.com_excursion_covariance[0, 0], 0.2**2)

    def test_tilt_and_mass_convergence(self):
        """Simulate a stationary block on a tilted surface to verify convergence with Y-vertical gravity."""
        true_mass = 75.0
        true_theta_x = 0.02
        true_theta_z = -0.01
        
        # Plate-to-global rotation for vertical Y axis
        R = np.array([
            [1.0, -true_theta_z, 0.0],
            [true_theta_z, 1.0, -true_theta_x],
            [0.0, true_theta_x, 1.0]
        ])
        F_global = np.array([0.0, true_mass * 9.81, 0.0])
        F_measured = R.T @ F_global

        kf = ComKalmanFilter(initial_mass=65.0, mass_std=15.0, tilt_std_rad=0.1)

        dt = 0.01
        for _ in range(800):
            kf.update(F_measured, dt)

        # Confirm convergence within tolerances
        self.assertTrue(abs(kf.mass - true_mass) < 1.5)
        self.assertTrue(abs(kf.tilt_angles[0] - true_theta_x) < 0.005)
        self.assertTrue(abs(kf.tilt_angles[1] - true_theta_z) < 0.005)

        # Confirm reduction in parameter uncertainties
        self.assertTrue(kf.mass_variance < 5.0)
        self.assertTrue(kf.tilt_angles_covariance[0, 0] < 1e-4)

if __name__ == "__main__":
    unittest.main()