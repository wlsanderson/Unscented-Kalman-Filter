import numpy as np
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_COV
from UKF.ukf import UKF
from UKF.sigma_points import SigmaPoints
from UKF.ukf_functions import state_transition_function, print_c_array

def process_covariance_function(dt):
        """
        Process noise covariance matrix
        """
        qvar = np.float32(np.ones((15, 15))) * np.float32(1.0)
        return qvar * dt

sigma_points = SigmaPoints(
            # n is the dimension of the state, minus one due to the quaternion representation
            n = STATE_DIM - 1,
            alpha = ALPHA,
            beta = BETA,
            kappa = KAPPA,
        )
ukf = UKF(
            dim_x = STATE_DIM,
            dim_z = MEASUREMENT_DIM,
            points = sigma_points,
        )
ukf.P = np.float32(INITIAL_STATE_COV.copy())
ukf.X = np.array([0.0, 0.0, 4.0, 0, 1.0, 5.5, 1.0, 2.5, -2.5, 6.4, 0.2, -0.3, 0.4, -0.8, -0.1, 0.5], dtype=np.float32)
ukf.F = state_transition_function
ukf.Q = process_covariance_function
ukf.predict(np.float32(0.05), 3)
print_c_array(ukf.X)
print()
ukf.predict(0.05, 3)
print_c_array(ukf.X)
print()
ukf.predict(0.05, 3)
print_c_array(ukf.X)
print()
print_c_array(ukf.P)




