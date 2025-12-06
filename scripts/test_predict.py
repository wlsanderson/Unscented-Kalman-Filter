import numpy as np
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_COV
from UKF.ukf import UKF
from UKF.sigma_points import SigmaPoints
from UKF.ukf_functions import state_transition_function

def process_covariance_function(dt):
        """
        Process noise covariance matrix
        """
        qvar = np.ones((21, 21)) * 1
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
ukf.P = INITIAL_STATE_COV.copy()
ukf.X = np.array([0, 0, 4.0, 0, 1.0, 0, 30.0, 2.0, 2.5, 6.4, 0.2, -0.3, -0.6, 0, 0, 0, 0, 0, 4.0, -2.0, -0.1, 0.5])
ukf.F = state_transition_function
ukf.Q = process_covariance_function
ukf.predict(0.1, 3)
ukf.predict(0.1, 3)
ukf.predict(0.1, 3)
ukf.predict(0.1, 3)
ukf.predict(0.1, 3)
list_arr = ukf.P.tolist()

# Join the elements of the list with a comma and space
comma_separated_string = ", ".join(map(str, list_arr))
print(comma_separated_string)



