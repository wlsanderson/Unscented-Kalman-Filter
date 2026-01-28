import numpy as np
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_COV
from UKF.ukf import UKF
from UKF.sigma_points import SigmaPoints
from UKF.ukf_functions import state_transition_function, measurement_function, print_c_array


sigmas = np.array([
        2.939400000e+01,  2.45960000e+00,  5.46970000e+00,  1.46970000e+02,
        1.079801570e+01,  7.34850786e+00,  3.00000000e+01,  2.00001604e+00,
        2.500008020e+00,  6.40000545e+00,  2.00005452e-01, -2.99994548e-01,
        5.05660013e-01,  8.43914113e-01, 1.499743390e-01, -9.81051397e-02
], dtype=np.float32)

#print_c_array(state_transition_function(sigmas, 0.1, 3))

init_p = 101328
mag_world = np.array([0.3, 0.9, -0.2])
mag_world /= np.linalg.norm(mag_world)

print_c_array(measurement_function(sigmas, init_p, mag_world))