from UKF.sigma_points import SigmaPoints
from UKF.ukf import UKF
from UKF.state import State, StandbyState
from pathlib import Path
from UKF.data_processor import DataProcessor
import timeit
import numpy as np
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_ESTIMATE, INITIAL_STATE_COV, ROCKET_MASS, DRAG_COEFFICIENT, AIR_DENSITY,REFERENCE_AREA, GRAVITY


sigma_points = SigmaPoints(
    n = STATE_DIM,
    alpha = ALPHA,
    beta = BETA,
    kappa = KAPPA,
)
ukf = UKF(
    dim_x = STATE_DIM,
    dim_z = MEASUREMENT_DIM,
    points = sigma_points,
)
def process_covariance_function(dt):
    """
    Process noise covariance matrix
    """
    qvar = 1
    q_covariance_matrix = np.zeros([STATE_DIM, STATE_DIM])
    q_covariance_matrix[-1][-1] = qvar
    return q_covariance_matrix

def base_state_transition(sigmas, dt, drag_option: bool = False, *F_args):
    next_acc = sigmas[2]
    next_vel = sigmas[1] + (next_acc - 9.81) * dt
    if drag_option:
        next_acc = next_acc - dt * calc_drag(next_vel) / ROCKET_MASS
        next_vel = sigmas[1] + (next_acc - 9.81) * dt
    next_alt = sigmas[0] + (next_vel * dt) + 0.5 * (next_acc * dt**2)
    return np.array([next_alt, next_vel, next_acc])


def calc_drag(velocity):
    return 0.5 * DRAG_COEFFICIENT * REFERENCE_AREA * AIR_DENSITY * velocity**2

def measurement_function(sigmas, **H_args):
    init_alt = H_args["H_args"]
    acc = sigmas[2]
    acc_measurement = -acc / GRAVITY
    alt_measurement = sigmas[0] + init_alt
    return np.array([alt_measurement, acc_measurement])

ukf.X = [1,1,1]
ukf.H = measurement_function
ukf.Q = process_covariance_function
ukf.F = base_state_transition
ukf.P = np.diag([1,1,1])
ukf.R = np.diag([1,1])

dp = DataProcessor(Path("launch_data/pelicanator_launch_2.csv"))

t= timeit.timeit(INITIAL_STATE_COV.copy, number=1000000)
print(t)