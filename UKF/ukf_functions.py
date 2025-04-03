from UKF.constants import GRAVITY, DRAG_COEFFICIENT, ROCKET_MASS, REFERENCE_AREA, AIR_DENSITY
import numpy as np
import numpy.typing as npt

def measurement_function(sigmas, init_alt):
    acc = sigmas[2]
    acc_measurement = -acc / GRAVITY
    alt_measurement = sigmas[0] + init_alt
    return np.array([alt_measurement, acc_measurement])

def base_state_transition(sigmas, dt, drag_option: bool = False) -> npt.NDArray:
    next_acc = sigmas[2]
    next_vel = sigmas[1] + (next_acc - 9.81) * dt
    if drag_option:
        next_acc = next_acc - dt * calc_drag(next_vel) / ROCKET_MASS
        next_vel = sigmas[1] + (next_acc - 9.81) * dt
    next_alt = sigmas[0] + (next_vel * dt) + 0.5 * (next_acc * dt**2)
    return np.array([next_alt, next_vel, next_acc])


def calc_drag(velocity):
    return 0.5 * DRAG_COEFFICIENT * REFERENCE_AREA * AIR_DENSITY * velocity**2