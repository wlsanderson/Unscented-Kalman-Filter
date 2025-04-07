from UKF.constants import GRAVITY, DRAG_COEFFICIENT, ROCKET_MASS, REFERENCE_AREA, AIR_DENSITY
import numpy as np
import numpy.typing as npt
from UKF.quaternion import eul2quat, quat_multiply

def measurement_function(sigmas, **H_args):
    init_alt = H_args["H_args"]
    acc = sigmas[2]
    acc_measurement = -acc / GRAVITY
    alt_measurement = sigmas[0] + init_alt
    return np.array([alt_measurement, acc_measurement])

def base_state_transition(sigmas, dt, drag_option: bool = False, *F_args) -> npt.NDArray:
    next_acc = sigmas[2]
    next_vel = sigmas[1] + (next_acc - 9.81) * dt
    if drag_option:
        next_acc = next_acc - dt * calc_drag(next_vel) / ROCKET_MASS
        next_vel = sigmas[1] + (next_acc - 9.81) * dt
    next_alt = sigmas[0] + (next_vel * dt) + 0.5 * (next_acc * dt**2)

    gyro_x = sigmas[7]
    gyro_y = sigmas[8]
    gyro_z = sigmas[9]

    delta_quat = eul2quat(sigmas[7:10], dt)
    quat = quat_multiply(sigmas[2:6], delta_quat)
    return np.array([
        next_alt,
        next_vel, 
        next_acc,
        quat[0],
        quat[1],
        quat[2],
        quat[3],
        gyro_x,
        gyro_y,
        gyro_z])


def calc_drag(velocity):
    return 0.5 * DRAG_COEFFICIENT * REFERENCE_AREA * AIR_DENSITY * velocity**2