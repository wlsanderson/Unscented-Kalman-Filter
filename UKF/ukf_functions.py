from UKF.constants import GRAVITY, DRAG_COEFFICIENT, ROCKET_MASS, REFERENCE_AREA, AIR_DENSITY
import numpy as np
import numpy.typing as npt
from UKF.quaternion import rotvec2quat, quat_multiply, quat_rotate

def measurement_function(sigmas, **H_args):
    init_alt = H_args["H_args"]
    global_acc = np.array([0, 0, -sigmas[2] / GRAVITY])
    acc = quat_rotate(sigmas[6:10], global_acc)
    alt = sigmas[0] + init_alt
    gyro = sigmas[3:6]
    return np.array([alt, acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]])

def base_state_transition(sigmas, dt, drag_option: bool = False, *F_args) -> npt.NDArray:
    next_acc = sigmas[2]
    next_vel = sigmas[1] + (next_acc - 9.81) * dt
    if drag_option:
        next_acc = next_acc - dt * calc_drag(next_vel) / ROCKET_MASS
        next_vel = sigmas[1] + (next_acc - 9.81) * dt
    next_alt = sigmas[0] + (next_vel * dt) + 0.5 * (next_acc * dt**2)

    gyro_x = sigmas[3]
    gyro_y = sigmas[4]
    gyro_z = sigmas[5]

    delta_quat = rotvec2quat(sigmas[3:6], dt)
    quat = quat_multiply(sigmas[6:10], delta_quat)
    return np.array([
        next_alt,
        next_vel, 
        next_acc,
        gyro_x,
        gyro_y,
        gyro_z,
        quat[0],
        quat[1],
        quat[2],
        quat[3]
        ])

def calc_drag(velocity):
    return 0.5 * DRAG_COEFFICIENT * REFERENCE_AREA * AIR_DENSITY * velocity**2