from UKF.constants import GRAVITY, MIN_VEL_FOR_DRAG
import numpy as np
import numpy.typing as npt
import quaternion as q

def measurement_function(sigmas, init_pressure, X):
    n = len(sigmas)
    quat_state = X[n-4:n]
    quat_state /= np.linalg.norm(quat_state)
    quat_state = q.from_float_array(quat_state)

    pressure = init_pressure * np.power(1 - (sigmas[0] / 44330.0), 5.255876)
    acc_x = sigmas[6]/np.sqrt(2) + sigmas[7]/np.sqrt(2) + sigmas[12]
    acc_y = -sigmas[6]/np.sqrt(2) + sigmas[7]/np.sqrt(2) + sigmas[13]
    acc_z = sigmas[8] + sigmas[14]
    global_acc = q.from_float_array(np.array([0, acc_x, acc_y, acc_z]) / GRAVITY)
    acc_measurement = quat_state.conjugate() * global_acc * quat_state
    gyro_x = sigmas[9]/np.sqrt(2) + sigmas[10]/np.sqrt(2) + sigmas[15]
    gyro_y = -sigmas[9]/np.sqrt(2) + sigmas[10]/np.sqrt(2) + sigmas[16]
    gyro_z = sigmas[11] + sigmas[17]

    m_world = q.quaternion(0, *np.array([1, 0, 0]))
    m_body = quat_state.conjugate() * m_world * quat_state


    return np.array([
        pressure,
        acc_measurement.x,
        acc_measurement.y,
        acc_measurement.z,
        gyro_x,
        gyro_y,
        gyro_z,
        m_body.x,
        m_body.y,
        m_body.z,
        ])

# def state_transition_function(sigmas, dt) -> npt.NDArray:
#     accel = sigmas[6:9]
#     accel_ms = accel * GRAVITY
#     accel_ms[0] -= GRAVITY
#     next_vels = sigmas[3:6] + accel_ms * dt
#     next_positions = sigmas[0:3] + next_vels * dt
#     return np.array([
#         next_positions[0],
#         next_positions[1],
#         next_positions[2],
#         next_vels[0],
#         next_vels[1],
#         next_vels[2],
#         sigmas[6],
#         sigmas[7],
#         sigmas[8],
#         sigmas[9],
#         sigmas[10],
#         sigmas[11],
#     ])


def state_transition_function(sigmas, dt) -> npt.NDArray:
    n = len(sigmas)
    # quaternions always last 4 states
    quat = q.from_float_array(sigmas[n-4:n])
    # get delta theta from angular velocity
    gyro = sigmas[9:12]
    delta_theta = gyro * dt
    # update quaternion with small rotation (delta theta -> delta quaternion)
    delta_q = q.from_rotation_vector(delta_theta)
    next_quat = (quat * delta_q).normalized()
    
    accel = sigmas[6:9]
    accel_grav = accel * GRAVITY
    accel_grav[2] -= GRAVITY
    # calculate next vels without drag first
    next_vels = sigmas[3:6] + accel_grav * dt
    next_positions = sigmas[0:3] - sigmas[3:6] * dt

        

    return np.array([
        next_positions[0],
        next_positions[1],
        next_positions[2],
        next_vels[0],
        next_vels[1],
        next_vels[2],
        accel[0],
        accel[1],
        accel[2],
        gyro[0],
        gyro[1],
        gyro[2],
        sigmas[12],
        sigmas[13],
        sigmas[14],
        sigmas[15],
        sigmas[16],
        sigmas[17],
        next_quat.w,
        next_quat.x,
        next_quat.y,
        next_quat.z,
        ])

