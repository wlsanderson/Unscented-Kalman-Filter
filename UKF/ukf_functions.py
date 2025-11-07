from UKF.constants import GRAVITY, MIN_VEL_FOR_DRAG
import numpy as np
import numpy.typing as npt
import quaternion as q

def measurement_function(sigmas, init_pressure, init_mag, X):
    pressure = init_pressure * np.power(1 - (sigmas[2] / 44330.0), 5.255876)
    quat_state = X[-4:]
    quat_state /= np.linalg.norm(quat_state)
    quat_state = q.from_float_array(quat_state)

    global_acc = q.quaternion(0, *sigmas[6:9])
    # globla accel is put into vehicle reference frame
    acc_vehicle_frame = quat_state.conjugate() * global_acc * quat_state
    # rotates vehicle frame accel 45 degrees ccw to line up with how imu is mounted on board
    acc_x = acc_vehicle_frame.x/np.sqrt(2) + acc_vehicle_frame.y/np.sqrt(2) + sigmas[12]
    acc_y = -acc_vehicle_frame.x/np.sqrt(2) + acc_vehicle_frame.y/np.sqrt(2) + sigmas[13]
    acc_z = acc_vehicle_frame.z + sigmas[14]

    # same process with gyro: rotate to vehicle frame, then rotate 45 degrees
    global_gyro = sigmas[9:12] * (180.0 / np.pi)
    global_gyro = q.from_rotation_vector(global_gyro)
    gyro_x = global_gyro.x/np.sqrt(2) + global_gyro.y/np.sqrt(2) + sigmas[15]
    gyro_y = -global_gyro.x/np.sqrt(2) + global_gyro.y/np.sqrt(2) + sigmas[16]
    gyro_z = global_gyro.z + sigmas[17]

    initial_mag = init_mag.copy()
    initial_mag[2] = -init_mag[2]
    init_mag_q = q.quaternion(0, *initial_mag)
    mag_body_q = (quat_state * (init_mag_q * quat_state).conjugate()).conjugate()
    # convert to numpy array
    mag_measurement = np.array([mag_body_q.x, mag_body_q.y, -mag_body_q.z])
    return np.array([
        pressure,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        mag_measurement[0],
        mag_measurement[1],
        mag_measurement[2],
        ])


def state_transition_function(sigmas, dt, u) -> npt.NDArray:
    next_state = np.zeros(len(sigmas))
    # quaternions always last 4 states
    quat = q.from_float_array(sigmas[-4:]).normalized()
    # these last states are always predicted as x_k+1 = x_k
    next_state[6:22] = sigmas[6:22]
    if u[0] is None:
        delta_theta = sigmas[9:12] * dt
        # update quaternion with small rotation (delta theta -> delta quaternion)
        delta_q = q.from_rotation_vector(delta_theta)
        next_quat = (quat * delta_q).normalized()
        next_state[18:22] = q.as_float_array(next_quat)
        accel_grav = sigmas[6:9] * GRAVITY
        accel_grav[2] -= GRAVITY
        next_state[3:6] = sigmas[3:6] + accel_grav * dt
        next_state[0:3] = sigmas[0:3] + sigmas[3:6] * dt
        return next_state
    next_state[0:6] = u[0:6]
    next_state[6:9] = next_state[6:9] / np.linalg.norm(next_state[6:9])
    next_state[9:12] = u[6:9]
    return next_state

