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
    global_gyro = q.quaternion(0, *global_gyro)
    gyro_vehicle_frame = quat_state.conjugate() * global_gyro * quat_state
    gyro_x = gyro_vehicle_frame.x/np.sqrt(2) + gyro_vehicle_frame.y/np.sqrt(2) + sigmas[15]
    gyro_y = -gyro_vehicle_frame.x/np.sqrt(2) + gyro_vehicle_frame.y/np.sqrt(2) + sigmas[16]
    gyro_z = gyro_vehicle_frame.z + sigmas[17]

    initial_mag = init_mag.copy()
    initial_mag[2] = -init_mag[2]
    init_mag_q = q.quaternion(0, *initial_mag)
    mag_body_q = (quat_state * (init_mag_q * quat_state).conjugate()).conjugate()
    # convert to numpy array
    mag_measurement = np.array([mag_body_q.x, mag_body_q.y, mag_body_q.z])
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
    next_quat = (delta_q * quat).normalized()
    
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

