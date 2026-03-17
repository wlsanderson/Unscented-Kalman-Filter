from UKF.constants import GRAVITY
import numpy as np
import numpy.typing as npt
import quaternion as q


def measurement_function(sigmas, init_pressure, mag_world, board_to_imu, board_to_mag):
    pressure = init_pressure * np.power(1.0 - (sigmas[2] / 44330.0), 5.255876)
    quat_state = sigmas[-4:]
    quat_state = q.from_float_array(quat_state).normalized()

    # global accel is put into vehicle/board reference frame
    global_acc = q.quaternion(0, *sigmas[6:9])
    acc_vehicle_q = quat_state.conjugate() * global_acc * quat_state
    acc_vehicle = np.array([acc_vehicle_q.x, acc_vehicle_q.y, acc_vehicle_q.z], dtype=np.float32)
    # rotate board frame accel to imu sensor frame
    acc_imu = board_to_imu @ acc_vehicle
    if np.abs(acc_imu[0]) > 19.2882:
        acc_imu[0] = np.clip(acc_imu[0], -19.2882, 19.2882, dtype=np.float32)
    if np.abs(acc_imu[1]) > 19.6925:
        acc_imu[1] = np.clip(acc_imu[1], -19.6925, 19.6925, dtype=np.float32)

    # gyro: body frame rad/s -> deg/s, then rotate to imu sensor frame
    body_gyro_dps = sigmas[9:12] * (180.0 / np.pi)
    gyro_imu = board_to_imu @ body_gyro_dps

    # mag: rotate world mag to vehicle frame, then to mag sensor frame
    mag_world_q = q.quaternion(np.float32(0.0), *mag_world)
    mag_vehicle_q = quat_state.conjugate() * mag_world_q * quat_state
    mag_vehicle = np.array([mag_vehicle_q.x, mag_vehicle_q.y, mag_vehicle_q.z], dtype=np.float32)
    mag_sensor_pred = board_to_mag @ mag_vehicle

    return np.array([
        pressure,
        acc_imu[0],
        acc_imu[1],
        acc_imu[2],
        gyro_imu[0],
        gyro_imu[1],
        gyro_imu[2],
        mag_sensor_pred[0],
        mag_sensor_pred[1],
        mag_sensor_pred[2],
        ], dtype=np.float32)


def state_transition_function(sigmas, dt, state) -> npt.NDArray:
    state_dim = len(sigmas)
    next_state = np.float32(np.zeros(len(sigmas)))
    # quaternions always last 4 states
    quat = q.from_float_array(sigmas[-4:]).normalized()
    # these last states are always predicted as x_k+1 = x_k
    next_state[6:] = sigmas[6:state_dim]
    if state == 1 or state == 2 or state == 3:
        delta_theta = sigmas[9:12] * dt
        
        # update quaternion with small rotation (delta theta -> delta quaternion)
        delta_q = q.from_rotation_vector(delta_theta)
        next_quat = (quat * delta_q)
        next_state[-4:] = q.as_float_array(next_quat)
        accel_grav = sigmas[6:9] * GRAVITY
        accel_grav[2] -= GRAVITY
        next_state[3:6] = sigmas[3:6] + accel_grav * dt
        next_state[0:3] = sigmas[0:3] + sigmas[3:6] * dt
        return next_state
    if state == 4:
        # landed
        grav_vector = np.array([0, 0, GRAVITY])
        next_state[3:6] = sigmas[3:6] + (sigmas[6:9] * GRAVITY - grav_vector) * dt
        next_state[3] = sigmas[3] * 1e-2
        next_state[4] = sigmas[4] * 1e-2
        next_state[0:3] = sigmas[0:3] + sigmas[3:6] * dt
        return next_state
    # state == 0
    next_state[0:6] = np.float32(0)
    next_state[9:12] = next_state[9:12] / np.float32(2)
    

    return next_state

def print_c_array(arr, float_format="{:.8f}"):
    arr = np.asarray(arr)

    # 1D array
    if arr.ndim == 1:
        line = "F, ".join(float_format.format(x) for x in arr)
        print(line + "F")
        return

    # 2D array
    for row in arr:
        line = "F, ".join(float_format.format(x) for x in row)
        print(f"{line}F,")