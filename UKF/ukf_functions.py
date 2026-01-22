from UKF.constants import GRAVITY, MIN_VEL_FOR_DRAG, DRAG_PARAM
import numpy as np
import numpy.typing as npt
import quaternion as q

def measurement_function(sigmas, init_pressure, mag_world):
    pressure = init_pressure * np.power(1.0 - (sigmas[2] / 44330.0), 5.255876)
    quat_state = sigmas[-4:]
    quat_state = q.from_float_array(quat_state).normalized()

    
    global_acc = q.quaternion(0, *sigmas[6:9])
    # global accel is put into vehicle reference frame
    acc_vehicle_frame = quat_state.conjugate() * global_acc * quat_state
    # rotates vehicle frame accel 45 degrees ccw to line up with how imu is mounted on board
    acc_x = acc_vehicle_frame.x/np.sqrt(2) + acc_vehicle_frame.y/np.sqrt(2)
    acc_y = -acc_vehicle_frame.x/np.sqrt(2) + acc_vehicle_frame.y/np.sqrt(2)
    acc_z = acc_vehicle_frame.z
    if np.abs(acc_x) > 19.2882:
        acc_x = np.clip(acc_x, -19.2882, 19.2882)
    if np.abs(acc_y) > 19.6925:
        acc_y = np.clip(acc_y, -19.6925, 19.6925)

    # same process with gyro: rotate to vehicle frame, then rotate 45 degrees
    global_gyro = sigmas[9:12] * (180.0 / np.pi)
    gyro_x = global_gyro[0]/np.sqrt(2) + global_gyro[1]/np.sqrt(2)
    gyro_y = -global_gyro[0]/np.sqrt(2) + global_gyro[1]/np.sqrt(2)
    gyro_z = global_gyro[2]

    R_mag_to_vehicle = np.diag([1.0, 1.0, -1.0])
    R_vehicle_to_mag = R_mag_to_vehicle.T
    mag_world_q = q.quaternion(0.0, *mag_world)

    # rotate mag_world into VEHICLE frame:
    mag_vehicle_q = quat_state.conjugate() * mag_world_q * quat_state
    mag_vehicle = np.array([mag_vehicle_q.x, mag_vehicle_q.y, mag_vehicle_q.z], dtype=float)

    # convert VEHICLE-frame mag into sensor mag frame using vehicle->mag_sensor (transpose of R_mag_to_vehicle)
    mag_sensor_pred = R_vehicle_to_mag @ mag_vehicle
    return np.array([
        pressure,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        mag_sensor_pred[0],
        mag_sensor_pred[1],
        mag_sensor_pred[2],
        ])


def state_transition_function(sigmas, dt, state) -> npt.NDArray:
    next_state = np.zeros(len(sigmas))
    # quaternions always last 4 states
    quat = q.from_float_array(sigmas[-4:]).normalized()
    # these last states are always predicted as x_k+1 = x_k
    next_state[6:16] = sigmas[6:16]
    if state == 1 or state == 2 or state == 3:
        delta_theta = sigmas[9:12] * dt
        # update quaternion with small rotation (delta theta -> delta quaternion)
        delta_q = q.from_rotation_vector(delta_theta)
        next_quat = (quat * delta_q).normalized()
        next_state[12:16] = q.as_float_array(next_quat)
        accel_grav = sigmas[6:9] * GRAVITY
        accel_grav[2] -= GRAVITY
        next_state[3:6] = sigmas[3:6] + accel_grav * dt
        
        # update velocity and acceleration states to use drag if the velocity is high enough
        if next_state[5] > MIN_VEL_FOR_DRAG:
            # calculate expected drag accel
            drag_acc = 0.5 * DRAG_PARAM * next_state[5]**2
            next_state[5] += drag_acc * dt
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
    next_state[0:6] = 0
    next_state[9:12] = next_state[9:12] / 2
    

    return next_state

def print_c_array(arr, float_format="{:.15f}"):
    arr = np.asarray(arr)

    # 1D array
    if arr.ndim == 1:
        line = ", ".join(float_format.format(x) for x in arr)
        print(line)
        return

    # 2D array
    for row in arr:
        line = ", ".join(float_format.format(x) for x in row)
        print(f"{line},")