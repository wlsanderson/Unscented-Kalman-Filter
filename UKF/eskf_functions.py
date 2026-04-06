"""ESKF dynamics, measurement model, and Jacobians (vertical-only state)."""

import numpy as np
import quaternion as q

from UKF.constants import GRAVITY

_SQRT2_INV = 1.0 / np.sqrt(2.0)

# ---- v2 hardware (current PCB) ----
R_IMU_TO_BOARD_V2 = np.array([
    [_SQRT2_INV, -_SQRT2_INV, 0.0],
    [_SQRT2_INV,  _SQRT2_INV, 0.0],
    [0.0,         0.0,        1.0],
], dtype=np.float32)

R_MAG_TO_BOARD_V2 = np.array([
    [ 0.0,  1.0,  0.0],
    [ -1.0, 0.0,  0.0],
    [ 0.0,  0.0,  -1.0],
], dtype=np.float32)

# ---- v1 hardware (legacy PCB) ----
R_IMU_TO_BOARD_V1 = np.array([
    [ _SQRT2_INV, _SQRT2_INV, 0.0],
    [-_SQRT2_INV, _SQRT2_INV, 0.0],
    [ 0.0,        0.0,        1.0],
], dtype=np.float32)

R_MAG_TO_BOARD_V1 = np.array([
    [ 0.0,  1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0],
], dtype=np.float32)

# Backwards-compatible aliases (default to v2 since current hardware is v2)
R_IMU_TO_BOARD = R_IMU_TO_BOARD_V2
R_MAG_TO_BOARD = R_MAG_TO_BOARD_V2


def skew(v):
    """3x3 skew-symmetric matrix of vector v."""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ], dtype=np.float32)


def _imu_to_board(accel_sensor, gyro_sensor, R_imu=None):
    """Rotate IMU sensor-frame data into board frame and convert units."""
    if R_imu is None:
        R_imu = R_IMU_TO_BOARD
    accel_board = R_imu @ accel_sensor * GRAVITY  # g → m/s²
    gyro_board = R_imu @ np.deg2rad(gyro_sensor)  # deg/s → rad/s
    return accel_board, gyro_board


# =====================================================================
# Nominal predict
# =====================================================================

def nominal_predict(x_nom, u, dt, R_imu=None):
    """Propagate nominal state by dt using board-frame IMU inputs."""
    x = x_nom.copy()

    a_board_ms2, w_board_rads = _imu_to_board(u[0:3], u[3:6], R_imu)

    # quaternion: board -> world
    quat = q.from_float_array(x[2:6]).normalized()
    # world-frame acceleration
    a_board_ms2_q = q.from_float_array([0.0, a_board_ms2[0], a_board_ms2[1], a_board_ms2[2]])
    a_world = quat * a_board_ms2_q * quat.conjugate()
    a_world = np.float32(q.as_float_array(a_world)[1:4])
    a_world[2] -= GRAVITY

    # integrate position and velocity
    x[0] += x[1] * dt
    x[1] += a_world[2] * dt

    # quaternion integration
    delta_theta = w_board_rads * dt
    delta_q = q.from_rotation_vector(delta_theta)
    new_quat = quat * delta_q
    x[2:6] = q.as_float_array(new_quat.normalized())

    return x


# =====================================================================
# Error-state Jacobian
# =====================================================================

def error_state_jacobian(x_nom, u, dt, R_imu=None):
    """Discrete error-state Jacobian F_d (5x5). Only used as reference
    to show math, below is error_State_jacobian_opt which is the unrolled
    version of this function, for use in actual implementation."""
    a_board_ms2, w_board_rads = _imu_to_board(u[0:3], u[3:6], R_imu)

    quat = q.from_float_array(x_nom[2:6]).normalized()
    R_b2w = q.as_rotation_matrix(quat)

    F = np.eye(5, dtype=np.float32)

    # δẑ += δv_z * dt
    F[0, 1] += dt

    # δv̇_z = -(R @ skew(a_board))_z * δθ * dt
    F[1, 2:5] = (-R_b2w @ skew(a_board_ms2))[2, :] * dt

    # δθ̇ = -skew(ω) @ δθ * dt
    F[2:5, 2:5] += -skew(w_board_rads) * dt
    return F

def error_state_jacobian_opt(x_nom, u, dt, R_imu=None):
    a_board_ms2, w_board_rads = _imu_to_board(u[0:3], u[3:6], R_imu)

    # Assuming x_nom[2:6] is [qw, qx, qy, qz]. 
    # If it is [qx, qy, qz, qw], simply map the variables accordingly.
    quat = q.from_float_array(x_nom[2:6]).normalized()
    
    ax, ay, az = a_board_ms2
    wx, wy, wz = w_board_rads
    F = np.eye(5, dtype=np.float32)

    # δẑ += δv_z * dt
    F[0, 1] = dt

    # 1. Directly compute the 3rd row of the rotation matrix from the quaternion
    r31 = 2.0 * (quat.x * quat.z - quat.w * quat.y)
    r32 = 2.0 * (quat.y * quat.z + quat.w * quat.x)
    r33 = 1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)

    # 2. δv̇_z = (a_board x r_3) * dt
    F[1, 2] = (ay * r33 - az * r32) * dt
    F[1, 3] = (az * r31 - ax * r33) * dt
    F[1, 4] = (ax * r32 - ay * r31) * dt

    # δθ̇ = -skew(ω) @ δθ * dt (Unrolled)
    F[2, 3] =  wz * dt
    F[2, 4] = -wy * dt
    
    F[3, 2] = -wz * dt
    F[3, 4] =  wx * dt
    
    F[4, 2] =  wy * dt
    F[4, 3] = -wx * dt
    return F


# =====================================================================
# Measurement model
# =====================================================================

def measurement_function(x_nom, init_alt, mag_world, R_mag=None):
    """Predicted measurement: pressure + mag (sensor frame)."""
    if R_mag is None:
        R_mag = R_MAG_TO_BOARD
    altitude = x_nom[0]
    quat = q.from_float_array(x_nom[2:6])

    # barometric pressure from altitude
    pressure = np.float32(101325.0) * np.power(1.0 - ((altitude + init_alt) / 44330.8), 5.255883)

    # magnetometer: rotate world mag into board frame, then into mag sensor frame
    mag_world_q = q.from_float_array([0, mag_world[0], mag_world[1], mag_world[2]])
    mag_board_q = quat.conjugate() * mag_world_q * quat
    mag_board = np.array([mag_board_q.x, mag_board_q.y, mag_board_q.z], dtype=np.float32)
    mag_sensor = R_mag @ mag_board

    return np.array([pressure, mag_sensor[0], mag_sensor[1], mag_sensor[2]], dtype=np.float32)


def measurement_function_pressure_only(x_nom, init_alt):
    """Pressure-only predicted measurement (1-dim)."""
    altitude = x_nom[0]
    pressure = np.float32(101325.0) * np.power(1.0 - ((altitude + init_alt) / 44330.8), 5.255883)
    return np.array([pressure], dtype=np.float32)


def measurement_jacobian(x_nom, init_alt, mag_world, R_mag):
    """Measurement Jacobian H (4x5). Only used as reference
    to show math, below is measurement_jacobian_opt which is the unrolled
    version of this function, for use in actual implementation."""
    altitude = x_nom[0]
    quat = q.from_float_array(x_nom[2:6]).normalized()

    H = np.zeros((4, 5), dtype=np.float32)

    # ∂pressure/∂altitude
    base = 1.0 - (altitude + init_alt) / 44330.8
    if base > 0:
        dp_dalt = 101325.0 * 5.255883 * np.power(base, 4.255883) * (-1.0 / 44330.8)
    else:
        dp_dalt = 0.0
    H[0, 0] = dp_dalt

    # ∂mag_sensor/∂δθ
    R_b2w = q.as_rotation_matrix(quat)
    mag_board = R_b2w.T @ mag_world
    H[1:4, 2:5] = R_mag @ skew(mag_board)
    return H

def measurement_jacobian_opt(x_nom, init_alt, mag_world, R_mag):
    """Discrete Measurement Jacobian H (4x5)."""
    altitude = x_nom[0]
    
    # Assuming x_nom[2:6] is [qw, qx, qy, qz]
    qw, qx, qy, qz = x_nom[2:6]

    wx, wy, wz = mag_world
    
    H = np.zeros((4, 5), dtype=np.float32)

    # 1. ∂pressure/∂altitude (Simplified constants)
    base = 1.0 - (altitude + init_alt) / 44330.8
    if base > 0:
        # Pre-calculated 101325 * 5.255883 * (-1.0 / 44330.8) = -12.01314537466
        H[0, 0] = -12.01314537466 * np.power(base, 4.255883)
    else:
        H[0, 0] = 0.0

    # 2. Compute mag_board = R_b2w.T @ mag_world directly from quat
    # These are the dot products of mag_world with the columns of R_b2w
    mx = (1.0 - 2.0*(qy*qy + qz*qz))*wx + 2.0*(qx*qy + qw*qz)*wy + 2.0*(qx*qz - qw*qy)*wz
    my = 2.0*(qx*qy - qw*qz)*wx + (1.0 - 2.0*(qx*qx + qz*qz))*wy + 2.0*(qy*qz + qw*qx)*wz
    mz = 2.0*(qx*qz + qw*qy)*wx + 2.0*(qy*qz - qw*qx)*wy + (1.0 - 2.0*(qx*qx + qy*qy))*wz

    # 3. ∂mag_sensor/∂δθ = R_mag @ skew(mag_board) 
    # Unrolled as the cross product of R_mag rows and mag_board
    # R_mag row 0
    H[1, 2] = R_mag[0, 1] * mz - R_mag[0, 2] * my
    H[1, 3] = R_mag[0, 2] * mx - R_mag[0, 0] * mz
    H[1, 4] = R_mag[0, 0] * my - R_mag[0, 1] * mx

    # R_mag row 1
    H[2, 2] = R_mag[1, 1] * mz - R_mag[1, 2] * my
    H[2, 3] = R_mag[1, 2] * mx - R_mag[1, 0] * mz
    H[2, 4] = R_mag[1, 0] * my - R_mag[1, 1] * mx

    # R_mag row 2
    H[3, 2] = R_mag[2, 1] * mz - R_mag[2, 2] * my
    H[3, 3] = R_mag[2, 2] * mx - R_mag[2, 0] * mz
    H[3, 4] = R_mag[2, 0] * my - R_mag[2, 1] * mx
    return H


def measurement_jacobian_pressure_only(x_nom, init_alt):
    """Pressure-only measurement Jacobian H (1x5)."""
    altitude = x_nom[0]
    H = np.zeros((1, 5), dtype=np.float32)

    base = 1.0 - (altitude + init_alt) / 44330.8
    if base > 0:
        dp_dalt = 101325 * 5.255883 * np.power(base, 4.255883) * (-1.0 / 44330.8)
    else:
        dp_dalt = 0.0
    H[0, 0] = dp_dalt
    return H


# =====================================================================
# Process noise
# =====================================================================

def process_noise_matrix(x_nom, u, dt, qvar):
    """Discrete process noise covariance Q_d (5x5)."""
    return np.diag(qvar * dt).astype(np.float32)
