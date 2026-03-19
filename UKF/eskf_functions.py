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


def quat_to_rotation_matrix(quat):
    """Convert a numpy-quaternion to a 3x3 rotation matrix."""
    return q.as_rotation_matrix(quat)


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

    # rotation matrix: board -> world
    quat = q.from_float_array(x[2:6]).normalized()
    R_b2w = quat_to_rotation_matrix(quat)

    # world-frame acceleration
    a_world = R_b2w @ a_board_ms2
    a_world[2] -= GRAVITY

    # integrate position and velocity
    x[0] += x[1] * dt
    x[1] += a_world[2] * dt

    # quaternion integration
    delta_theta = w_board_rads * dt
    delta_q = q.from_rotation_vector(delta_theta)
    new_quat = quat * delta_q
    x[2:6] = q.as_float_array(new_quat)

    return x


# =====================================================================
# Error-state Jacobian
# =====================================================================

def error_state_jacobian(x_nom, u, dt, R_imu=None):
    """Discrete error-state Jacobian F_d (5x5)."""
    a_board_ms2, w_board_rads = _imu_to_board(u[0:3], u[3:6], R_imu)

    quat = q.from_float_array(x_nom[2:6]).normalized()
    R_b2w = quat_to_rotation_matrix(quat)

    F = np.eye(5, dtype=np.float32)

    # δẑ += δv_z * dt
    F[0, 1] += dt

    # δv̇_z = -(R @ skew(a_board))_z * δθ * dt
    F[1, 2:5] = (-R_b2w @ skew(a_board_ms2))[2, :] * dt

    # δθ̇ = -skew(ω) @ δθ * dt
    F[2:5, 2:5] += -skew(w_board_rads) * dt

    return F


# =====================================================================
# Measurement model
# =====================================================================

def measurement_function(x_nom, init_pressure, mag_world, R_mag=None):
    """Predicted measurement: pressure + mag (sensor frame)."""
    if R_mag is None:
        R_mag = R_MAG_TO_BOARD
    altitude = x_nom[0]
    quat = q.from_float_array(x_nom[2:6]).normalized()

    # barometric pressure from altitude
    pressure = init_pressure * np.power(1.0 - (altitude / 44330.0), 5.255876)

    # magnetometer: rotate world mag into board frame, then into mag sensor frame
    # R_mag is stored as board→sensor (matches C eskf_config.c).
    R_board_to_world = quat_to_rotation_matrix(quat)
    mag_board = R_board_to_world.T @ mag_world
    mag_sensor = R_mag @ mag_board

    return np.array([pressure, mag_sensor[0], mag_sensor[1], mag_sensor[2]], dtype=np.float32)


def measurement_function_pressure_only(x_nom, init_pressure):
    """Pressure-only predicted measurement (1-dim)."""
    altitude = x_nom[0]
    pressure = init_pressure * np.power(1.0 - (altitude / 44330.0), 5.255876)
    return np.array([pressure], dtype=np.float32)


def measurement_jacobian(x_nom, init_pressure, mag_world, R_mag=None):
    """Measurement Jacobian H (4x5)."""
    if R_mag is None:
        R_mag = R_MAG_TO_BOARD
    altitude = x_nom[0]
    quat = q.from_float_array(x_nom[2:6]).normalized()

    H = np.zeros((4, 5), dtype=np.float32)

    # ∂pressure/∂altitude
    base = 1.0 - altitude / 44330.0
    if base > 0:
        dp_dalt = init_pressure * 5.255876 * np.power(base, 4.255876) * (-1.0 / 44330.0)
    else:
        dp_dalt = 0.0
    H[0, 0] = dp_dalt

    # ∂mag_sensor/∂δθ
    R_b2w = quat_to_rotation_matrix(quat)
    mag_board = R_b2w.T @ mag_world
    H[1:4, 2:5] = R_mag @ skew(mag_board)

    return H


def measurement_jacobian_pressure_only(x_nom, init_pressure):
    """Pressure-only measurement Jacobian H (1x5)."""
    altitude = x_nom[0]
    H = np.zeros((1, 5), dtype=np.float32)

    base = 1.0 - altitude / 44330.0
    if base > 0:
        dp_dalt = init_pressure * 5.255876 * np.power(base, 4.255876) * (-1.0 / 44330.0)
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
