"""
ESKF dynamics, measurement model, and Jacobians (vertical-only position/velocity).

Nominal state vector (6):
    [pos_z, vel_z, quat(w,x,y,z)]
    Indices: 0 pos_z, 1 vel_z, 2-5 quat(w,x,y,z)

Error state vector (5):
    [δpos_z, δvel_z, δθ(3)]
    Indices: 0 δpos_z, 1 δvel_z, 2-4 δθ

Control input (6):
    [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    Raw IMU in sensor frame, with fixed biases already subtracted.

Measurement (4):
    [pressure, mag_x, mag_y, mag_z]

Sensor-to-board rotation matrices
----------------------------------
Board frame: +X forward, +Y left, +Z up (right-handed).

Two hardware versions exist:
  v1: IMU rotated +45° on PCB → Rz(-45°) to get to board frame
  v2: IMU rotated -45° on PCB → Rz(+45°) to get to board frame

Magnetometer: same on both versions.
  sensor-to-board = Rz(90°) @ Fz  →  board-to-sensor = transpose of that
"""

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
    [ 0.0,  0.0, -1.0],
], dtype=np.float32)

# ---- v1 hardware (legacy PCB — all existing datasets) ----
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
    """Returns the 3x3 skew-symmetric matrix of vector v."""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ], dtype=np.float32)


def quat_to_rotation_matrix(quat):
    """Convert a numpy-quaternion to a 3x3 rotation matrix."""
    return q.as_rotation_matrix(quat)


def _imu_to_board(accel_sensor, gyro_sensor, R_imu=None):
    """
    Transform raw IMU readings from sensor frame to board frame and convert units.

    Accelerometer: g-units → m/s² (board frame)
    Gyroscope:     deg/s   → rad/s (board frame)

    Returns (accel_board_ms2, gyro_board_rads).
    """
    if R_imu is None:
        R_imu = R_IMU_TO_BOARD
    accel_board = R_imu @ accel_sensor * GRAVITY  # g → m/s²
    gyro_board = R_imu @ np.deg2rad(gyro_sensor)  # deg/s → rad/s
    return accel_board, gyro_board


# =====================================================================
# Nominal predict
# =====================================================================

def nominal_predict(x_nom, u, dt, R_imu=None):
    """
    Propagate the 6-dim nominal state forward by dt.

    Parameters
    ----------
    x_nom : ndarray (6,)
    u     : ndarray (6,) — [accel_xyz, gyro_xyz] sensor frame, bias pre-subtracted
    dt    : float
    R_imu : ndarray (3,3) — sensor-to-board rotation (default: R_IMU_TO_BOARD)
    """
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
    """
    Compute the discrete error-state transition Jacobian F_d (5x5).

    Error state: [δpos_z, δvel_z, δθ(3)]

    F_d ≈ I + F_c * dt  (first-order)
    """
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


def error_state_jacobian_init(x_nom, u, dt, R_imu=None):
    """Init-phase Jacobian: clamp pos/vel dynamics, only angular."""
    _, w_board_rads = _imu_to_board(u[0:3], u[3:6], R_imu)

    F = np.eye(5, dtype=np.float32)

    # Clamp position/velocity error dynamics
    F[0, 0] = 0.0
    F[1, 1] = 0.0

    # Angular dynamics
    F[2:5, 2:5] += -skew(w_board_rads) * dt

    return F


# =====================================================================
# Measurement model
# =====================================================================

def measurement_function(x_nom, init_pressure, mag_world, R_mag=None):
    """
    Compute predicted measurement from nominal state.

    z_pred = [pressure, mag_sensor(3)]
    """
    if R_mag is None:
        R_mag = R_MAG_TO_BOARD
    altitude = x_nom[0]
    quat = q.from_float_array(x_nom[2:6]).normalized()

    # barometric pressure from altitude
    pressure = init_pressure * np.power(1.0 - (altitude / 44330.0), 5.255876)

    # magnetometer: rotate world mag into board frame, then into mag sensor frame
    R_board_to_world = quat_to_rotation_matrix(quat)
    mag_board = R_board_to_world.T @ mag_world
    mag_sensor = R_mag.T @ mag_board

    return np.array([pressure, mag_sensor[0], mag_sensor[1], mag_sensor[2]], dtype=np.float32)


def measurement_function_pressure_only(x_nom, init_pressure):
    """Pressure-only predicted measurement (1-dim)."""
    altitude = x_nom[0]
    pressure = init_pressure * np.power(1.0 - (altitude / 44330.0), 5.255876)
    return np.array([pressure], dtype=np.float32)


def measurement_jacobian(x_nom, init_pressure, mag_world, R_mag=None):
    """
    Compute the measurement Jacobian H (4x5).

    Error state: [δpos_z, δvel_z, δθ(3)]

    Non-zero blocks:
        H[0, 2]    : ∂pressure/∂altitude
        H[1:4, 6:9]: ∂mag_sensor/∂δθ
    """
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
    H[1:4, 2:5] = R_mag.T @ skew(mag_board)

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
    """
    Compute the discrete process noise covariance Q_d (5x5).

    Parameters
    ----------
    qvar : ndarray (9,) — diagonal noise scaling per error-state dimension
    """
    return np.diag(qvar * dt).astype(np.float32)
