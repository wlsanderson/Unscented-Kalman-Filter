import numpy as np
import numpy.typing as npt
from enum import Enum


STATE_DIM = 16
"""Number of states in the state vector"""


INITIAL_STATE_ESTIMATE = np.array([
    0.0, 0.0, 0.0, # position (x, y, z)
    0.0, 0.0, 0.0, # velocity (x, y, z)
    0.0, 0.0, 1.0, # accel (x, y, z)
    0.0, 0.0, 0.0, # gyro (x, y, z)
    1, 0, 0, 0, # quaternion orientation (w, x, y, z)
    ])
"""State vector initial estimate"""

# initial state covariance
INITIAL_STATE_COV = np.diag([
    1e-6, 1e-6, 1e-6, # position (x, y, z)
    1e-6, 1e-6, 1e-6, # velocity (x, y, z)
    1e-2, 1e-2, 1e-2, # accel (x, y, z)
    1e-5, 1e-5, 1e-5, # gyro (x, y, z)
    1, 1, 1, # quaternion orientation (w, x, y, z)
])

class States(Enum):
    """Represents the state names and associated index of state vector"""
    POS_X = 0
    POS_Y = 1
    POS_Z = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    VELOCITY_Z = 5
    ACCEL_X = 6
    ACCEL_Y = 7
    ACCEL_Z = 8
    GYRO_X = 9
    GYRO_Y = 10
    GYRO_Z = 11
    QUATERNION_W = 12
    QUATERNION_X = 13
    QUATERNION_Y = 14
    QUATERNION_Z = 15




# measurement vector constants
MEASUREMENT_DIM = 10
MEASUREMENT_FIELDS = [
    "pressure",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    ]

class StateNum(Enum):
    """
    Enum that represents state number for the state transition function for each flight state.
    """
    STANDBY = 0
    MOTOR_BURN = 1
    COAST = 2
    FREEFALL = 3
    LANDED = 4

    @property
    def val(self) -> np.float32:
        """Returns as numpy float"""
        return np.float32(self.value)


class StateProcessCovariance(Enum):
    """
    Enum that represents process variance scalars on the diagonal of the process noise covariance
    matrix for each flight state.
    """

    STANDBY = (
        [1e-5, 1e-5, 1e-5, # position (x, y, z)
         1e-5, 1e-5, 1e-5, # velocity (x, y, z)
         1e-3, 1e-3, 1e-3, # acceleration (x, y, z)
         1, 1, 1, # gyro (x, y, z)
         1e-1, 1e-1, 1e-1] # orientation (r, p, y)
        ,)

    MOTOR_BURN = (
        [1, 1, 1e-3, # position (x, y, z)
         1e-1, 1e-1, 1e-3, # velocity (x, y, z)
         1, 1, 3e1, # acceleration (x, y, z)
         1, 1, 1, # gyro (x, y, z)
         1, 1, 1] # orientation (r, p, y)
        ,)
    COAST = (
        [1e-2, 1e-2, 1e-2, # position (x, y, z)
         1e-3, 1e-3, 1e-3, # velocity (x, y, z)
         1e1, 1e1, 1e1, # acceleration (x, y, z)
         1e3, 1e3, 1e3, # gyro (x, y, z)
         1e1, 1e1, 1e1] # orientation (r, p, y)
        ,)
    FREEFALL = (
        [1e-1, 1e-1, 1e-1, # position (x, y, z)
         1, 1, 1, # velocity (x, y, z)
         1, 1, 1, # acceleration (x, y, z)
         1e2, 1e2, 1e2, # gyro (x, y, z)
         1e1, 1e1, 1e1] # orientation (r, p, y)
        ,)
    LANDED = (
        [1, 1, 1, # position (x, y, z)
         1, 1, 1, # velocity (x, y, z)
         1e2, 1e2, 1e2, # acceleration (x, y, z)
         1e2, 1e2, 1e2, # gyro (x, y, z)
         1, 1, 1] # orientation (r, p, y)
        ,)

    @property
    def array(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0], dtype=np.float32)


class StateMeasurementNoise(Enum):
    """Enum that represents measurement noise covariance diagonal matrices for each flight state"""

    STANDBY = ([5e1, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2],)
    MOTOR_BURN = ([1e2, 5e-2, 5e-2, 5e-2, 1, 1, 1, 1e-2, 1e-2, 1e-2],)
    COAST = ([5e2, 1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-3, 1e-3, 1e-3],)
    FREEFALL = ([5e1, 1e-1, 1e-1, 1e-1, 1e2, 1e2, 1e2, 1e-1, 1e-1, 1e-1],)
    LANDED = ([5e1, 1e-2, 1e-2, 1e-2, 1e1, 1e1, 1e1, 1e-1, 1e-1, 1e-1],)

    @property
    def matrix(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0], dtype=np.float32)



# Sigma Point Constants
ALPHA = 0.3
BETA = 2
KAPPA = 0

# State changes
TAKEOFF_ACCELERATION_GS = 2
MAX_VELOCITY_THRESHOLD = 0.98
MAX_ALTITUDE_THRESHOLD = 0.99
LANDED_ACCELERATION_GS = 5
GROUND_ALTITUDE_METERS = 20

# aerodynamic constants
GRAVITY = 9.798

# log files
TIMESTAMP_COL_NAME = "timestamp"
TIMESTAMP_UNITS = 1


# =============================================================================
# ESKF (Error-State Extended Kalman Filter) Constants
# =============================================================================

ESKF_NOMINAL_DIM = 10
"""Nominal state dimension: pos(3) + vel(3) + quat(4)"""

ESKF_ERROR_DIM = 9
"""Error state dimension: δpos(3) + δvel(3) + δθ(3)"""

ESKF_MEASUREMENT_DIM = 4
"""Measurement dimension: pressure(1) + magnetometer(3)"""

ESKF_CONTROL_DIM = 6
"""Control input dimension: accel(3) + gyro(3), from IMU in sensor frame"""

ESKF_PRESSURE_VEL_COUPLING_SPEED = 20.0
"""Speed (m/s) below which pressure corrections fully couple to velocity."""

ESKF_PRESSURE_VEL_COUPLING_SHARPNESS = 1
"""Sigmoid sharpness for the coupling transition.
At 0.5: ~99% coupling at 20 m/s, ~50% at 30 m/s, ~1% at 40 m/s.
Increase for a sharper cutoff (1.0 ≈ 5 m/s band, 2.0 ≈ 2.5 m/s band)."""


class ESKFNominalStates(Enum):
    """Index mapping for the 10-element nominal state vector."""
    POS_X = 0
    POS_Y = 1
    POS_Z = 2
    VEL_X = 3
    VEL_Y = 4
    VEL_Z = 5
    QUAT_W = 6
    QUAT_X = 7
    QUAT_Y = 8
    QUAT_Z = 9


class ESKFErrorStates(Enum):
    """Index mapping for the 9-element error state vector."""
    DPOS_X = 0
    DPOS_Y = 1
    DPOS_Z = 2
    DVEL_X = 3
    DVEL_Y = 4
    DVEL_Z = 5
    DTHETA_X = 6
    DTHETA_Y = 7
    DTHETA_Z = 8


ESKF_INITIAL_STATE_ESTIMATE = np.array([
    0.0, 0.0, 0.0,         # position (x, y, z)
    0.0, 0.0, 0.0,         # velocity (x, y, z)
    1.0, 0.0, 0.0, 0.0,    # quaternion (w, x, y, z)
])
"""ESKF nominal state initial estimate (10-dim)"""

ESKF_INITIAL_STATE_COV = np.diag([
    1e-6, 1e-6, 1e-6,      # δposition
    1e-6, 1e-6, 1e-6,      # δvelocity
    1e-3, 1e-3, 1e-3,      # δθ (angular error)
]).astype(np.float64)
"""ESKF error-state initial covariance (9×9)"""


ESKF_Q_DIAG = np.array([
    1e-1, 1e-1, 1e-2,     # δposition
    1e-1, 1e-1, 1e-2,     # δvelocity
    1e-3, 1e-3, 1e-3,     # δθ
], dtype=np.float64)
"""Process noise diagonal (9 elements). Matches C eskf_q_diag."""

ESKF_R_DIAG = np.array([
    5e1, 1e-3, 1e-3, 1e-3,
], dtype=np.float64)
"""Measurement noise diagonal (4 elements). Matches C eskf_r_diag."""
