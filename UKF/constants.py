import numpy as np
from enum import Enum

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

# aerodynamic constants
GRAVITY = 9.798

# log files
TIMESTAMP_COL_NAME = "timestamp"
TIMESTAMP_UNITS = 1


# =============================================================================
# ESKF (Error-State Extended Kalman Filter) Constants
# =============================================================================

ESKF_NOMINAL_DIM = 6
"""Nominal state dimension: pos(1) + vel(1) + quat(4)"""

ESKF_ERROR_DIM = 5
"""Error state dimension: δpos(1) + δvel(1) + δθ(3)"""

ESKF_MEASUREMENT_DIM = 4
"""Measurement dimension: pressure(1) + magnetometer(3)"""

ESKF_CONTROL_DIM = 6
"""Control input dimension: accel(3) + gyro(3), from IMU in sensor frame"""

ESKF_PRESSURE_VEL_COUPLING_SPEED = 20
"""Speed (m/s) below which pressure corrections fully couple to velocity."""

ESKF_PRESSURE_VEL_COUPLING_SHARPNESS = 3
"""Sigmoid sharpness for the coupling transition.
At 0.5: ~99% coupling at 20 m/s, ~50% at 30 m/s, ~1% at 40 m/s.
Increase for a sharper cutoff (1.0 ≈ 5 m/s band, 2.0 ≈ 2.5 m/s band)."""


class ESKFNominalStates(Enum):
    """Index mapping for the 6-element nominal state vector."""
    POS_Z = 0
    VEL_Z = 1
    QUAT_W = 2
    QUAT_X = 3
    QUAT_Y = 4
    QUAT_Z = 5


class ESKFErrorStates(Enum):
    """Index mapping for the 5-element error state vector."""
    DPOS_Z = 0
    DVEL_Z = 1
    DTHETA_X = 2
    DTHETA_Y = 3
    DTHETA_Z = 4


ESKF_INITIAL_STATE_ESTIMATE = np.array([
    0.0,         # position (x, y, z)
    0.0,         # velocity (x, y, z)
    1.0, 0.0, 0.0, 0.0,    # quaternion (w, x, y, z)
])
"""ESKF nominal state initial estimate (6-dim)"""

ESKF_INITIAL_STATE_COV = np.diag([
    1e-6,      # δposition
    1e-6,      # δvelocity
    1e-3, 1e-3, 1e-3,      # δθ (angular error)
]).astype(np.float64)
"""ESKF error-state initial covariance (5x5)"""


ESKF_Q_DIAG = np.array([
    1e-2,     # δposition
    5e-2,     # δvelocity
    1e-4, 1e-4, 1e-4,     # δθ
], dtype=np.float64)
"""Process noise diagonal (5 elements). Matches C eskf_q_diag."""

ESKF_R_DIAG = np.array([
    5e1, 1e-1, 1e-1, 1e-1,
], dtype=np.float64)
"""Measurement noise diagonal (4 elements). Matches C eskf_r_diag."""

ESKF_R_DIAG_PRESSURE = np.array([
    5e1,
], dtype=np.float64)
"""Measurement noise diagonal for pressure-only mode (1 element)."""
