import numpy as np
import numpy.typing as npt
from enum import Enum

# state vector constants
STATE_DIM = 3
"""Altitude, Vertical Velocity, Vertical Acceleration, qw, qx, qy, qz, Gyro X, Gyro Y, Gyro Z"""
INITIAL_STATE_ESTIMATE = np.array([0.0, 0.0, 9.8, 1.0, 0, 0, 0, 0, 0, 0])

class States(Enum):
    """Represents the state names and associated index of state vector"""
    ALTITUDE = 0
    VELOCITY = 1
    ACCELERATION = 2
    QUAT_W = 3
    QUAT_X = 4
    QUAT_Y = 5
    QUAT_Z = 6
    GYRO_X = 7
    GYRO_Y = 8
    GYRO_Z = 9


# initial state covariance
INITIAL_STATE_COV = np.diag([1.0, 1.0, 0.1, 1e6, 1e6, 1e6, 1e6, 0.1, 0.1, 0.1])

# measurement vector constants
MEASUREMENT_DIM = 10
MEASUREMENT_FIELDS = [
    "estPressureAlt",
    "scaledAccelX",
    "scaledAccelY",
    "scaledAccelZ",
    "scaledGyroX",
    "scaledGyroY",
    "scaledGyroZ",
    "magneticFieldX",
    "magneticFieldY",
    "magneticFieldZ",
    ]

class StateProcessCovariance(Enum):
    """Enum that represents process variance scalars on kinematic, quaternion, and gyro covariances"""

    STANDBY = ([1e-6, 1e-6, 1e-6],)
    MOTOR_BURN= ([1e6, 1, 1],)
    COAST = ([1e-4, 1, 1],)
    FREEFALL = ([10, 1e6, 1e6],)
    LANDED = ([1e-6, 1e-6, 1e-6],)

    @property
    def array(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])


class StateMeasurementNoise(Enum):
    """Enum that represents measurement noise covariance diagonal matrices for each flight state"""

    STANDBY = ([0.44025, 5e-6, 5e-6, 5e-6, 1e-3, 1e-3, 1e-3, 2, 2, 2],)
    MOTOR_BURN = ([70, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 2, 2, 2],)
    COAST = ([0.04, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 2, 2, 2],)
    FREEFALL = ([0.04, 1e-2, 1e-2, 1e-2, 1e2, 1e2, 1e2, 2, 2, 2],)
    LANDED = ([0.44025, 5e-6, 5e-6, 5e-6, 1e-3, 1e-3, 1e-3, 2, 2, 2],)

    @property
    def matrix(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])



# Sigma Point Constants
ALPHA = 0.1
BETA = 2
KAPPA = 3 - STATE_DIM

# State changes
TAKEOFF_ACCELERATION_GS = 6
MAX_VELOCITY_THRESHOLD = 0.96
MAX_ALTITUDE_THRESHOLD = 0.96
LANDED_ACCELERATION_GS = 5
GROUND_ALTITUDE_METERS = 20

# aerodynamic constants
GRAVITY = 9.81
ROCKET_MASS = 19.46
AIR_DENSITY = 1.15
REFERENCE_AREA = 0.01929
DRAG_COEFFICIENT = 0.45

# log files
TIMESTAMP_COL_NAME = "timestamp"
LOG_HEADER_STATES = {0: "current_altitude", 1: "vertical_velocity", 2: "scaledAccelZ"}