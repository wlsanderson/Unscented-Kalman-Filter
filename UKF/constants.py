import numpy as np
import numpy.typing as npt
from enum import Enum


STATE_DIM = 22
"""Number of states in the state vector"""


INITIAL_STATE_ESTIMATE = np.array([
    0.0, 0.0, 0.0, # position (x, y, z)
    0.0, 0.0, 0.0, # velocity (x, y, z)
    0.0, 0.0, 1.0, # accel (x, y, z)
    0.0, 0.0, 0.0, # gyro (x, y, z)
    0, 0, 0, # accelerometer offsets (x, y, z)
    0, 0, 0, # gyro offsets (x, y, z)
    1, 0, 0, 0, # quaternion orientation (w, x, y, z)
    ])
"""State vector initial estimate"""

# initial state covariance
INITIAL_STATE_COV = np.diag([
    1e-6, 1e-6, 1e-6, # position (x, y, z)
    1e-6, 1e-6, 1e-6, # velocity (x, y, z)
    1e-2, 1e-2, 1e-2, # accel (x, y, z)
    1e-5, 1e-5, 1e-5, # gyro (x, y, z)
    1e-4, 1e-4, 1e-4, # accelerometer offsets (x, y, z)
    1e-4, 1e-4, 1e-4, # gyro offsets (x, y, z)
    1e-1, 1e-1, 1e-1, # quaternion orientation (w, x, y, z)
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
    ACC_OFFSET_X = 12
    ACC_OFFSET_Y = 13
    ACC_OFFSET_Z = 14
    GYRO_OFFSET_X = 15
    GYRO_OFFSET_Y = 16
    GYRO_OFFSET_Z = 17
    QUATERNION_W = 18
    QUATERNION_X = 19
    QUATERNION_Y = 20
    QUATERNION_Z = 21




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

class StateControlInput(Enum):
    """
    Enum that represents control inputs for the state transition function for each flight state.
    """
    STANDBY = (
        [0, 0, 0, # position (x, y, z)
         0, 0, 0,] # velocity (x, y, z)
         ,)
    MOTOR_BURN = ([None],)
    COAST = ([None],)
    FREEFALL = ([None],)
    LANDED = ([0, 0, 0],) # velocity (x, y, z)

    @property
    def array(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])


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
         0, 0, 0, # accelerometer offset (x, y, z)
         0, 0, 0, # gyroscope offset (x, y, z)
         1e-1, 1e-1, 1e-1] # orientation (r, p, y)
        ,)

    MOTOR_BURN = (
        [1e-2, 1e-2, 1e-2, # position (x, y, z)
         1e-2, 1e-2, 1e-2, # velocity (x, y, z)
         1e1, 1e1, 1e1, # acceleration (x, y, z)
         1e4, 1e4, 1e4, # gyro (x, y, z)
         0, 0, 0, # accelerometer offset (x, y, z)
         0, 0, 0, # gyroscope offset (x, y, z)
         1e2, 1e2, 1e2] # orientation (r, p, y)
        ,)
    COAST = (
        [1e-2, 1e-2, 1e-2, # position (x, y, z)
         1e-3, 1e-3, 1e-3, # velocity (x, y, z)
         1e1, 1e1, 1e1, # acceleration (x, y, z)
         1e3, 1e3, 1e3, # gyro (x, y, z)
         0, 0, 0, # accelerometer offset (x, y, z)
         0, 0, 0, # gyroscope offset (x, y, z)
         1e1, 1e1, 1e1] # orientation (r, p, y)
        ,)
    FREEFALL = (
        [1e-1, 1e-1, 1e-1, # position (x, y, z)
         1, 1, 1, # velocity (x, y, z)
         1, 1, 1, # acceleration (x, y, z)
         1e2, 1e2, 1e2, # gyro (x, y, z)
         0, 0, 0, # accelerometer offset (x, y, z)
         0, 0, 0, # gyroscope offset (x, y, z)
         1e1, 1e1, 1e1] # orientation (r, p, y)
        ,)
    LANDED = (
        [1, 1, 1, # position (x, y, z)
         1, 1, 1, # velocity (x, y, z)
         1e2, 1e2, 1e2, # acceleration (x, y, z)
         1e2, 1e2, 1e2, # gyro (x, y, z)
         0, 0, 0, # accelerometer offset (x, y, z)
         0, 0, 0, # gyroscope offset (x, y, z)
         1, 1, 1] # orientation (r, p, y)
        ,)

    @property
    def array(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])


class StateMeasurementNoise(Enum):
    """Enum that represents measurement noise covariance diagonal matrices for each flight state"""

    STANDBY = ([1e2, 1e-2, 1e-2, 1e-2, 3e-1, 3e-1, 3e-1, 1e-2, 1e-2, 1e-2],)
    MOTOR_BURN = ([1e3, 1e-2, 1e-2, 1e-2, 1e2, 1e2, 1e2, 1e-3, 1e-3, 1e-3],)
    COAST = ([1e3, 1e-1, 1e-1, 1e-1, 1, 1, 1, 1e-3, 1e-3, 1e-3],)
    FREEFALL = ([1e3, 1e-1, 1e-1, 1e-1, 1e2, 1e2, 1e2, 1e-1, 1e-1, 1e-1],)
    LANDED = ([1e1, 1e-2, 1e-2, 1e-2, 1e1, 1e1, 1e1, 1e-1, 1e-1, 1e-1],)

    @property
    def matrix(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])



# Sigma Point Constants
ALPHA = 1e-3
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
MIN_VEL_FOR_DRAG = 25.0  # m/s
DRAG_PARAM = -2.5e-4

# log files
TIMESTAMP_COL_NAME = "timestamp"
TIMESTAMP_UNITS = 1
