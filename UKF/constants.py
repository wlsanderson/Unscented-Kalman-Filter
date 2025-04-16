import numpy as np
import numpy.typing as npt
from enum import Enum

# state vector constants
STATE_DIM = 12
"""Altitude, Vertical Velocity, accel x, accel y, accel z, Gyro X, Gyro Y, Gyro Z, qw, qx, qy, qz"""
INITIAL_STATE_ESTIMATE = np.array([0, 0, 0, 0, -9.81, 0.0, 0.0, 0.0, 0.6436035, 0.68179065, 0.01206461, 0.3475495])


class States(Enum):
    """Represents the state names and associated index of state vector"""
    ALTITUDE = 0
    VELOCITY = 1
    ACCELERATION_X = 2
    ACCELERATION_Y = 3
    ACCELERATION_Z = 4
    GYRO_X = 5
    GYRO_Y = 6
    GYRO_Z = 7
    QUAT_W = 8
    QUAT_X = 9
    QUAT_Y = 10
    QUAT_Z = 11


# initial state covariance
INITIAL_STATE_COV = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 0.1, 0.1, 0.1, 0.1])

# measurement vector constants
MEASUREMENT_DIM = 7
MEASUREMENT_FIELDS = [
    "alt_interp_rand",
    "scaledAccelX",
    "scaledAccelY",
    "scaledAccelZ",
    "estAngularRateX",
    "estAngularRateY",
    "estAngularRateZ",
    ]
# MEASUREMENT_FIELDS = [
#     "pressureAlt",
#     "estCompensatedAccelX",
#     "estCompensatedAccelY",
#     "estCompensatedAccelZ",
#     "estAngularRateX",
#     "estAngularRateY",
#     "estAngularRateZ",
#     "magneticFieldX",
#     "magneticFieldY",
#     "magneticFieldZ",

#     ]

class StateProcessCovariance(Enum):
    """Enum that represents process variance scalars on kinematic and gyro covariances"""
    # acc x, acc y, acc z, gyro x, gyro y, gyro z
    STANDBY = ([1e-1, 1e-1, 1e-2, 1e-4, 1e-4, 1e-5],)
    MOTOR_BURN= ([1e2, 1e2, 1e2, 1, 1, 1],)
    COAST = ([1e-4, 1e-4, 1e-4, 1, 1, 1],)
    FREEFALL = ([10, 10, 10, 1e6, 1e6, 1e6],)
    LANDED = ([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],)

    @property
    def array(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])


class StateMeasurementNoise(Enum):
    """Enum that represents measurement noise covariance diagonal matrices for each flight state"""

    STANDBY = ([1, 1, 1, 1e-2, 1e-4, 1e-4, 1e-4],)
    MOTOR_BURN = ([1e-1, 1e-3, 1e-3, 1e-3, 3e-4, 3e-4, 1e-3],)
    COAST = ([0.04, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4],)
    FREEFALL = ([0.04, 1e-2, 1e-2, 1e-2, 1e2, 1e2, 1e2],)
    LANDED = ([0.44025, 5e-6, 5e-6, 5e-6, 1e-3, 1e-3, 1e-3],)   

    @property
    def matrix(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])

# Magnetic Field
# https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
class MagneticField(Enum):
    NED = ([22.223, -3.613, 43.318],)
    DECLINATION = -9 + (1/60) * -14 # degrees
    INCLINATION = 62 + (1/60) * 32 # degrees
    HORIZONTAL_INTENSITY = 22.515

    @property
    def vector(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])


# Sigma Point Constants
ALPHA = 0.3
BETA = 2
KAPPA = 0

# State changes
TAKEOFF_ACCELERATION_GS = 2
MAX_VELOCITY_THRESHOLD = 0.96
MAX_ALTITUDE_THRESHOLD = 0.96
LANDED_ACCELERATION_GS = 5
GROUND_ALTITUDE_METERS = 20

# aerodynamic constants
GRAVITY = 9.798
ROCKET_MASS = 19.46
AIR_DENSITY = 1.15
REFERENCE_AREA = 0.01929
DRAG_COEFFICIENT = 0.45

# log files
#TIMESTAMP_COL_NAME = "update_timestamp_ns"
TIMESTAMP_COL_NAME = "timestamp"
TIMESTAMP_UNITS = 1e9
#LOG_HEADER_STATES = {0: "current_altitude", 1: "vertical_velocity", 2: "estCompensatedAccelX", 3: "estAngularRateX",  4: "estAngularRateY", 5: "estAngularRateZ"}
LOG_HEADER_STATES = {
    0: "alt_interp",
    1: "vertical_velocity",
    2: "scaledAccelX",
    3: "scaledAccelY",
    4: "scaledAccelZ",
    5: "estAngularRateX",
    6: "estAngularRateY",
    7: "estAngularRateZ",
    8: "estOrientQuaternionW",
    9: "estOrientQuaternionX",
    10: "estOrientQuaternionY",
    11: "estOrientQuaternionZ",
    }