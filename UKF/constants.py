import numpy as np
import numpy.typing as npt
from enum import Enum

# state vector constants
STATE_DIM = 7
"""Gyro X, Gyro Y, Gyro Z, qw, qx, qy, qz"""
INITIAL_STATE_ESTIMATE = np.array([0.0, 0.0, 0.0, 0.4686, 0, -0.01765, -0.88337])

class States(Enum):
    """Represents the state names and associated index of state vector"""
    GYRO_X = 0
    GYRO_Y = 1
    GYRO_Z = 2
    QUAT_W = 3
    QUAT_X = 4
    QUAT_Y = 5
    QUAT_Z = 6


# initial state covariance
INITIAL_STATE_COV = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# measurement vector constants
MEASUREMENT_DIM = 3
MEASUREMENT_FIELDS = [
    "estAngularRateX",
    "estAngularRateY",
    "estAngularRateZ",
    ]

class StateProcessCovariance(Enum):
    """Enum that represents process variance scalars on gyro covariances"""
    # gyro x, gyro y, gyro z
    STANDBY = ([1e-4, 1e-4, 2e-3],)
    MOTOR_BURN= ([3e-1, 1e-1, 1],)
    COAST = ([1, 1, 1],)
    FREEFALL = ([1e6, 1e6, 1e6],)
    LANDED = ([1e-6, 1e-6, 1e-6],)

    @property
    def array(self) -> npt.NDArray:
        """Returns as numpy array and makes immutable"""
        return np.array(self.value[0])


class StateMeasurementNoise(Enum):
    """Enum that represents measurement noise covariance diagonal matrices for each flight state"""

    STANDBY = ([2e-4, 2e-4, 5e-3],)
    MOTOR_BURN = ([1e-3, 3e-4, 1e-3],)
    COAST = ([1e-4, 1e-4, 1e-4],)
    FREEFALL = ([1e2, 1e2, 1e2],)
    LANDED = ([1e-3, 1e-3, 1e-3],)   

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
MB_TIME = 1741461573291483136
C_TIME = 1741461575713260288
FF_TIME = 1741461588650890240
L_TIME = 1741461642655990528

# aerodynamic constants
GRAVITY = 9.798
ROCKET_MASS = 19.46
AIR_DENSITY = 1.15
REFERENCE_AREA = 0.01929
DRAG_COEFFICIENT = 0.45

# log files
TIMESTAMP_COL_NAME = "timestamp"
TIMESTAMP_UNITS = 1e9
LOG_HEADER_STATES = {
    0: "estAngularRateX",
    1: "estAngularRateY",
    2: "estAngularRateZ",
    3: "estOrientQuaternionW",
    4: "estOrientQuaternionX",
    5: "estOrientQuaternionY",
    6: "estOrientQuaternionZ",
    }