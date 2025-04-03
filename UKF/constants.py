import numpy as np
import numpy.typing as npt
from enum import Enum

# state vector constants
STATE_DIM = 3
INITIAL_STATE_ESTIMATE = np.array([0.0, 0.0, 9.8]) #alt, vel_z, acc_z

# initial state covariance
INITIAL_STATE_COV = np.diag([1.0, 1.0, 0.1])

# measurement vector constants
MEASUREMENT_DIM = 2
MEASUREMENT_FIELDS = ["estPressureAlt", "scaledAccelZ"]

class StateProcessCovariance(Enum):
    """Enum that represents process variance scalar for each flight state"""

    STANDBY = 1e-6
    MOTOR_BURN= 1e6
    COAST = 1
    FREEFALL = 10
    LANDED = 1e-6


class StateMeasurementNoise(Enum):
    """Enum that represents measurement noise covariance diagonal matrices for each flight state"""

    STANDBY = ([0.44025, 0.571041128017971e-5],)
    MOTOR_BURN = ([70, 0.002],)
    COAST = ([0.04275, 2.17e-4],)
    FREEFALL = ([0.04275, 2.17e-4],)
    LANDED = ([0.44025, 0.571041128017971e-5],)

    @property
    def matrix(self) -> npt.NDArray:
        """Returns as diagonal numpy matrix and makes immutable"""
        return np.array(self.value[0])



# Sigma Point Constants
ALPHA = 0.5
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