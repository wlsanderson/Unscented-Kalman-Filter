import numpy as np
from enum import Enum


# state vector constants
STATE_DIM = 3
INITIAL_STATE_ESTIMATE = np.array([0.0, 0.0, 9.8]) #alt, vel_z, acc_z

# initial state covariance
INITIAL_STATE_COV = np.diag([1.0, 1.0, 0.1])

# Process covariance
class StateProcessCovariance(Enum):
    STANDBY_QVAR = 1e-6
    MOTOR_BURN_QVAR = 1e6
    COAST_QVAR = 1
    FREEFALL_QVAR = 10
    LANDED_QVAR = 1e-6

# measurement vector constants
MEASUREMENT_DIM = 2

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

GRAVITY = 9.81
ROCKET_MASS = 19.46
AIR_DENSITY = 1.15
REFERENCE_AREA = 0.01929
DRAG_COEFFICIENT = 0.45

