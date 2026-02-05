from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path
import numpy as np
import quaternion as q
from UKF.ukf_functions import print_c_array

quat_a = q.from_float_array(np.array([0.1, 0.2, 0.3, 0.4]))
quat_b = q.from_float_array(np.array([4.2, -5.6, 0.0, 0.1]))
quat_a = quat_a.normalized()
quat_b = quat_b.normalized()
print(quat_a * quat_b)

