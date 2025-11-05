from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path

import numpy as np
import quaternion

def compute_pitch(X_data):
    q = X_data[:, 8:12]
    r = quaternion.from_float_array(q)
    euler = quaternion.as_euler_angles(r)
    pitch = euler[:, 1]  # second column is pitch
    return pitch * (180/np.pi)

def run():
    launch_log = np.array([Path("launch_data/pressure_sensor_data.csv"), Path("launch_data/imu_data.csv"), Path("launch_data/magnetometer_data.csv")])

    min_t = 0
    max_t = 15


    plotter = Plotter()
    data_processor = DataProcessor(bmp_data = launch_log[0], imu_data = launch_log[1], mag_data = launch_log[2], min_t=min_t, max_t=max_t)
    context = Context(data_processor, plotter)
    run_data_loop(context)
    

def run_data_loop(context: Context):
    while True:
        context.update()
        if context.shutdown_requested:
            break


if __name__ == "__main__":
    run()