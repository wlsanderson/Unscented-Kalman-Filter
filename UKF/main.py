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
    launch_log = Path("launch_data/test_standby_alt.csv")

    min_r = 2
    max_r = 100000


    plotter = Plotter(file_path=launch_log, min_r=min_r, max_r=max_r)
    data_processor = DataProcessor(launch_log, min_r=min_r, max_r=max_r)
    context = Context(data_processor, plotter=plotter)
    run_data_loop(context)

def run_data_loop(context: Context):
    while True:
        context.update()
        
        if context.shutdown_requested:
            break


if __name__ == "__main__":
    run()