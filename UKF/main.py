from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path
from UKF.constants import States
import numpy as np
import quaternion

def compute_pitch(X_data):
    q = X_data[:, 8:12]
    r = quaternion.from_float_array(q)
    euler = quaternion.as_euler_angles(r)
    pitch = euler[:, 1]  # second column is pitch
    return pitch * (180/np.pi)

def run():
    launch_log = Path("launch_data/pelicanator_launch_2.csv")

    min_r = 5002
    max_r=9700
    #plot_state = [States.GYRO_Y.value, States.GYRO_Z.value]
    #plot_state = [States.QUAT_X.value, States.QUAT_Z.value]
    #plot_state = [States.ACCELERATION_Z.value, compute_pitch]
    plot_state = compute_pitch

    plotter = Plotter(state_index=plot_state, file_path=launch_log, min_r=min_r, max_r=max_r)
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