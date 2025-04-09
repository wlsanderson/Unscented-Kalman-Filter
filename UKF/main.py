from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path
from UKF.constants import States
import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_pitch(X_data):
    q = X_data[:, 6:10]  # [qw, qx, qy, qz]
    # Convert to scipy-friendly order: [qx, qy, qz, qw]
    q_scipy = np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
    r = R.from_quat(q_scipy)
    euler = r.as_euler('zyx', degrees=True)  # yaw, pitch, roll
    pitch = euler[:, 1]  # second column is pitch
    return pitch

def run():
    launch_log = Path("launch_data/pelicanator_launch_3.csv")

    min_r = 59300
    max_r=60000
    plot_state = [States.ACCELERATION.value]
    #plot_state = compute_pitch

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