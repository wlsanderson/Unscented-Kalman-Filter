from UKF.context import Context
from UKF.plotter import Plotter
from UKF.data_processor import DataProcessor
from pathlib import Path

import numpy as np
import yaml


def run():
    launch_folder = Path("launch_data/government_work_launch_1_avab")
    #launch_folder = Path("launch_data/sailor")
    #launch_folder = Path("launch_data/lil_frank")
    #launch_folder = Path("launch_data/test")
    #launch_folder = Path("launch_data/jackpot_ab")
    launch_log = np.array([
        launch_folder / "BMP581_data.csv",
        launch_folder / "ICM45686_data.csv",
        launch_folder / "MMC5983MA_data.csv",
    ], dtype=object)

    # sailor
    #min_t = 1360
    #max_t = 1410

    # gov work avab
    min_t = 902
    max_t = 1000 - 60

    # gov work nc
    #min_t = 1273.42
    #max_t = 1297.4
    
    # lil frank
    #min_t = 1700
    #max_t = 1760

    # jackpot nc
    #min_t = 746
    #max_t = 800

    # jackpot ab
    #min_t = 1190
    #max_t = 1240


    # read calibration.yaml from the launch folder (if present)
    cal_file = launch_folder / "calibration.yaml"
    if cal_file.exists():
        with open(cal_file, "r") as f:
            cal_root = yaml.safe_load(f) or {}
        cal = cal_root.get("calibration", {})
        acc_offset = cal.get("accel_offset", [0, 0, 0])
        gyro_offset = cal.get("gyro_offset", [0, 0, 0])
        mag_offset = cal.get("mag_offset", [0, 0, 0])
        mag_scale = cal.get("mag_scale", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        imu_to_board = cal.get("imu_to_board", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mag_to_board = cal.get("mag_to_board", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        acc_offset = [0, 0, 0]
        gyro_offset = [0, 0, 0]
        mag_offset = [0, 0, 0]
        mag_scale = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        imu_to_board = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mag_to_board = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Export option: set to True to save `timestamps` and `X_data` to CSV after the run
    EXPORT_STATES_ON_EXIT = False
    EXPORT_STATES_FILENAME = "ukf_states.csv"

    plotter = Plotter(bmp_data_path=launch_log[0], min_t=min_t, max_t=max_t)
    data_processor = DataProcessor(
        bmp_data=launch_log[0],
        imu_data=launch_log[1],
        mag_data=launch_log[2],
        min_t=min_t,
        max_t=max_t,
        acc_cal_offset=acc_offset,
        gyro_cal_offset=gyro_offset,
        mag_cal_offset=mag_offset,
        mag_cal_scale=mag_scale,
        imu_to_board=imu_to_board,
        mag_to_board=mag_to_board,
    )
    context = Context(data_processor, plotter)
    run_data_loop(context)
    # After the run ends, optionally export the collected UKF states/timestamps
    if EXPORT_STATES_ON_EXIT:
        out_path = launch_folder / EXPORT_STATES_FILENAME
        try:
            plotter.export_states_csv(out_path)
        except Exception as e:
            print(f"Failed to export states CSV: {e}")
    

def run_data_loop(context: Context):
    while True:
        context.update()
        if context.shutdown_requested:
            break


if __name__ == "__main__":
    run()