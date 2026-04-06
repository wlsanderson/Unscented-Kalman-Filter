from UKF.eskf_context import ESKFContext
from UKF.plotter import Plotter, ESKF_STATE_LABELS, ESKF_MEASUREMENT_LABELS
from UKF.data_processor import DataProcessor
from pathlib import Path

import numpy as np
import yaml


# Set to False to run pressure-only measurements (skip magnetometer)
USE_MAGNETOMETER = True


def run():
    #launch_folder = Path("launch_data/government_work_launch_1_nc")
    #launch_folder = Path("launch_data/government_work_launch_1_avab")
    #launch_folder = Path("launch_data/sailor")
    #launch_folder = Path("launch_data/lil_frank")
    #launch_folder = Path("launch_data/jackpot_1_nc")
    #launch_folder = Path("launch_data/jackpot_1_ab")
    #launch_folder = Path("launch_data/jackpot_2_ab")
    #launch_folder = Path("launch_data/jackpot_2_grave")
    #launch_folder = Path("launch_data/jackpot_2_zombie") # bad pressure data
    #launch_folder = Path("launch_data/kai")
    launch_folder = Path("launch_data/jackpot_3_ab")

    match str(launch_folder.name):
        case "government_work_launch_1_nc":
            min_t = 1273.42
            max_t = 1297.4
        case "government_work_launch_1_avab":
            min_t = 902
            max_t = 950
        case "sailor":
            min_t = 1360
            max_t = 1410
        case "lil_frank":
            min_t = 1700
            max_t = 1760
        case "jackpot_1_nc":
            min_t = 746
            max_t = 800
        case "jackpot_1_ab":
            min_t = 1205
            max_t = 1222
        case "jackpot_2_ab":
            min_t = 883.37 - 800
            max_t = 901
        case "jackpot_2_grave":
            min_t = 1503.163
            max_t = 1530
        case "jackpot_2_zombie":
            min_t = 1178
            max_t = 1208
        case "kai":
            min_t = 1170
            max_t = 1200
        case "jackpot_3_ab":
            min_t = 780
            max_t = 805

    launch_log = np.array([
        launch_folder / "BMP581_data.csv",
        launch_folder / "ICM45686_data.csv",
        launch_folder / "MMC5983MA_data.csv",
    ], dtype=object)
    
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
        hardware_version = cal.get("hardware")
    else:
        acc_offset = [0, 0, 0]
        gyro_offset = [0, 0, 0]
        mag_offset = [0, 0, 0]
        mag_scale = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        hardware_version = 1.0

    # Export option: set to True to save `timestamps` and `X_data` to CSV after the run
    EXPORT_STATES_ON_EXIT = False
    EXPORT_STATES_FILENAME = "eskf_states.csv"

    plotter = Plotter(
        state_labels=ESKF_STATE_LABELS,
        meas_labels=(ESKF_MEASUREMENT_LABELS if USE_MAGNETOMETER else ["pressure"]),
    )

    # Try to load a reference altitude CSV from the launch folder.
    # These are produced by a separate flight recorder on the rocket and contain
    # an "estPressureAlt" column that can be overlaid on the filter output.
    # Convention: the file is named after the base launch folder (without the
    # board-specific suffix like "_nc" or "_avab").
    ref_csv_candidates = sorted(launch_folder.glob("*.csv"))
    sensor_names = {"BMP581_data.csv", "ICM45686_data.csv", "MMC5983MA_data.csv", "ukf_states.csv"}
    for candidate in ref_csv_candidates:
        if candidate.name not in sensor_names:
            plotter.load_reference_altitude(candidate)
            break
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
    )
    context = ESKFContext(data_processor, plotter, hw_version=hardware_version, use_mag=USE_MAGNETOMETER)
    run_data_loop(context)
    # After the run ends, optionally export the collected UKF states/timestamps
    if EXPORT_STATES_ON_EXIT:
        out_path = launch_folder / EXPORT_STATES_FILENAME
        try:
            plotter.export_states_csv(out_path)
        except Exception as e:
            print(f"Failed to export states CSV: {e}")
    

def run_data_loop(context):
    while True:
        context.update()
        if context.shutdown_requested:
            break


if __name__ == "__main__":
    run()