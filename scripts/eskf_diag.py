"""Quick diagnostic run for ESKF tuning — prints key state values at intervals."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UKF.eskf_context import ESKFContext
from UKF.data_processor import DataProcessor
from pathlib import Path
import numpy as np
import yaml

launch_folder = Path("launch_data/government_work_launch_1_nc")
launch_log = [
    launch_folder / "BMP581_data.csv",
    launch_folder / "ICM45686_data.csv",
    launch_folder / "MMC5983MA_data.csv",
]

min_t = 1273.42
max_t = 1297.4

cal_file = launch_folder / "calibration.yaml"
with open(cal_file, "r") as f:
    cal_root = yaml.safe_load(f) or {}
cal = cal_root.get("calibration", {})

data_processor = DataProcessor(
    bmp_data=launch_log[0],
    imu_data=launch_log[1],
    mag_data=launch_log[2],
    min_t=min_t,
    max_t=max_t,
    acc_cal_offset=cal.get("accel_offset", [0, 0, 0]),
    gyro_cal_offset=cal.get("gyro_offset", [0, 0, 0]),
    mag_cal_offset=cal.get("mag_offset", [0, 0, 0]),
    mag_cal_scale=cal.get("mag_scale", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
)

context = ESKFContext(data_processor, plotter=None)

step = 0
print(f"{'step':>6} {'time':>8} {'state':>10} {'alt':>8} {'vel_z':>8} {'abias_z':>9} {'gbias_z':>9} {'qw':>7} {'P_vel_z':>9} {'P_abias_z':>10} {'mahal':>8}")
print("-" * 120)
while True:
    context.update()
    if context.shutdown_requested:
        break
    step += 1
    x = context.eskf.x_nom
    P = context.eskf.P
    state_name = type(context._flight_state).__name__.replace("ESKF", "").replace("State", "")
    mahal = context.eskf.mahalanobis_dist if context.eskf.mahalanobis_dist is not None else 0
    if step % 200 == 0 or step <= 3 or (2300 <= step <= 2700 and step % 10 == 0):
        print(f"{step:6d} {context._timestamp:8.4f} {state_name:>10} {x[2]:8.2f} {x[5]:8.2f} {x[8]:9.5f} {x[11]:9.5f} {x[12]:7.4f} {P[5,5]:9.2e} {P[11,11]:10.2e} {mahal:8.1f}")

print(f"\nMax altitude: {context._max_altitude:.2f} m")
print(f"Max velocity: {context._max_velocity:.2f} m/s")
