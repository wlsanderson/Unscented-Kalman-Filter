"""Quick test of both ESKF modes."""
import numpy as np
import yaml
from pathlib import Path
from UKF.plotter import Plotter, ESKF_STATE_LABELS, ESKF_MEASUREMENT_LABELS
from UKF.data_processor import DataProcessor
from UKF.eskf_context import ESKFContext

Plotter.start_plot = lambda self: None

launch_folder = Path("launch_data/government_work_launch_1_avab")
launch_log = [
    launch_folder / "BMP581_data.csv",
    launch_folder / "ICM45686_data.csv",
    launch_folder / "MMC5983MA_data.csv",
]
with open(launch_folder / "calibration.yaml") as f:
    cal = yaml.safe_load(f).get("calibration", {})

kwargs = dict(
    acc_cal_offset=cal.get("accel_offset", [0, 0, 0]),
    gyro_cal_offset=cal.get("gyro_offset", [0, 0, 0]),
    mag_cal_offset=cal.get("mag_offset", [0, 0, 0]),
    mag_cal_scale=cal.get("mag_scale", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
)

for mode, use_sm in [("STATE MACHINE", True), ("CONSTANT DYNAMICS", False)]:
    print(f"\n=== {mode} ===")
    plotter = Plotter(
        state_labels=ESKF_STATE_LABELS,
        meas_labels=ESKF_MEASUREMENT_LABELS,
        filter_name="ESKF",
    )
    dp = DataProcessor(
        bmp_data=launch_log[0], imu_data=launch_log[1], mag_data=launch_log[2],
        min_t=902, max_t=926, **kwargs,
    )
    ctx = ESKFContext(dp, plotter, use_state_machine=use_sm)
    while not ctx.shutdown_requested:
        ctx.update()

    X = np.array(plotter.X_data)
    print(f"  Max velocity: {X[:, 5].max():.2f} m/s")
    print(f"  Max altitude: {X[:, 2].max():.2f} m")
    print(f"  State times ({len(plotter.state_times)}): "
          + ", ".join(f"{t:.2f}" for t in plotter.state_times))
