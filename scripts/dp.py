from UKF.data_processor import DataProcessor
from pathlib import Path
import numpy as np

launch_log = np.array([Path("launch_data/pressure_sensor_data.csv"), Path("launch_data/imu_data.csv"), Path("launch_data/magnetometer_data.csv")])
min_t = 0
max_t = 5
data_processor = DataProcessor(bmp_data = launch_log[0], imu_data = launch_log[1], mag_data = launch_log[2], min_t=min_t, max_t=max_t)
data_processor.fetch()
