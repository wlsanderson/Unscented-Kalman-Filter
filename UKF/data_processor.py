import pandas as pd
from pathlib import Path
from UKF.constants import MEASUREMENT_FIELDS, CONTROL_INPUT_FIELDS, TIMESTAMP_COL_NAME, MAG_CAL_OFFSET, MAG_CAL_SCALE_MATRIX
import numpy as np


class DataProcessor:

    __slots__ = (
        "_df",
        "_iterator",
        "dt",
        "measurements",
        "inputs",
        "_last_data",
        "_min_t",
    )

    def __init__(self, bmp_data: Path, imu_data: Path, mag_data: Path, min_t = None, max_t = None):
        bmp_df = self.get_sensor_df(bmp_data)
        imu_df = self.get_sensor_df(imu_data)
        mag_df = self.get_sensor_df(mag_data)
        mag_df = self.fix_mag_data(mag_df)

        self._df = pd.concat([bmp_df, imu_df, mag_df], ignore_index=True)
        self._df.sort_values(by=TIMESTAMP_COL_NAME, inplace=True, ignore_index=True)

        if max_t is not None:
            self._df = self._df.loc[self._df['timestamp'] < max_t]
        self._min_t = min_t
        if min_t is not None:
            self._df = self._df.loc[self._df['timestamp'] > min_t]
        self._iterator = self._df.itertuples(index=False, name=None)
        self._last_data = np.full(len(MEASUREMENT_FIELDS) + len(CONTROL_INPUT_FIELDS) + 1, None, dtype=object)


    def fetch(self):
        try:
            last_timestamp = self._last_data[0]
            
            new_data = self._last_data.copy()
            updated_flags = np.zeros(len(new_data), dtype=bool)

            # loop over iterator
            for row in self._iterator:
                timestamp = row[0]
                new_data[0] = timestamp

                # check which entries in this row are not NaN (skip timestamp)
                for i in range(1, len(row)):
                    val = row[i]
                    if val == val:  # isnan check
                        new_data[i] = val
                        updated_flags[i] = True

                # If all fields except timestamp have been updated, break early
                if np.all(updated_flags[1:]):
                    self.dt = timestamp - self._min_t if last_timestamp is None else timestamp - last_timestamp
                    
                    n_meas = len(MEASUREMENT_FIELDS)
                    n_inputs = len(CONTROL_INPUT_FIELDS)
                    self.measurements = np.array(new_data[1 : 1 + n_meas], dtype=np.float64)
                    self.inputs = np.array(new_data[1 + n_meas : 1 + n_meas + n_inputs])
                    mag_idx = slice(n_meas - 3, n_meas)
                    mag = self.measurements[mag_idx]
                    # calibrate
                    mag = (mag - MAG_CAL_OFFSET) @ MAG_CAL_SCALE_MATRIX
                    mag_norm = np.linalg.norm(mag)
                    self.measurements[mag_idx] =  mag / mag_norm
                    self._last_data = new_data
                    return True
            print("eof")
            return False

        except StopIteration:
            print("eof")
            return False
    
    
    def get_sensor_df(self, data):
        headers = pd.read_csv(data, nrows=0)
        needed_measurements = list((set(MEASUREMENT_FIELDS) | set(CONTROL_INPUT_FIELDS) | set([TIMESTAMP_COL_NAME])) & set(headers.columns))
        return pd.read_csv(data, usecols=needed_measurements)
    
    def fix_mag_data(self, data):
        df = data.copy()
        replace_idx = list(range(10, len(df), 11))
        cols_to_replace = df.columns[1:]
        df.loc[replace_idx, cols_to_replace] = df.loc[[i - 1 for i in replace_idx], cols_to_replace].values
        return df