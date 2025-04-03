import pandas as pd
from pathlib import Path
from UKF.constants import MEASUREMENT_FIELDS, TIMESTAMP_COL_NAME
import numpy as np
import numpy.typing as npt


class DataProcessor:

    __slots__ = (
        "_df",
        "_headers",
        "_needed_measurements",
        "_iterator",
        "_iterator",
    )

    def __init__(self, launch_log: Path, cutoff = None):
        self._headers: pd.DataFrame = pd.read_csv(launch_log, nrows=0)
        self._needed_measurements = list(
            (set(MEASUREMENT_FIELDS) | set([TIMESTAMP_COL_NAME])) & set(self._headers.columns)
        )
        self._df: pd.DataFrame = pd.read_csv(launch_log, usecols=self._needed_measurements)

        # puts dataframe in correct order
        field_order = MEASUREMENT_FIELDS.copy()
        field_order.insert(0, TIMESTAMP_COL_NAME)
        self._df = self._df[field_order]
        self._headers = self._headers[field_order]
        if cutoff is not None:
            self._df = self._df.head(cutoff)
            print(self._df.size)
        self._iterator = self._df.iterrows()

        

    def fetch(self):
        data: pd.Series | None = None
        try:
            _, data = next(self._iterator)
        except StopIteration:
            print("eof")
            return None

        return data
    
    def get_initial_vals(self):
        initial_vals = np.full(len(MEASUREMENT_FIELDS) + 1, None, dtype=object)
        i = 0
        while any(val is None for val in initial_vals) and i < len(self._df):
            row = self._df.loc[i].values
            for col_index, val in enumerate(row):
                if initial_vals[col_index] is None and pd.notna(val):
                    initial_vals[col_index] = val
            i += 1
        return initial_vals
