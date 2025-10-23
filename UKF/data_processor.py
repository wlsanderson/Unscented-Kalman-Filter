import pandas as pd
from pathlib import Path
from UKF.constants import MEASUREMENT_FIELDS, TIMESTAMP_COL_NAME
import numpy as np


class DataProcessor:

    __slots__ = (
        "_df",
        "_headers",
        "_needed_measurements",
        "_iterator",
        "_iterator",
    )

    def __init__(self, launch_log: Path, min_r = None, max_r = None):
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
        if max_r is not None:
            self._df = self._df.loc[:max_r]
            
        if min_r is not None:
            self._df = self._df.loc[min_r - 1:] # so the initial values will not cause the first dt to be zero
        self._iterator = self._df.itertuples(index=False, name=None)
        # when reading rows, skip first row. This only happenes because the first row was reserved
        # for the initial values. Initial values cant make the first dt zero.
        next(self._iterator)


    def fetch(self):
        data: pd.Series | None = None
        try:
            data = next(self._iterator)
            return np.array(data)
        except StopIteration:
            print("eof")
            return None
    
    def get_initial_vals(self):
        initial_vals = np.full(len(MEASUREMENT_FIELDS) + 1, None, dtype=object)
        i = self._df.head(1).index[0]
        while any(val is None for val in initial_vals) and i < self._df.index[-1]:
            row = self._df.loc[i].values
            for col_index, val in enumerate(row):
                if initial_vals[col_index] is None and pd.notna(val):
                    initial_vals[col_index] = val
            i += 1
        return initial_vals
