import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from UKF.constants import TIMESTAMP_COL_NAME, GRAVITY, LOG_HEADER_STATES
import quaternion
from scipy.spatial.transform import Rotation as R

# Define a helper to get a label for a state (either an int or a callable)
def get_label(s, i):
    if callable(s):
        return s.__name__ if hasattr(s, '__name__') else f"Derived[{i}]"
    else:
        if s in LOG_HEADER_STATES:
            return LOG_HEADER_STATES[s]
        else:
            return f"X[{s}]"

class Plotter:
    __slots__ = (
        "state_indices",
        "X_data",
        "X_data_pred",
        "P_data",
        "timestamps",
        "csv_data",
        "csv_time",
        "state_times",
    )

    def __init__(self, state_index: int | list, file_path: Path = None, min_r=None, max_r=None):
        pio.renderers.default = "browser"

        if isinstance(state_index, int) or callable(state_index):
            self.state_indices = [state_index]
        elif isinstance(state_index, list) and all(isinstance(s, (int, type(lambda: 0))) for s in state_index):
            self.state_indices = state_index
        else:
            raise ValueError("state_index must be an int, a function, or a list of ints/functions.")

        self.X_data = []
        self.X_data_pred = []
        self.P_data = []
        self.timestamps = []
        self.state_times = []
        self.csv_data = {}
        self.csv_time = {}

        if file_path:
            df = pd.read_csv(file_path)
            if max_r is not None:
                df = df.loc[:max_r]
            if min_r is not None:
                df = df.loc[min_r:]

            for s in self.state_indices:
                if isinstance(s, int):
                    col_name = LOG_HEADER_STATES[s]
                    if col_name not in df:
                        raise ValueError(f"Missing column '{col_name}' in CSV for state index {s}.")
                    meas_col = df[col_name]
                    time_col = df[TIMESTAMP_COL_NAME]
                    if s in range(2, 5):
                        meas_col *= -GRAVITY
                    mask = meas_col.notna()
                    meas_array = meas_col[mask].to_numpy(dtype=np.float64)
                    time_array = time_col[mask].to_numpy(dtype=np.float64)

                elif callable(s):
                    quat_cols = ["estOrientQuaternionW", "estOrientQuaternionX", "estOrientQuaternionY", "estOrientQuaternionZ"]
                    if not all(col in df for col in quat_cols):
                        raise ValueError("Quaternion columns missing from CSV for derived state.")
                    quat_df = df[quat_cols]
                    time_col = df[TIMESTAMP_COL_NAME]
                    mask = ~quat_df.isna().any(axis=1)
                    meas_array = quat_df[mask].to_numpy(dtype=np.float64)
                    time_array = time_col[mask].to_numpy(dtype=np.float64)

                else:
                    raise ValueError(f"Unsupported state entry: {s}")

                time_array = (time_array - time_array[0]) / 1e9
                self.csv_data[s] = meas_array
                self.csv_time[s] = time_array

    def start_plot(self):
        fig = go.Figure()

        for i, s in enumerate(self.state_indices):
            yaxis_name = "y" if i == 0 else "y2"
            label = get_label(s, i)
            if callable(s):
                csv_pitch = self.compute_pitch(self.csv_data[s])
                fig.add_trace(go.Scatter(
                    x=self.csv_time[s],
                    y=csv_pitch,
                    mode="lines",
                    name=f"CSV {label}",
                    yaxis=yaxis_name
                ))
            elif s in self.csv_data and s in self.csv_time:
                fig.add_trace(go.Scatter(
                    x=self.csv_time[s],
                    y=self.csv_data[s],
                    mode="lines",
                    name=f"CSV {label}",
                    yaxis=yaxis_name
                ))

        if self.X_data and self.timestamps:
            timestamps = np.array(self.timestamps, dtype=np.float64)
            X_data = np.array(self.X_data, dtype=np.float64)
            X_data_pred = np.array(self.X_data_pred, dtype=np.float64) if self.X_data_pred else None
            timestamps = (timestamps - timestamps[0]) / 1e9

            for i, s in enumerate(self.state_indices):
                yaxis_name = "y" if i == 0 else "y2"
                label = get_label(s, i)
                if callable(s):
                    internal_trace = s(X_data)
                    internal_trace_pred = s(X_data_pred) if X_data_pred is not None else None
                else:
                    internal_trace = X_data[:, s]
                    internal_trace_pred = X_data_pred[:, s] if X_data_pred is not None else None

                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=internal_trace,
                    mode="lines",
                    name=f"Simulated {label}",
                    yaxis=yaxis_name
                ))
                if internal_trace_pred is not None:
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=internal_trace_pred,
                        mode="markers",
                        name=f"Simulated {label} pred",
                        yaxis=yaxis_name
                    ))

        if len(self.state_times) > 1:
            st = np.array(self.state_times, dtype=np.float64)
            st = ((st - st[0]) / 1e9)[1:]
            for x_coord in st:
                fig.add_vline(x=x_coord)

        layout = {
            "title": "State Variable vs Time",
            "xaxis": {"title": "Time (seconds)"},
            "template": "plotly_dark"
        }
        if len(self.state_indices) == 1:
            layout["yaxis"] = {"title": f"{get_label(self.state_indices[0], 0)}"}
        else:
            layout["yaxis"] = {"title": f"{get_label(self.state_indices[0], 0)}", "side": "left"}
            layout["yaxis2"] = {"title": f"{get_label(self.state_indices[1], 1)}", "overlaying": "y", "side": "right"}

        fig.update_layout(**layout)
        fig.update_traces(marker=dict(size=3))
        fig.show()

    def compute_pitch(self, q):
        r = quaternion.from_float_array(q)
        euler = quaternion.as_euler_angles(r)
        pitch = euler[:, 1]
        return pitch * (180 / np.pi)


# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.io as pio
# from pathlib import Path
# from UKF.constants import TIMESTAMP_COL_NAME, GRAVITY, LOG_HEADER_STATES
# import quaternion


# class Plotter:
#     __slots__ = (
#         "state_indices",
#         "X_data",
#         "X_data_pred",
#         "P_data",
#         "timestamps",
#         "csv_data",
#         "csv_time",
#         "state_times",
#     )

#     def __init__(self, state_index: int | list, file_path: Path = None, min_r = None, max_r = None):
#         """
#         Plots the state variables from the UKF data and csv data if provided.

#         :param state_index: index of state vector to plot, or list of 1 or 2 state vectors to plot
#         :param file_path: Path of csv file to plot, ideally matching the csv used for UKF data
#         :param min_r: the first row to start at in csv
#         :param mar_r: last row to end at in csv
#         """
#         pio.renderers.default = "browser"

#         if isinstance(state_index, int) or callable(state_index):
#             self.state_indices = [state_index]
#         elif isinstance(state_index, list) and all(isinstance(s, (int, type(lambda:0))) for s in state_index):
#             self.state_indices = state_index
#         else:
#             raise ValueError("state_index must be an int, a function, or a list of ints/functions.")

#         self.X_data: list = []
#         self.X_data_pred: list = []
#         self.P_data: list = []
#         self.timestamps: list = [] # nanoseconds
#         self.state_times: list = [] # nanoseconds

#         # Initialize CSV data and timestamp dicts
#         self.csv_data: dict = {}
#         self.csv_time: dict = {}

#         # If a CSV file path is provided, load and process the data.
#         if file_path:
#             headers = [TIMESTAMP_COL_NAME]
#             for s in self.state_indices:
#                 if s in LOG_HEADER_STATES:
#                     headers.append(LOG_HEADER_STATES[s])
#                 elif callable(s):
#                     headers.append("estOrientQuaternionW")
#                     headers.append("estOrientQuaternionX")
#                     headers.append("estOrientQuaternionY")
#                     headers.append("estOrientQuaternionZ")
#                 else:
#                     raise ValueError(f"Unsupported state index {s} for CSV reading.")

            
#             # Read CSV with desired columns
#             df = pd.read_csv(file_path, usecols=headers)
#             if max_r is not None:
#                 df = df.loc[:max_r]
#             if min_r is not None:
#                 df = df.loc[min_r:]

#             for s in self.state_indices:
#                 if callable(s) :
#                     meas_col = df[headers[1:]]
#                 else:
#                     col_name = LOG_HEADER_STATES[s]
#                     meas_col = df[col_name]
            
#                 if s in range(2,5):
#                     meas_col *= -GRAVITY
#                 # if s == 2:
#                 #     meas_col *= -1
#                 time_col = df[TIMESTAMP_COL_NAME]

#                 # Create a mask for rows where the measurement is not NaN
#                 if isinstance(meas_col, pd.Series):
#                     mask = meas_col.notna()
#                     meas_array = meas_col[mask].to_numpy(dtype=np.float64)
#                     time_array = time_col[mask].to_numpy(dtype=np.float64)
#                 else:
#                     mask = ~meas_col.isna().any(axis=1)
#                     meas_array = meas_col[mask].to_numpy(dtype=np.float64)
#                     time_array = time_col[mask].to_numpy(dtype=np.float64)
#                 time_array = (time_array - time_array[0]) / 1e9
#                 self.csv_data[s] = meas_array
#                 self.csv_time[s] = time_array
    
#     def start_plot(self):
#         fig = go.Figure()

#         # Plot CSV data if available
#         for i, s in enumerate(self.state_indices):
#             if callable(s):
#                 fig.add_trace(go.Scatter(
#                     x = self.csv_time[s],
#                     y = self.compute_pitch(self.csv_data[s]),
#                     mode="lines",
#                     name="CSV Pitch"
#                 ))
#             elif s in self.csv_data and s in self.csv_time:
#                 if len(self.state_indices) == 1:
#                     fig.add_trace(go.Scatter(
#                         x = self.csv_time[s],
#                         y = self.csv_data[s],
#                         mode="lines",
#                         name=f"CSV Data (col {s})"
#                     ))
#                 else:
#                     yaxis_name = "y" if i == 0 else "y2"
#                     fig.add_trace(go.Scatter(
#                         x=self.csv_time[s],
#                         y=self.csv_data[s],
#                         mode="lines",
#                         name=f"CSV Data (col {s})",
#                         yaxis=yaxis_name
#                     ))
        
#         # Plot simulated data
#         if self.X_data and self.timestamps:
#             timestamps = np.array(self.timestamps, dtype=np.float64)
#             X_data = np.array(self.X_data, dtype=np.float64)
#             X_data_pred = np.array(self.X_data_pred, dtype=np.float64)
#             timestamps = (timestamps - timestamps[0])/1e9


#             for i, s in enumerate(self.state_indices):
#                 if callable(s):
#                     internal_trace = s(X_data)
#                     internal_trace_pred = s(X_data_pred)
#                     name = getattr(s, "__name__", f"Derived[{i}]")
#                 else:
#                     internal_trace = X_data[:, s]
#                     internal_trace_pred = X_data_pred[:, s]
#                     name = f"Simulated X[{s}]"

#                 if len(self.state_indices) == 1:
#                     fig.add_trace(go.Scatter(
#                         x=timestamps,
#                         y=internal_trace,
#                         mode="lines",
#                         name=name
#                     ))
#                     fig.add_trace(go.Scatter(
#                         x=timestamps,
#                         y=internal_trace_pred,
#                         name=(name + " pred"),
#                         mode="markers"
#                     ))

#                 else:
#                     # if 2 indices, use 2 y axes
#                     yaxis_name = "y" if i == 0 else "y2"
#                     fig.add_trace(go.Scatter(
#                         x=timestamps,
#                         y=internal_trace,
#                         mode="lines",
#                         name=name,
#                         yaxis=yaxis_name,
#                     ))
#                     fig.add_trace(go.Scatter(
#                         x=timestamps,
#                         y=internal_trace_pred,
#                         mode="markers",
#                         name=(name + " pred"),
#                         yaxis=yaxis_name,
#                     ))

#         # Plot state change lines
#         if len(self.state_times) > 1:
#             st = np.array(self.state_times, dtype=np.float64)
#             st = ((st - st[0])/1e9)[1:] # converting to seconds and dropping first point (first timestamp of csv)
#             for x_coord in st:
#                 fig.add_vline(x=x_coord)

#         # setup layout with multiple y-axes if needed
#         layout = {
#             "title": "State Variable vs Time",
#             "xaxis": {"title": "Time (seconds)"},
#             "template": "plotly_dark"
#         }
#         if len(self.state_indices) == 1:
#             layout["yaxis"] = {"title": f"X[{self.state_indices[0]}]"}
#         else:
#             layout["yaxis"] = {"title": f"X[{self.state_indices[0]}]", "side": "left"}
#             layout["yaxis2"] = {"title": f"X[{self.state_indices[1]}]", "overlaying": "y", "side": "right"}
        
#         # Update figure layout
#         fig.update_layout(**layout)
#         fig.update_traces(marker=dict(size=3))
#         fig.show()

#     def compute_pitch(self, q):
#         r = quaternion.from_float_array(q)
#         euler = quaternion.as_euler_angles(r)
#         pitch = euler[:, 1]  # second column is pitch
#         return pitch * (180/np.pi)

