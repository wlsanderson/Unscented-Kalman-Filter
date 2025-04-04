import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from UKF.constants import TIMESTAMP_COL_NAME, GRAVITY, LOG_HEADER_STATES

class Plotter:
    __slots__ = (
        "state_indices",
        "X_data",
        "P_data",
        "timestamps",
        "csv_data",
        "csv_time",
        "state_times",
    )

    def __init__(self, state_index: int | list, file_path: Path = None, min_r = None, max_r = None):
        """
        Plots the state variables from the UKF data and csv data if provided.

        :param state_index: index of state vector to plot, or list of 1 or 2 state vectors to plot
        :param file_path: Path of csv file to plot, ideally matching the csv used for UKF data
        :param min_r: the first row to start at in csv
        :param mar_r: last row to end at in csv
        """
        pio.renderers.default = "browser"

        if isinstance(state_index, int):
            self.state_indices = [state_index]
        elif isinstance(state_index, list) and (len(state_index) in [1,2]):
            self.state_indices = state_index
        else:
            raise ValueError("state_index must be an int or a list of 1 or 2 ints.")

        self.X_data: list = []
        self.P_data: list = []
        self.timestamps: list = [] # nanoseconds
        self.state_times: list = [] # nanoseconds

        # Initialize CSV data and timestamp dicts
        self.csv_data: dict = {}
        self.csv_time: dict = {}

        # If a CSV file path is provided, load and process the data.
        if file_path:
            headers = [TIMESTAMP_COL_NAME]
            for s in self.state_indices:
                if s in LOG_HEADER_STATES:
                    headers.append(LOG_HEADER_STATES[s])
                else:
                    raise ValueError(f"Unsupported state index {s} for CSV reading.")

            
            # Read CSV with desired columns
            df = pd.read_csv(file_path, usecols=headers)
            if max_r is not None:
                df = df.loc[:max_r]
            if min_r is not None:
                df = df.loc[min_r:]

            for s in self.state_indices:
                col_name = LOG_HEADER_STATES[s]
                meas_col = df[col_name]
            
                if s == 2:
                    meas_col *= -GRAVITY
                time_col = df[TIMESTAMP_COL_NAME]

                # Create a mask for rows where the measurement is not NaN
                mask = meas_col.notna()
                meas_array = meas_col[mask].to_numpy(dtype=np.float64)
                time_array = time_col[mask].to_numpy(dtype=np.float64)
                time_array = (time_array - time_array[0]) / 1e9
                self.csv_data[s] = meas_array
                self.csv_time[s] = time_array
    
    def start_plot(self):
        fig = go.Figure()

        # Plot CSV data if available
        for i, s in enumerate(self.state_indices):
            if s in self.csv_data and s in self.csv_time:
                if len(self.state_indices) == 1:
                    fig.add_trace(go.Scatter(
                        x = self.csv_time[s],
                        y = self.csv_data[s],
                        mode="lines",
                        name=f"CSV Data (col {s})"
                    ))
                else:
                    yaxis_name = "y" if i == 0 else "y2"
                    fig.add_trace(go.Scatter(
                        x=self.csv_time[s],
                        y=self.csv_data[s],
                        mode="lines",
                        name=f"CSV Data (col {s})",
                        yaxis=yaxis_name
                    ))
        
        # Plot simulated data
        if self.X_data and self.timestamps:
            timestamps = np.array(self.timestamps, dtype=np.float64)
            X_data = np.array(self.X_data, dtype=np.float64)
            timestamps = (timestamps - timestamps[0])/1e9

            for i, s in enumerate(self.state_indices):
                internal_trace = X_data[:, s]

                if len(self.state_indices) == 1:
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=internal_trace,
                        mode="lines",
                        name=f"Simulated X[{s}]"
                    ))
                else:
                    # if 2 indices, use 2 y axes
                    yaxis_name = "y" if i == 0 else "y2"
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=internal_trace,
                        mode="lines",
                        name=f"Simulated X[{s}]",
                        yaxis=yaxis_name,
                    ))
                    

        # Plot state change lines
        if len(self.state_times) > 1:
            st = np.array(self.state_times, dtype=np.float64)
            st = ((st - st[0])/1e9)[1:] # converting to seconds and dropping first point (first timestamp of csv)
            for x_coord in st:
                fig.add_vline(x=x_coord)

        # setup layout with multiple y-axes if needed
        layout = {
            "title": "State Variable vs Time",
            "xaxis": {"title": "Time (seconds)"},
            "template": "plotly_dark"
        }
        if len(self.state_indices) == 1:
            layout["yaxis"] = {"title": f"X[{self.state_indices[0]}]"}
        else:
            layout["yaxis"] = {"title": f"X[{self.state_indices[0]}]", "side": "left"}
            layout["yaxis2"] = {"title": f"X[{self.state_indices[1]}]", "overlaying": "y", "side": "right"}
        
        # Update figure layout
        fig.update_layout(**layout)
        fig.show()

