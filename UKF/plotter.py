import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from UKF.constants import TIMESTAMP_COL_NAME, GRAVITY

class Plotter:
    __slots__ = (
        "state_index",
        "X_data",
        "P_data",
        "timestamps",
        "csv_data",
        "csv_time",
    )

    def __init__(self, state_index: int, file_path: str = None, minrow = None, maxrow = None):
        # Set renderer to open in a browser (WSL-friendly)
        pio.renderers.default = "browser"

        self.state_index = state_index
        self.X_data: list = []  # Expected to be a list of 3-element NumPy arrays
        self.P_data: list = []
        self.timestamps: list = []  # Expected to be a list of time stamps (as numbers)

        # Initialize CSV-related attributes
        self.csv_data = None
        self.csv_time = None

        # If a CSV file path is provided, load and process the data.
        if file_path:
            headers = [TIMESTAMP_COL_NAME]
            match state_index:
                case 0:
                    headers.append("current_altitude")
                case 1:
                    headers.append("vertical_velocity")
                case 2:
                    headers.append("scaledAccelZ")
            
            # Read CSV without header (adjust header argument if you have one)
            df = pd.read_csv(file_path, usecols=headers)
            if maxrow is not None:
                df = df.loc[:maxrow]
            
            if minrow is not None:
                df = df.loc[minrow:]
            # Extract measurement column (state_index) and time column (index 10)
            meas_col = df.iloc[:, 1]
            if state_index == 2:
                meas_col *= -GRAVITY
            time_col = df.iloc[:, 0]

            # Create a mask for rows where the measurement is not NaN
            mask = meas_col.notna()
            self.csv_data = meas_col[mask].to_numpy(dtype=np.float64)
            csv_time = time_col[mask].to_numpy(dtype=np.float64)
            self.csv_time = (csv_time - csv_time[0])/1e9
    
    def start_plot(self):
        # Check if there is internal simulation data to plot
        if not self.X_data or not self.timestamps:
            print("No internal data to plot.")
        else:
            # Convert lists to NumPy arrays for internal data
            timestamps = np.array(self.timestamps, dtype=np.float64)
            X_data = np.array(self.X_data, dtype=np.float64)  # Expected shape: (N, 3)

            # Normalize internal timestamps (subtract initial value, convert from nanoseconds to seconds)
            timestamps = (timestamps - timestamps[0]) / 1e9

            # Extract the state column based on state_index
            internal_trace = X_data[:, self.state_index]

        # Create Plotly figure
        fig = go.Figure()

        # Plot internal data if available
        if not self.X_data or not self.timestamps:
            pass
        else:
            fig.add_trace(go.Scatter(
                x=timestamps, 
                y=internal_trace, 
                mode="lines", 
                name=f"Simulated X[{self.state_index}]"
            ))

        # Plot CSV data if available
        if self.csv_data is not None and self.csv_time is not None:
            fig.add_trace(go.Scatter(
                x=self.csv_time, 
                y=self.csv_data, 
                mode="lines", 
                name=f"CSV Data (col {self.state_index})"
            ))

        # Update figure layout
        fig.update_layout(
            title="State Variable vs Time",
            xaxis_title="Time (seconds)",
            yaxis_title=f"X[{self.state_index}]",
            template="plotly_dark"
        )

        # Show the figure
        fig.show()

