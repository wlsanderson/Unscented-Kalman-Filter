import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.io as pio

class Plotter:

    __slots__ = (
        "X_data",
        "P_data",
        "timestamps",
    )

    def __init__(self):
        pio.renderers.default = "browser"
        self.X_data: list = []
        self.P_data: list = []
        self.timestamps: list = []

    def start_plot(self):

        # Convert lists to NumPy arrays for easier manipulation
        timestamps = np.array(self.timestamps, dtype=np.float64)
        X_data = np.array(self.X_data, dtype=np.float64)  # Shape: (N, 3)

        # Normalize timestamps
        timestamps = (timestamps - timestamps[0]) / 1e9

        # Extract first element of each X_data array
        X_first_component = X_data[:, 0]

        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=X_first_component, mode="lines", name="X[0]"))

        # Set layout details
        fig.update_layout(
            title="State Variable X[0] vs Time",
            xaxis_title="Time (seconds)",
            yaxis_title="X[0]",
            template="plotly_dark"
        )

        fig.show()

