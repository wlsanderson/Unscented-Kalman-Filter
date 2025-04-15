import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input
from pathlib import Path
from UKF.constants import TIMESTAMP_COL_NAME, LOG_HEADER_STATES, TIMESTAMP_UNITS, MEASUREMENT_FIELDS

class Plotter:
    __slots__ = (
        "X_data",
        "X_data_pred",
        "mahal",
        "timestamps",
        "timestamps_pred",
        "csv_data",
        "csv_time",
        "state_times",
        "z_error_score",
    )

    def __init__(self, file_path: Path, min_r=None, max_r=None):

        self.mahal = []
        self.z_error_score = []
        self.X_data = []
        self.X_data_pred = []
        self.timestamps_pred = []
        self.timestamps = []
        self.state_times = []
        self.csv_data = {}
        self.csv_time = {}

        df = pd.read_csv(file_path)
        # limit data to only what was specified
        if max_r is not None:
            df = df.loc[:max_r]
        if min_r is not None:
            df = df.loc[min_r:]

        # this is assuming that the state index is in the same order as the associated
        # log header states
        for i in LOG_HEADER_STATES:
            col_name = LOG_HEADER_STATES[i]
            if col_name not in df:
                raise ValueError(f"Missing column '{col_name}' in CSV for state index {i}.")
            meas_col = df[col_name]
            time_col = df[TIMESTAMP_COL_NAME]
            mask = meas_col.notna()
            meas_array = meas_col[mask].to_numpy(dtype=np.float64)
            time_array = time_col[mask].to_numpy(dtype=np.float64)
            time_array = (time_array - time_array[0]) / TIMESTAMP_UNITS
            self.csv_data[i] = meas_array
            self.csv_time[i] = time_array

    def start_plot(self):
        timestamps = np.array(self.timestamps, dtype=np.float64)
        timestamps_pred = np.array(self.timestamps_pred, dtype=np.float64)
        timestamps = (timestamps - timestamps[0]) / TIMESTAMP_UNITS
        timestamps_pred = (timestamps_pred - timestamps_pred[0]) / TIMESTAMP_UNITS
        X_data = np.array(self.X_data, dtype=np.float64)
        X_data_pred = np.array(self.X_data_pred, dtype=np.float64) if self.X_data_pred else None
        mahal = np.array(self.mahal, dtype=np.float64) if self.mahal else None
        z_error_score = np.array(self.z_error_score, dtype=np.float64) if self.z_error_score else None

        app = Dash(__name__)
        fig = go.Figure()

        for s in LOG_HEADER_STATES:
            label = LOG_HEADER_STATES[s]
            fig.add_trace(go.Scatter(x=self.csv_time.get(s, []), y=self.csv_data.get(s, []), name=f"CSV {label}"))
            fig.add_trace(go.Scatter(x=timestamps, y=X_data[:, s] if len(X_data) else [], name=f"UKF {label}"))
            if X_data_pred is not None:
                fig.add_trace(go.Scatter(x=timestamps_pred, y=X_data_pred[:, s], name=f"UKF {label} pred", mode="markers"))

        if mahal is not None:
            fig.add_trace(go.Scatter(x=timestamps, y=mahal, name="Mahalanobis Distance"))
        if z_error_score is not None:
            for i, label in enumerate(MEASUREMENT_FIELDS):
                fig.add_trace(go.Scatter(x=timestamps, y=z_error_score[:, i], name=f"Z Error Score: {label}"))

        app.layout = html.Div([
            html.Div([
                dcc.Checklist(
                    id="state_selector",
                    options=[{"label": LOG_HEADER_STATES[s], "value": s} for s in LOG_HEADER_STATES],
                    value=[list(LOG_HEADER_STATES.keys())[0]],
                    labelStyle={"display": "block", "color": "white"},
                    style={"margin-bottom": "20px"}
                ),
                dcc.Checklist(
                    id="mahal_toggle",
                    options=[{"label": "Mahalanobis Distance", "value": "mahal"}] +
                            [{"label": f"Z Error Score: {label}", "value": f"z_{label}"} for label in MEASUREMENT_FIELDS],
                    value=[],
                    labelStyle={"display": "block", "color": "white"}
                ),
            ], style={
                "width": "15%",
                "height": "100vh",
                "overflowY": "auto",
                "padding": "10px",
                "box-sizing": "border-box",
                "float": "left",
                "background-color": "#111"
            }),

            html.Div([
                dcc.Graph(id="ukf_plot", figure=fig, style={
                    "height": "100vh",
                    "width": "100%",
                })
            ], style={
                "width": "85%",
                "height": "100vh",
                "float": "right",
            })
        ], style={
            "margin": "0",
            "padding": "0",
            "height": "100vh",
            "overflow": "hidden"
        })

        @app.callback(
            Output("ukf_plot", "figure"),
            Input("state_selector", "value"),
            Input("mahal_toggle", "value")
        )
        def update_plot(selected_states, toggles):
            new_fig = go.Figure()
            for s in selected_states:
                label = LOG_HEADER_STATES[s]
                new_fig.add_trace(go.Scatter(x=self.csv_time.get(s, []), y=self.csv_data.get(s, []), name=f"CSV {label}"))
                new_fig.add_trace(go.Scatter(x=timestamps, y=X_data[:, s] if len(X_data) else [], name=f"UKF {label}"))
                if X_data_pred is not None:
                    new_fig.add_trace(go.Scatter(x=timestamps_pred, y=X_data_pred[:, s], name=f"UKF {label} pred", mode="markers"))

            if mahal is not None and "mahal" in toggles:
                new_fig.add_trace(go.Scatter(x=timestamps, y=mahal, name="Mahalanobis Distance"))

            if z_error_score is not None:
                for i, label in enumerate(MEASUREMENT_FIELDS):
                    if f"z_{label}" in toggles:
                        new_fig.add_trace(go.Scatter(x=timestamps, y=z_error_score[:, i], name=f"Z Error Score: {label}"))

            new_fig.update_layout(
                title="UKF State Comparison (Dash)",
                xaxis_title="Time (seconds)",
                yaxis_title="Measurement",
                template="plotly_dark"
            )
            new_fig.update_traces(marker=dict(size=3))
            return new_fig

        app.run(debug=False, use_reloader=False)
