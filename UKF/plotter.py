import numpy as np
# note: pandas is imported locally in export_states_csv to avoid a global dependency
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input
from pathlib import Path
from UKF.constants import (
    TIMESTAMP_UNITS,
    MEASUREMENT_FIELDS,
    STATE_DIM,
)
from UKF.constants import States
import socket


def _find_free_port() -> int:
    """Return an available TCP port on localhost.

    This is used so the Dash server created by the plotter doesn't accidentally
    collide with a previously-running server from another dataset/run.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

class Plotter:
    __slots__ = (
        "X_data",
        "X_data_pred",
        "mahal",
        "timestamps",
        "timestamps_pred",
        "state_times",
        "z_error_score",
        "uncerts",
    )

    def __init__(self):

        self.mahal = []
        self.z_error_score = []
        self.X_data = []
        self.X_data_pred = []
        self.timestamps_pred = []
        self.timestamps = []
        self.state_times = []
        self.uncerts = []

    def clear_history(self) -> None:
        """Clear all stored time-series data in the plotter.

        Call this before starting a new run if you reuse the same Plotter
        instance across multiple datasets in the same Python process.
        """
        self.mahal.clear()
        self.z_error_score.clear()
        self.X_data.clear()
        self.X_data_pred.clear()
        self.timestamps_pred.clear()
        self.timestamps.clear()
        self.state_times.clear()
        self.uncerts.clear()

    def export_states_csv(self, path: Path) -> None:
        """Export stored timestamps and state vectors to a CSV file.

        The CSV will have a `timestamp` column followed by one column per state.
        State column names use the `States` enum when available, otherwise
        `state_0`, `state_1`, ...
        """
        import pandas as _pd

        if not self.X_data or not self.timestamps:
            print("Plotter: no state data available to export")
            return

        ts = np.array(self.timestamps, dtype=np.float64)
        X = np.array(self.X_data, dtype=np.float64)

        # Ensure dimensions match
        if X.shape[0] != ts.shape[0]:
            # If they differ, try to trim to the shortest length
            n = min(X.shape[0], ts.shape[0])
            ts = ts[:n]
            X = X[:n, :]

        # Build column names using States enum where possible
        cols = ["timestamp"]
        try:
            state_names = [States(i).name for i in range(X.shape[1])]
        except Exception:
            state_names = [f"state_{i}" for i in range(X.shape[1])]
        cols.extend(state_names)

        df = _pd.DataFrame(np.hstack([ts.reshape(-1, 1), X]), columns=cols)
        df.to_csv(path, index=False)
        print(f"Plotter: exported {X.shape[0]} state rows to {path}")


    def start_plot(self):
        timestamps = np.array(self.timestamps, dtype=np.float64)
        timestamps_pred = np.array(self.timestamps_pred, dtype=np.float64)
        timestamps = (timestamps - timestamps[0]) / TIMESTAMP_UNITS
        timestamps_pred = (timestamps_pred - timestamps_pred[0]) / TIMESTAMP_UNITS
        X_data = np.array(self.X_data, dtype=np.float64)
        X_data_pred = np.array(self.X_data_pred, dtype=np.float64) if self.X_data_pred else None
        X_uncerts = np.array(self.uncerts, dtype=np.float64) if self.uncerts else None
        mahal = np.array(self.mahal, dtype=np.float64) if self.mahal else None
        z_error_score = np.array(self.z_error_score, dtype=np.float64) if self.z_error_score else None


        if X_uncerts is not None:
            X_pos_sigma = np.zeros(X_data.shape)
            X_neg_sigma = np.zeros(X_data.shape)
            X_pos_sigma[:, :-4] = X_data[:, :-4] + np.square(X_uncerts[:, :-3])
            X_neg_sigma[:, :-4] = X_data[:, :-4] - np.square(X_uncerts[:, :-3])
            X_delta_quat_sigma = np.sum(X_uncerts[:, -3:], axis=1)
            X_pos_sigma[:, -4:] = np.add(X_data[:, -4:], np.square(X_delta_quat_sigma)[:, np.newaxis])
            X_neg_sigma[:, -4:] = np.subtract(X_data[:, -4:], np.square(X_delta_quat_sigma)[:, np.newaxis])
        

        app = Dash(__name__)
        fig = go.Figure()
        # prefer scientific notation for y-axis (consistent across updates)
        fig.update_layout(template="plotly_dark", yaxis=dict(tickformat=".3e"))

        # Create traces for every state index (0..STATE_DIM-1). If a CSV counterpart exists
        # we will plot it alongside; otherwise we just plot the UKF state.
        for s in range(STATE_DIM):
            # use a friendly label where possible
            try:
                label = States(s).name
            except Exception:
                label = f"state_{s}"


            # UKF state trace
            ukf_y = X_data[:, s] if len(X_data) else []
            fig.add_trace(go.Scatter(x=timestamps, y=ukf_y, name=f"UKF {label}"))

            # Prediction points (if available)
            if X_data_pred is not None:
                pred_y = X_data_pred[:, s] if X_data_pred.size else []
                fig.add_trace(go.Scatter(x=timestamps_pred, y=pred_y, name=f"UKF {label} pred", mode="markers"))

            if X_uncerts is not None:
                fig.add_trace(go.Scatter(x=timestamps_pred, y=X_pos_sigma[:,s], name=f"UKF {label} positive sigma"))
                fig.add_trace(go.Scatter(x=timestamps_pred, y=X_neg_sigma[:,s], name=f"UKF {label} negative sigma"))

        if mahal is not None:
            fig.add_trace(go.Scatter(x=timestamps, y=mahal, name="Mahalanobis Distance"))
        if z_error_score is not None:
            for i, label in enumerate(MEASUREMENT_FIELDS):
                fig.add_trace(go.Scatter(x=timestamps, y=z_error_score[:, i], name=f"Z Error Score: {label}"))

        app.layout = html.Div([
            html.Div([
                dcc.Checklist(
                    id="state_selector",
                    options=[{"label": States(s).name, "value": s} for s in range(STATE_DIM)],
                    value=[0],
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
                s = int(s)
                try:
                    label = States(s).name
                except Exception:
                    label = f"state_{s}"



                new_fig.add_trace(go.Scatter(x=timestamps, y=X_data[:, s] if len(X_data) else [], name=f"UKF {label}"))
                if X_data_pred is not None:
                    new_fig.add_trace(go.Scatter(x=timestamps_pred, y=X_data_pred[:, s], name=f"UKF {label} pred", mode="markers"))
                if X_uncerts is not None:
                    new_fig.add_trace(go.Scatter(x=timestamps_pred, y=X_pos_sigma[:,s], name=f"UKF {label} positive sigma"))
                    new_fig.add_trace(go.Scatter(x=timestamps_pred, y=X_neg_sigma[:,s], name=f"UKF {label} negative sigma"))
                        
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
                template="plotly_dark",
                yaxis=dict(tickformat=".3e")
            )
            new_fig.update_traces(marker=dict(size=3))
            return new_fig

        # Run Dash app on a free ephemeral port to avoid accidentally connecting to
        # a previously-running server that may be serving a different dataset.
        port = _find_free_port()
        print(f"Starting Dash server at http://127.0.0.1:{port} (open in browser)")
        app.run(debug=False, use_reloader=False, host="127.0.0.1", port=port)
