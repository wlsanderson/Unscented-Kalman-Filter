import numpy as np
# note: pandas is imported locally in export_states_csv to avoid a global dependency
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input
from pathlib import Path
from UKF.constants import TIMESTAMP_UNITS
import socket


# ESKF-specific labels (matching reduced 6-state ordering)
ESKF_STATE_LABELS = [
    "POS_Z", "VEL_Z",
    "QUAT_W", "QUAT_X", "QUAT_Y", "QUAT_Z",
]
ESKF_MEASUREMENT_LABELS = ["pressure", "mag_x", "mag_y", "mag_z"]


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
        "pressure_alt",
        "pressure_nis_ref",
        "_ref_alt_time",
        "_ref_alt",
        "_state_labels",
        "_meas_labels",
    )

    def __init__(self, state_labels, meas_labels):
        """
        Parameters
        ----------
        state_labels : list[str]
            Labels for each nominal state index..
        meas_labels : list[str]
            Labels for each measurement index.
        """
        self.mahal = []
        self.z_error_score = []
        self.X_data = []
        self.X_data_pred = []
        self.timestamps_pred = []
        self.timestamps = []
        self.state_times = []
        self.uncerts = []
        self.pressure_alt = []
        self.pressure_nis_ref = []
        self._ref_alt_time = None   # np.ndarray | None – seconds relative to reference launch
        self._ref_alt = None        # np.ndarray | None – zeroed pressure altitude
        self._state_labels = list(state_labels)
        self._meas_labels = list(meas_labels)

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
        self.pressure_alt.clear()
        self.pressure_nis_ref.clear()

    def load_reference_altitude(self, csv_path: Path, accel_threshold: float = 50.0) -> None:
        """Load estPressureAlt from a reference flight recorder CSV.

        The CSV must contain ``timestamp`` (nanoseconds since epoch),
        ``estPressureAlt``, and ``vertical_acceleration`` columns.

        Processing steps
        ----------------
        1. Drop rows where ``estPressureAlt`` is NaN (alternating-rate data).
        2. Detect the launch moment as the first row where
           ``|vertical_acceleration| > accel_threshold``.
        3. Zero the altitude by subtracting the standby mean (all rows before launch).
        4. Convert timestamps to seconds relative to the detected launch moment.

        Alignment with the filter timeline is done later in ``start_plot()``
        using ``state_times[0]``.
        """
        import pandas as _pd

        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"Plotter: reference CSV not found - {csv_path}")
            return

        df = _pd.read_csv(csv_path, low_memory=False)

        required = {"timestamp", "estPressureAlt", "vertical_acceleration"}
        if not required.issubset(df.columns):
            print(f"Plotter: reference CSV missing columns {required - set(df.columns)}")
            return

        # Keep only rows with valid estPressureAlt (baro-rate rows)
        df = df.dropna(subset=["estPressureAlt"]).reset_index(drop=True)

        # Detect launch: first row where |vertical_acceleration| exceeds threshold.
        # vertical_acceleration may have NaN; forward-fill for the comparison.
        va = df["vertical_acceleration"].ffill().fillna(0.0)
        launch_mask = va.abs() > accel_threshold
        if not launch_mask.any():
            print("Plotter: could not detect launch in reference CSV")
            return
        launch_idx = launch_mask.idxmax()

        # Standby mean for zeroing (all rows before launch)
        standby_alt = df.loc[:launch_idx - 1, "estPressureAlt"]
        alt_offset = standby_alt.mean() if len(standby_alt) > 0 else 0.0

        # Build arrays: time in seconds relative to launch, zeroed altitude
        t_ns = df["timestamp"].to_numpy(dtype=np.float64)
        launch_t_ns = t_ns[launch_idx]
        self._ref_alt_time = (t_ns - launch_t_ns) / 1e9          # seconds since launch
        self._ref_alt = df["estPressureAlt"].to_numpy(dtype=np.float64) - alt_offset

        print(
            f"Plotter: loaded {len(self._ref_alt)} reference altitude samples "
            f"(launch idx {launch_idx}, standby offset {alt_offset:.2f} m)"
        )

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

        # Build column names using stored labels
        cols = ["timestamp"]
        if X.shape[1] <= len(self._state_labels):
            state_names = self._state_labels[:X.shape[1]]
        else:
            state_names = self._state_labels + [f"state_{i}" for i in range(len(self._state_labels), X.shape[1])]
        cols.extend(state_names)

        df = _pd.DataFrame(np.hstack([ts.reshape(-1, 1), X]), columns=cols)
        df.to_csv(path, index=False)
        print(f"Plotter: exported {X.shape[0]} state rows to {path}")


    def start_plot(self):
        timestamps_raw = np.array(self.timestamps, dtype=np.float64)
        timestamps_pred_raw = np.array(self.timestamps_pred, dtype=np.float64)

        # Zero the timeline at the motor-burn transition so t=0 corresponds
        # to launch.  state_times[0] is Standby init (t≈0); state_times[1] is
        # the Standby -> MotorBurn transition.  Fall back to the first sample
        # if no transition was recorded.
        if len(self.state_times) >= 2:
            t_origin = self.state_times[1]
        else:
            t_origin = timestamps_raw[0]

        timestamps = (timestamps_raw - t_origin) / TIMESTAMP_UNITS
        timestamps_pred = (timestamps_pred_raw - t_origin) / TIMESTAMP_UNITS

        X_data = np.array(self.X_data, dtype=np.float64)
        X_data_pred = np.array(self.X_data_pred, dtype=np.float64) if self.X_data_pred else None
        X_uncerts = np.array(self.uncerts, dtype=np.float64) if self.uncerts else None
        mahal = np.array(self.mahal, dtype=np.float64) if self.mahal else None
        z_error_score = np.array(self.z_error_score, dtype=np.float64) if self.z_error_score else None
        pressure_alt = np.array(self.pressure_alt, dtype=np.float64) if self.pressure_alt else None
        pressure_nis_ref = np.array(self.pressure_nis_ref, dtype=np.float64) if self.pressure_nis_ref else None

        # Reference altitude is already zeroed at launch (t=0), so it aligns
        # directly now that the filter timeline is also zeroed at motor burn.
        ref_alt_time = None
        ref_alt = None
        if self._ref_alt_time is not None and self._ref_alt is not None:
            ref_alt_time = self._ref_alt_time
            ref_alt = self._ref_alt

        n_states = X_data.shape[1] if X_data.ndim == 2 else len(self._state_labels)
        state_labels = self._state_labels
        meas_labels = self._meas_labels
        filter_name = "ESKF"

        if X_uncerts is not None:
            n_cov = X_uncerts.shape[1] if X_uncerts.ndim == 2 else 0
            X_pos_sigma = np.zeros(X_data.shape)
            X_neg_sigma = np.zeros(X_data.shape)
            # For the vector portion: columns that have a 1:1 match in cov
            # UKF: 12 vector + 3 quat tangent = 15 cov; ESKF: 15 cov
            n_vec = n_states - 4  # everything except quaternion
            n_quat_cov = n_cov - n_vec  # should be 3
            if n_vec <= n_cov:
                X_pos_sigma[:, :n_vec] = X_data[:, :n_vec] + np.sqrt(np.abs(X_uncerts[:, :n_vec]))
                X_neg_sigma[:, :n_vec] = X_data[:, :n_vec] - np.sqrt(np.abs(X_uncerts[:, :n_vec]))
            if n_quat_cov > 0:
                X_delta_quat_sigma = np.sum(X_uncerts[:, n_vec:n_vec+n_quat_cov], axis=1)
                X_pos_sigma[:, -4:] = X_data[:, -4:] + np.sqrt(np.abs(X_delta_quat_sigma))[:, np.newaxis]
                X_neg_sigma[:, -4:] = X_data[:, -4:] - np.sqrt(np.abs(X_delta_quat_sigma))[:, np.newaxis]
        

        app = Dash(__name__)
        fig = go.Figure()
        # prefer scientific notation for y-axis (consistent across updates)
        fig.update_layout(template="plotly_dark", yaxis=dict(tickformat=".3e"))

        # Create traces for every state index (0..STATE_DIM-1). If a CSV counterpart exists
        # we will plot it alongside; otherwise we just plot the UKF state.
        for s in range(n_states):
            label = state_labels[s] if s < len(state_labels) else f"state_{s}"


            # state trace
            ukf_y = X_data[:, s] if len(X_data) else []
            fig.add_trace(go.Scatter(x=timestamps, y=ukf_y, name=f"{filter_name} {label}"))

            # Prediction points (if available)
            if X_data_pred is not None:
                pred_y = X_data_pred[:, s] if X_data_pred.size else []
                fig.add_trace(go.Scatter(x=timestamps_pred, y=pred_y, name=f"{filter_name} {label} pred", mode="markers"))

            if X_uncerts is not None:
                fig.add_trace(go.Scatter(x=timestamps_pred, y=X_pos_sigma[:,s], name=f"{filter_name} {label} positive sigma"))
                fig.add_trace(go.Scatter(x=timestamps_pred, y=X_neg_sigma[:,s], name=f"{filter_name} {label} negative sigma"))

        if mahal is not None:
            fig.add_trace(go.Scatter(x=timestamps, y=mahal, name="Mahalanobis Distance"))
        if z_error_score is not None:
            for i in range(z_error_score.shape[1]):
                ml = meas_labels[i] if i < len(meas_labels) else f"meas_{i}"
                fig.add_trace(go.Scatter(x=timestamps, y=z_error_score[:, i], name=f"Z Error Score: {ml}"))

        app.layout = html.Div([
            html.Div([
                dcc.Checklist(
                    id="state_selector",
                    options=[{"label": (state_labels[s] if s < len(state_labels) else f"state_{s}"), "value": s} for s in range(n_states)],
                    value=[0],
                    labelStyle={"display": "block", "color": "white"},
                    style={"margin-bottom": "20px"}
                ),
                dcc.Checklist(
                    id="mahal_toggle",
            options=[{"label": "Mahalanobis Distance", "value": "mahal"},
                     {"label": "Pressure Altitude", "value": "pressure_alt"},
                     {"label": "Reference Altitude", "value": "ref_alt"},
                     {"label": "Pressure NIS Ref", "value": "nis_ref"}] +
                [{"label": f"Z Error Score: {meas_labels[i] if i < len(meas_labels) else f'meas_{i}'}", "value": f"z_{i}"} for i in range(len(meas_labels))],
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
                label = state_labels[s] if s < len(state_labels) else f"state_{s}"

                new_fig.add_trace(go.Scatter(x=timestamps, y=X_data[:, s] if len(X_data) else [], name=f"{filter_name} {label}"))
                if X_data_pred is not None:
                    new_fig.add_trace(go.Scatter(x=timestamps_pred, y=X_data_pred[:, s], name=f"{filter_name} {label} pred", mode="markers"))
                if X_uncerts is not None:
                    new_fig.add_trace(go.Scatter(x=timestamps_pred, y=X_pos_sigma[:,s], name=f"{filter_name} {label} positive sigma"))
                    new_fig.add_trace(go.Scatter(x=timestamps_pred, y=X_neg_sigma[:,s], name=f"{filter_name} {label} negative sigma"))
                        
            if mahal is not None and "mahal" in toggles:
                new_fig.add_trace(go.Scatter(x=timestamps, y=mahal, name="Mahalanobis Distance"))

            if pressure_alt is not None and "pressure_alt" in toggles:
                new_fig.add_trace(go.Scatter(x=timestamps, y=pressure_alt, name="Pressure Altitude"))

            if ref_alt is not None and "ref_alt" in toggles:
                new_fig.add_trace(go.Scatter(x=ref_alt_time, y=ref_alt, name="Reference Altitude"))

            if pressure_nis_ref is not None and "nis_ref" in toggles:
                new_fig.add_trace(go.Scatter(x=timestamps, y=pressure_nis_ref, name="Pressure NIS Ref"))

            if z_error_score is not None:
                n_meas = z_error_score.shape[1]
                for i in range(n_meas):
                    ml = meas_labels[i] if i < len(meas_labels) else f"meas_{i}"
                    if f"z_{i}" in toggles:
                        new_fig.add_trace(go.Scatter(x=timestamps, y=z_error_score[:, i], name=f"Z Error Score: {ml}"))

            new_fig.update_layout(
                title=f"{filter_name} State Comparison (Dash)",
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
