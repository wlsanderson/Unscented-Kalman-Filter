from UKF.ukf import UKF
from UKF.sigma_points import SigmaPoints
from UKF.data_processor import DataProcessor
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_ESTIMATE, INITIAL_STATE_COV, TIMESTAMP_UNITS
from UKF.ukf_functions import measurement_function
from UKF.plotter import Plotter
from UKF.state import State, StandbyState
import numpy as np
import numpy.typing as npt

class Context:

    __slots__ = (
        "ukf",
        "data_processor",
        "shutdown_requested",
        "_last",
        "_plotter",
        "_last",
        "_flight_state",
        "_dt",
        "measurement",
    )

    def __init__(self, data_processor: DataProcessor, plotter: Plotter | None = None):
        sigma_points = SigmaPoints(
            # despite the state vector being 10 dimensions, the variance matrices live in 9D
            # due to quaternion noise being represented as a 3x3 matrix, not 4x4.
            n = np.size(INITIAL_STATE_COV, 0),
            alpha = ALPHA,
            beta = BETA,
            kappa = KAPPA,
        )
        self.ukf = UKF(
            dim_x = STATE_DIM,
            dim_z = MEASUREMENT_DIM,
            points = sigma_points,
        )
        self.data_processor: DataProcessor = data_processor
        self._plotter: Plotter = plotter
        self._last: npt.NDArray[np.float64] = data_processor.get_initial_vals()
        self._flight_state: State = StandbyState(self)
        self.shutdown_requested: bool = False
        self._dt: np.float64 = 0.0


        self.initialize_filter_settings()
        
    def initialize_filter_settings(self):
        self.ukf.X = INITIAL_STATE_ESTIMATE.copy()
        self.ukf.P = INITIAL_STATE_COV.copy()
        self.ukf.H = measurement_function

    def update(self):
        data = self.data_processor.fetch()
        if data is None:
            if self._plotter:
                self._plotter.start_plot()
            self.shutdown_requested = True
            return
            
        # fill in missing values by refetching
        while np.isnan(data).any():
            timestamp = data[0]
            self._dt = (timestamp - self._last[0]) / TIMESTAMP_UNITS
            self.ukf.predict(self._dt)
            
            if self._plotter:
                self._plotter.timestamps_pred.append(data[0])
                self._plotter.X_data_pred.append(self.ukf.X.copy())

            new_data = self.data_processor.fetch()
            if new_data is None:
                if self._plotter:
                    self._plotter.start_plot()
                self.shutdown_requested = True
                return

            # fill missing fields in `data` from `new_data`
            valid = ~np.isnan(new_data)
            data[valid] = new_data[valid]


        # full row ready, run normal update
        self._dt = (data[0] - self._last[0]) / TIMESTAMP_UNITS
        self._last[0:2] = data[0:2]  # update last after full row collected

        measurement_noise_diag = self._flight_state.measurement_noise_diagonals.copy()
        self.measurement = data

        self.ukf.R = np.diag(measurement_noise_diag)
        self.ukf.predict(self._dt)
        if self._plotter:
            self._plotter.timestamps_pred.append(data[0])
            self._plotter.X_data_pred.append(self.ukf.X.copy())

        self.ukf.update(data[1:])
        if self._plotter:
            self._plotter.X_data.append(self.ukf.X)
            self._plotter.timestamps.append(data[0])
            self._plotter.mahal.append(self.ukf.mahalanobis_dist)
            self._plotter.z_error_score.append(self.ukf.z_error_score)

        self._flight_state.update()
            
    def set_ukf_functions(self): 
        self.ukf.F = self._flight_state.state_transition_function
        self.ukf.Q = self._flight_state.process_covariance_function

    def set_state_time(self):
        if self._plotter:
            self._plotter.state_times.append(self._last[0])


