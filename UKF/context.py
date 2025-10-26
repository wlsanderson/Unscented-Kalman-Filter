from UKF.ukf import UKF
from UKF.sigma_points import SigmaPoints
from UKF.data_processor import DataProcessor
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, GRAVITY, INITIAL_STATE_ESTIMATE, INITIAL_STATE_COV, TIMESTAMP_UNITS
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
        "_initial_altitude",
        "_max_altitude",
        "_max_velocity",
        "measurement",
    )

    def __init__(self, data_processor: DataProcessor, plotter: Plotter | None = None):
        self._last: npt.NDArray[np.float64] = data_processor.get_initial_vals()
        sigma_points = SigmaPoints(
            # n is the dimension of the state, minus one due to the quaternion representation
            n = STATE_DIM - 1,
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
        self.initialize_filter_settings()
        self._flight_state: State = StandbyState(self)
        self.shutdown_requested: bool = False
        self._dt: np.float64 = 0.0
        self._initial_altitude: np.float64 = self._last[1]
        self._max_velocity = 0.0
        self._max_altitude = 0.0

        
        
    def initialize_filter_settings(self):
        self.ukf.X = INITIAL_STATE_ESTIMATE.copy()
        state_estimate = INITIAL_STATE_ESTIMATE.copy()
        state_estimate[2:5] = self._last[2:5] * GRAVITY
        self.ukf.X = state_estimate
        self.ukf.P = INITIAL_STATE_COV.copy()
        self.ukf.H = measurement_function

    def update(self):
        # start with array of nans
        data = np.empty(MEASUREMENT_DIM + 1)
        data[:] = np.nan
        while np.isnan(data).any():
            # fetched data wont always return a full measurement set, other values will be nan
            new_data = self.data_processor.fetch()
            # fetch returns none if at end of file
            if new_data is None:
                if self._plotter:
                    self._plotter.start_plot()
                self.shutdown_requested = True
                return
            # calculates dt and updates last measurements
            # new data should ALWAYS have a timestamp, should never be nan

            # checks which measurements of the new data are nan, and updates the data variable
            # with the non-nan values
            valid = ~np.isnan(new_data)
            data[valid] = new_data[valid]

        measurement_noise_diag = self._flight_state.measurement_noise_diagonals.copy()
        self.measurement = data
        self._dt = (new_data[0] - self._last[0]) / TIMESTAMP_UNITS
        self._last = new_data
        # runs predict with the calculated dt
        self.ukf.predict(self._dt)
        if self._plotter:
            self._plotter.timestamps_pred.append(new_data[0])
            self._plotter.X_data_pred.append(self.ukf.X.copy())
        self.ukf.R = np.diag(measurement_noise_diag)
        self.ukf.update(data[1:], self._initial_altitude)
        if self._plotter:
            self._plotter.X_data.append(self.ukf.X.copy())
            self._plotter.timestamps.append(data[0])
            self._plotter.mahal.append(self.ukf.mahalanobis_dist)
            self._plotter.z_error_score.append(self.ukf.z_error_score)

        self._max_altitude = max(self._max_altitude, self.ukf.X[0])
        self._max_velocity = max(self._max_velocity, self.ukf.X[1])
        self._flight_state.update()
            
    def set_ukf_functions(self): 
        self.ukf.F = self._flight_state.state_transition_function
        self.ukf.Q = self._flight_state.process_covariance_function

    def set_state_time(self):
        if self._plotter:
            self._plotter.state_times.append(self._last[0])


