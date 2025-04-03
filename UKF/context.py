import pandas as pd
from UKF.ukf import UKF
from UKF.sigma_points import SigmaPoints
from UKF.data_processor import DataProcessor
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_ESTIMATE, INITIAL_STATE_COV
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
    )

    def __init__(self, data_processor: DataProcessor, plotter: Plotter | None = None):
        sigma_points = SigmaPoints(
            n = STATE_DIM,
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
        self._flight_state: State = StandbyState(self)
        self.shutdown_requested: bool = False
        self._last: npt.NDArray[np.float64] = data_processor.get_initial_vals()
        self._dt: np.float64 = 0.0
        self._initial_altitude: np.float64 = self._last[1]

        self.initialize_filter_settings()
        
    def initialize_filter_settings(self):
        self.ukf.X = INITIAL_STATE_ESTIMATE.copy()
        self.ukf.P = INITIAL_STATE_COV.copy()
        self.ukf.H = measurement_function

    def update(self, exclude_repeated_vals: bool = False):
        data = self.data_processor.fetch()
        if data is None:
            # end of file
            if self._plotter:
                self._plotter.start_plot()
            self.shutdown_requested = True
            return
        measurement_noise_diag = self._get_measurement_noise(data, exclude_repeated_vals=True)
        data = self._last # _get_measurement_noise sets _last to the updated backfilled data
        self.ukf.R = np.diag(measurement_noise_diag)
        self.ukf.predict(self._dt)
        self.ukf.update(data[1:], H_args=self._initial_altitude)
        if self._plotter:
            self._plotter.P_data.append(self.ukf.P)
            self._plotter.X_data.append(self.ukf.X)
            self._plotter.timestamps.append(data[0])
        
    def _get_measurement_noise(self, data: pd.Series, exclude_repeated_vals: bool = False):
        measurement_noise_diags = self._flight_state.measurement_noise_diagonals.copy()
        # initialize R matrix with extremely high noise for repeated or null values
        z_noise = np.full(len(measurement_noise_diags), 1e9)
        nulls = data.isnull()
        non_nan_mesurements = ~nulls[1:] # get only the null values of measurements, not timestamps
        valid_measurements = non_nan_mesurements

        # if excluding repeated values, noise will stay high for values that match their last measurement
        if exclude_repeated_vals:
            non_repeated_measurements = data[1:] != self._last[1:]
            valid_measurements = valid_measurements & non_repeated_measurements
            
        # for values that are not null and not repeated, overwrite the high noise with actual noise
        z_noise[valid_measurements] = measurement_noise_diags[valid_measurements]

        # set the null values to the last actual non-null values recorded
        data[nulls] = self._last[nulls].astype(data.dtype)
        self._dt = (data.values[0] - self._last[0]) / 1e9
        self._last = np.array(data.values.astype(np.float64))
        return z_noise
    
    def set_ukf_functions(self): 
        self.ukf.F = self._flight_state.state_transition_function
        self.ukf.Q = self._flight_state.process_covariance_function

