from UKF.ukf import UKF
from UKF.sigma_points import SigmaPoints
from UKF.data_processor import DataProcessor
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_ESTIMATE, INITIAL_STATE_COV, TIMESTAMP_UNITS, CONTROL_INPUT_DIM
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
        "_flight_state",
        "_timestamp",
        "_initial_pressure",
        "_max_altitude",
        "_max_velocity",
        "measurement",
    )

    def __init__(self, data_processor: DataProcessor, plotter: Plotter | None = None):
        sigma_points = SigmaPoints(
            # n is the dimension of the state, minus one due to the quaternion representation
            n = STATE_DIM - 1,
            #n = STATE_DIM,
            alpha = ALPHA,
            beta = BETA,
            kappa = KAPPA,
        )
        self.ukf = UKF(
            dim_x = STATE_DIM,
            dim_z = MEASUREMENT_DIM,
            points = sigma_points,
        )
        self._timestamp = 0.0
        self.data_processor: DataProcessor = data_processor
        self.initialize_filter_settings()
        self._plotter = plotter
        self._flight_state: State = StandbyState(self)
        self.shutdown_requested: bool = False
        self._initial_pressure: np.float64 | None = None
        self._max_velocity = 0.0
        self._max_altitude = 0.0
        
        

        
        
    def initialize_filter_settings(self):
        state_estimate = INITIAL_STATE_ESTIMATE.copy()
        self.ukf.X = state_estimate
        self.ukf.P = INITIAL_STATE_COV.copy()
        self.ukf.H = measurement_function

    def update(self):
        if (not self.data_processor.fetch()):
            if self._plotter:
                self._plotter.start_plot()
            self.shutdown_requested = True
            return
        self._timestamp += self.data_processor.dt
        measurement_noise_diag = self._flight_state.measurement_noise_diagonals.copy()

        if self._initial_pressure is None:
            self._initial_pressure = self.data_processor.measurements[0]

        # runs predict with the calculated dt and control input
        self.ukf.predict(self.data_processor.dt)
        if self._plotter:
            self._plotter.timestamps_pred.append(self._timestamp)
            self._plotter.X_data_pred.append(self.ukf.X.copy())
        self.ukf.R = np.diag(measurement_noise_diag)
        self.ukf.update(self.data_processor.measurements, self._initial_pressure)
        if self._plotter:
            self._plotter.X_data.append(self.ukf.X.copy())
            self._plotter.timestamps.append(self._timestamp)
            self._plotter.mahal.append(self.ukf.mahalanobis_dist)
            self._plotter.z_error_score.append(self.ukf.z_error_score)
#        self._plotter.timestamps.append(self._timestamp)

        self._max_altitude = max(self._max_altitude, self.ukf.X[2])
        self._max_velocity = max(self._max_velocity, self.ukf.X[5])
        self._flight_state.update()
            
    def set_ukf_functions(self): 
        self.ukf.F = self._flight_state.state_transition_function
        self.ukf.Q = self._flight_state.process_covariance_function

    def set_state_time(self):
        if self._plotter:
            self._plotter.state_times.append(self._timestamp)
        pass


