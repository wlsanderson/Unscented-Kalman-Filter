from UKF.ukf import UKF
from UKF.sigma_points import SigmaPoints
from UKF.data_processor import DataProcessor
from UKF.constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_ESTIMATE, INITIAL_STATE_COV, TIMESTAMP_UNITS
from UKF.ukf_functions import measurement_function
from UKF.plotter import Plotter
from UKF.state import State, StandbyState
import numpy as np
import numpy.typing as npt
import quaternion as q

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
        "_initial_mag",
        "_initial_quat",
        "_max_altitude",
        "_max_velocity",
        "measurement",
    )

    def __init__(self, data_processor: DataProcessor, plotter: Plotter | None = None):
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
        self._timestamp = 0.0
        self.data_processor: DataProcessor = data_processor
        self.initialize_filter_settings()
        self._plotter = plotter
        self._flight_state: State = StandbyState(self)
        self.shutdown_requested: bool = False
        self._initial_pressure: np.float64 | None = None
        self._initial_mag: npt.NDArray | None = None
        self._initial_quat: npt.NDArray | None = None
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

        if self._initial_mag is None:
            self._initial_mag = self.data_processor.measurements[-3:]
        
        if self._initial_quat is None:
            acc = self.data_processor.measurements[1:4]
            mag = self.data_processor.measurements[-3:]
            self._initial_quat = self.calculate_initial_orientation(acc, mag)
            self.ukf.X[18:22] = self._initial_quat

        # runs predict with the calculated dt and control input
        control_input = self._flight_state.control_input.copy()
        self.ukf.predict(self.data_processor.dt, control_input)
        if self._plotter:
            self._plotter.timestamps_pred.append(self._timestamp)
            self._plotter.X_data_pred.append(self.ukf.X.copy())
        self.ukf.R = np.diag(measurement_noise_diag)

        self.ukf.update(self.data_processor.measurements, self._initial_pressure, self._initial_mag, self._initial_quat, control_input)
        if self._plotter:
            self._plotter.X_data.append(self.ukf.X.copy())
            self._plotter.timestamps.append(self._timestamp)
            self._plotter.uncerts.append(np.diag(self.ukf.P))
            self._plotter.mahal.append(self.ukf.mahalanobis_dist)
            self._plotter.z_error_score.append(self.ukf.z_error_score)
        
        # self._plotter.timestamps.append(self._timestamp)

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

    def calculate_initial_orientation(self, acc, mag):

        # -------- 1) Undo the board’s 45° mounting rotation --------
        # Your check code APPLIES:
        #   x' =  x/√2 + y/√2
        #   y' = -x/√2 + y/√2
        # So to go from *sensor* frame → true vehicle frame, we must apply the inverse:
        R45 = np.array([[ 1/np.sqrt(2), -1/np.sqrt(2), 0],
                        [ 1/np.sqrt(2),  1/np.sqrt(2), 0],
                        [ 0,              0,            1]])

        acc = R45 @ acc
        mag = R45 @ mag

        # -------- 2) Normalize sensors --------
        acc = acc / np.linalg.norm(acc)
        mag = mag / np.linalg.norm(mag)

        # -------- 3) Compute roll, pitch from accelerometer --------
        # ENU convention matching numpy.quaternion
        roll  = np.arctan2(acc[1], acc[2])
        pitch = np.arctan2(-acc[0], np.sqrt(acc[1]**2 + acc[2]**2))

        # -------- 4) Tilt-compensated yaw from magnetometer --------
        cr = np.cos(roll);  sr = np.sin(roll)
        cp = np.cos(pitch); sp = np.sin(pitch)

        mx, my, mz = mag

        mag_x = mx*cp + mz*sp
        mag_y = mx*sr*sp + my*cr - mz*sr*cp

        yaw = np.arctan2(-mag_y, mag_x)

        # -------- 5) Convert Euler→quaternion (ENU, intrinsic xyz) --------
        cy = np.cos(yaw/2);  sy = np.sin(yaw/2)
        cp = np.cos(pitch/2); sp = np.sin(pitch/2)
        cr = np.cos(roll/2);  sr = np.sin(roll/2)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        # numpy.quaternion expects quaternion(w, x, y, z)
        return np.array([qw, qx, qy, qz])