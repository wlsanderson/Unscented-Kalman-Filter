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
from UKF.ukf_functions import print_c_array

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

        
        if self._initial_quat is None:
            acc = self.data_processor.measurements[1:4]
            mag = self.data_processor.measurements[-3:]
            self._initial_quat, self._initial_mag = self.calculate_initial_orientation_from_sensors(acc, mag)
            self.ukf.X[18:22] = q.as_float_array(self._initial_quat)

        # runs predict with the calculated dt and control input
        control_input = self._flight_state.control_input
        self.ukf.predict(self.data_processor.dt, control_input)
        if self._plotter:
            self._plotter.timestamps_pred.append(self._timestamp)
            self._plotter.X_data_pred.append(self.ukf.X.copy())
        self.ukf.R = np.diag(measurement_noise_diag)

        self.ukf.update(self.data_processor.measurements, self._initial_pressure, self._initial_mag, control_input)
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

    def calculate_initial_orientation_from_sensors(self, acc_imu_raw, mag_raw):
        """
        Compute an initial quaternion (vehicle -> world) given raw imu accel and raw magnetometer.
        Both inputs are 3-element arrays in their respective sensor frames:
        - acc_imu_raw : accelerometer reading in IMU (acc+gyro) frame
        - mag_raw     : magnetometer reading in magnetometer sensor frame

        Returns:
        - init_quat : numpy-quaternion q object that maps VEHICLE -> WORLD (q: vehicle->world)
        - mag_world : 3-element numpy array of the magnetic field expressed in WORLD frame (normalized)
        Notes:
        - world +Z is UP.
        - The accelerometer at rest should measure the 'up' direction (specific force),
            so a level vehicle -> acc_vehicle ≈ [0,0,1] (after normalization).
        """
        vehicle_to_imu = np.array([
            [ 1.0/np.sqrt(2),  1.0/np.sqrt(2), 0.0],
            [-1.0/np.sqrt(2),  1.0/np.sqrt(2), 0.0],
            [ 0.0,      0.0,     1.0]
        ])
        imu_to_vehicle = vehicle_to_imu.T
        R_mag_to_vehicle = np.diag([1.0, 1.0, -1.0])

        # 1) Transform raw sensor vectors into VEHICLE frame using fixed, known transforms
        acc_vehicle = imu_to_vehicle @ np.asarray(acc_imu_raw, dtype=float)
        mag_vehicle = R_mag_to_vehicle @ np.asarray(mag_raw, dtype=float)
        # 2) Normalize (we only care about direction for attitude initialization)
        if np.linalg.norm(acc_vehicle) == 0 or np.linalg.norm(mag_vehicle) == 0:
            raise ValueError("Zero-length sensor vector passed to initialization")

        acc_v = acc_vehicle / np.linalg.norm(acc_vehicle)
        mag_v = mag_vehicle / np.linalg.norm(mag_vehicle)

        # 3) Compute roll/pitch from accelerometer (assumes acc measures specific force ≈ +up)
        # These formulas give roll=0,pitch=0 when acc = [0,0,1]
        ax, ay, az = acc_v
        roll  = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az))

        # 4) Level the magnetometer reading (rotate mag into level frame using roll/pitch)
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        mx, my, mz = mag_v

        # rotate mag by roll then pitch (body -> leveled body)
        # this matches the standard "tilt compensation" ordering
        mx2 = mx*cp + mz*sp
        my2 = mx*sr*sp + my*cr - mz*sr*cp

        # 5) Yaw from leveled magnetometer (sign convention chosen to match your previous code)
        yaw = np.arctan2(-my2, mx2)

        # 6) Convert yaw/pitch/roll (Z-Y-X) into quaternion (vehicle -> world)
        cy, sy = np.cos(yaw*0.5), np.sin(yaw*0.5)
        cp2, sp2 = np.cos(pitch*0.5), np.sin(pitch*0.5)
        cr2, sr2 = np.cos(roll*0.5), np.sin(roll*0.5)

        w = cr2*cp2*cy + sr2*sp2*sy
        x = sr2*cp2*cy - cr2*sp2*sy
        y = cr2*sp2*cy + sr2*cp2*sy
        z = cr2*cp2*sy - sr2*sp2*cy

        init_quat = q.quaternion(w, x, y, z).normalized()

        mag_vehicle_q = q.quaternion(0.0, *mag_v)
        mag_world_q = init_quat * mag_vehicle_q * init_quat.conjugate()
        mag_world = np.array([mag_world_q.x, mag_world_q.y, mag_world_q.z], dtype=float)
        mag_world /= np.linalg.norm(mag_world)
        return init_quat, mag_world