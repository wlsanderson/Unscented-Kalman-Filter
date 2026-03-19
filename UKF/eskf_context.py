"""
Context / orchestrator for the Error-State Extended Kalman Filter.

Single-state architecture matching the C implementation:
    1. Accumulate pressure + accel + mag for ~1 s
    2. Average accumulated samples, compute initial orientation & pressure
    3. Run predict → set_measurement → update loop

IMU biases are handled offline via calibration.yaml — they are NOT
computed or subtracted by the filter at runtime.
"""

import numpy as np
import numpy.typing as npt
import quaternion as q

from UKF.eskf import ESKF
from UKF.data_processor import DataProcessor
from UKF.constants import (
    ESKF_NOMINAL_DIM,
    ESKF_ERROR_DIM,
    ESKF_MEASUREMENT_DIM,
    ESKF_INITIAL_STATE_ESTIMATE,
    ESKF_INITIAL_STATE_COV,
    ESKF_Q_DIAG,
    ESKF_R_DIAG,
    ESKF_R_DIAG_PRESSURE,
)
from UKF.eskf_functions import (
    measurement_function,
    measurement_jacobian,
    measurement_function_pressure_only,
    measurement_jacobian_pressure_only,
    nominal_predict,
    error_state_jacobian,
    process_noise_matrix,
    R_IMU_TO_BOARD_V1,
    R_IMU_TO_BOARD_V2,
    R_MAG_TO_BOARD_V1,
    R_MAG_TO_BOARD_V2,
)
from UKF.plotter import Plotter


INIT_DURATION_SECONDS = 0.5
"""Accumulation window before filter initialisation."""


class ESKFContext:

    __slots__ = (
        "eskf",
        "data_processor",
        "shutdown_requested",
        "_plotter",
        "_timestamp",
        "_initial_pressure",
        "_initial_mag",
        "_max_altitude",
        "_max_velocity",
        # accumulation
        "_accel_accum",
        "_mag_accum",
        "_pressure_accum",
        "_accum_count",
        "_initialised",
        "_use_mag",
        # rotation matrices
        "_R_imu",
        "_R_mag",
    )

    def __init__(
        self,
        data_processor: DataProcessor,
        plotter: Plotter | None = None,
        hw_version: int = 1.0,
        use_mag: bool = True,
    ):
        self._use_mag = use_mag
        self.eskf = ESKF(
            dim_nom=ESKF_NOMINAL_DIM,
            dim_err=ESKF_ERROR_DIM,
            dim_z=ESKF_MEASUREMENT_DIM if use_mag else 1,
        )
        self._timestamp: np.float32 = np.float32(0.0)
        self.data_processor: DataProcessor = data_processor
        self._plotter = plotter

        self.shutdown_requested: bool = False
        self._initial_pressure: np.float32 = np.float32(0.0)
        self._initial_mag: npt.NDArray = np.zeros(3, dtype=np.float32)
        self._max_velocity: np.float32 = np.float32(0.0)
        self._max_altitude: np.float32 = np.float32(0.0)

        # accumulation buffers (zeroed, matching C memset)
        self._accel_accum = np.zeros(3, dtype=np.float32)
        self._mag_accum = np.zeros(3, dtype=np.float32)
        self._pressure_accum: np.float32 = np.float32(0.0)
        self._accum_count: int = 0
        self._initialised: bool = False

        # select rotation matrices by hardware version
        if hw_version == 0.1:
            self._R_imu = R_IMU_TO_BOARD_V1
            self._R_mag = R_MAG_TO_BOARD_V1
        elif hw_version == 1.0:
            self._R_imu = R_IMU_TO_BOARD_V2
            self._R_mag = R_MAG_TO_BOARD_V2

    def _initialise_filter(self):
        """Average accumulated samples and initialise filter state.

        Mirrors C ``eskf_init()``.
        """
        n = self._accum_count
        avg_accel = self._accel_accum / n
        avg_mag = self._mag_accum / n if self._use_mag else np.zeros(3, dtype=np.float32)

        self._initial_pressure = self._pressure_accum / n

        # compute initial orientation
        init_quat, mag_world = self._calculate_initial_orientation(avg_accel, avg_mag)

        # set nominal state
        self.eskf.x_nom = np.copy(ESKF_INITIAL_STATE_ESTIMATE).astype(np.float32)
        self.eskf.x_nom[2:6] = q.as_float_array(init_quat)

        # set covariance
        self.eskf.P = np.copy(ESKF_INITIAL_STATE_COV).astype(np.float32)

        # set Q and R (single set, matching C)
        self.eskf.R = np.diag(ESKF_R_DIAG if self._use_mag else ESKF_R_DIAG_PRESSURE).astype(np.float32)

        # store mag world reference
        self._initial_mag = mag_world

        # inject function handles
        self._wire_filter_functions()

        self._initialised = True

    def update(self):
        """Main loop body: fetch sensor data, accumulate or predict/update."""
        if not self.data_processor.fetch():
            if self._plotter:
                self._plotter.start_plot()
            self.shutdown_requested = True
            return
        dt = np.float32(self.data_processor.dt)
        self._timestamp += dt

        # ---- raw sensor readings ----
        pressure = self.data_processor.measurements[0]
        accel_sensor = np.array(self.data_processor.measurements[1:4], dtype=np.float32)
        gyro_sensor = np.array(self.data_processor.measurements[4:7], dtype=np.float32)
        mag_raw = np.array(self.data_processor.measurements[7:10], dtype=np.float32)

        # ---- accumulation phase ----
        if not self._initialised:
            self._accel_accum += accel_sensor
            if self._use_mag:
                self._mag_accum += mag_raw
            self._pressure_accum += pressure
            self._accum_count += 1

            if self._timestamp >= INIT_DURATION_SECONDS:
                self._initialise_filter()
                if self._plotter:
                    self._plotter.state_times.append(self._timestamp)
                print(
                    f"ESKF: initialised from {self._accum_count} samples\n"
                    f"  initial pressure: {self._initial_pressure:.2f}\n"
                    f"  initial quat: {self.eskf.x_nom[2:6]}\n"
                    f"  mag world: {self._initial_mag}"
                )
            return

        # ---- control input ----
        u = np.array([
            accel_sensor[0], accel_sensor[1], accel_sensor[2],
            gyro_sensor[0], gyro_sensor[1], gyro_sensor[2],
        ], dtype=np.float32)

        # ---- measurement (normalise mag) ----
        z = self._build_measurement(pressure, mag_raw)

        # ---- predict ----
        self.eskf.predict(dt, u)

        if self._plotter:
            self._plotter.timestamps_pred.append(self._timestamp)
            self._plotter.X_data_pred.append(self.eskf.x_nom.copy())

        # ---- update ----
        self.eskf.update(z, self._initial_pressure, self._initial_mag)

        if self._plotter:
            self._plotter.X_data.append(self.eskf.x_nom.copy())
            self._plotter.timestamps.append(self._timestamp)
            self._plotter.uncerts.append(np.diag(self.eskf.P))
            self._plotter.mahal.append(self.eskf.mahalanobis_dist)
            self._plotter.z_error_score.append(self.eskf.z_error_score)
            self._plotter.pressure_nis_ref.append(0.0)
            self._plotter.pressure_alt.append(self._compute_pressure_alt())

        self._max_altitude = max(self._max_altitude, self.eskf.x_nom[0])
        self._max_velocity = max(self._max_velocity, self.eskf.x_nom[1])

    def set_state_time(self):
        if self._plotter:
            self._plotter.state_times.append(self._timestamp)

    def _compute_pressure_alt(self) -> float:
        """Compute altitude from raw pressure, zeroed to launch-pad level."""
        p = float(self.data_processor.measurements[0])
        p0 = float(self._initial_pressure)
        if p <= 0 or p0 <= 0:
            return 0.0
        return 44330.0 * (1.0 - (p / p0) ** (1.0 / 5.255876))

    def _build_measurement(self, pressure: float, mag_raw: np.ndarray) -> np.ndarray:
        if not self._use_mag:
            return np.array([pressure], dtype=np.float32)

        mag = mag_raw.astype(np.float32)
        mag_norm = np.linalg.norm(mag)
        if mag_norm > 0:
            mag /= mag_norm

        return np.array([pressure, mag[0], mag[1], mag[2]], dtype=np.float32)

    def _wire_filter_functions(self) -> None:
        """Connect function hooks with the selected rotation matrices."""
        self.eskf.nominal_predict_func = lambda x, u, dt: nominal_predict(x, u, dt, R_imu=self._R_imu)
        self.eskf.error_jacobian_func = lambda x, u, dt: error_state_jacobian(x, u, dt, R_imu=self._R_imu)
        self.eskf.process_noise_func = lambda x, u, dt: process_noise_matrix(x, u, dt, ESKF_Q_DIAG)

        if self._use_mag:
            self.eskf.measurement_func = lambda x, p, m: measurement_function(x, p, m, R_mag=self._R_mag)
            self.eskf.measurement_jacobian_func = lambda x, p, m: measurement_jacobian(x, p, m, R_mag=self._R_mag)
        else:
            self.eskf.measurement_func = lambda x, p, m: measurement_function_pressure_only(x, p)
            self.eskf.measurement_jacobian_func = lambda x, p, m: measurement_jacobian_pressure_only(x, p)

    def _calculate_initial_orientation(self, acc_sensor_raw, mag_sensor_raw):
        """Compute initial board→world quaternion from averaged IMU (+mag)."""
        # normalise
        acc_norm = np.linalg.norm(acc_sensor_raw)
        if acc_norm == 0:
            raise ValueError("Zero-length sensor vector passed to initialization")

        acc_sensor_n = acc_sensor_raw / acc_norm

        if self._use_mag:
            mag_norm = np.linalg.norm(mag_sensor_raw)
            if mag_norm == 0:
                raise ValueError("Zero-length sensor vector passed to initialization")
            mag_sensor_n = mag_sensor_raw / mag_norm
        else:
            mag_sensor_n = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # rotate sensor → board
        acc_board = self._R_imu @ acc_sensor_n
        mag_board = self._R_mag.T @ mag_sensor_n

        # roll/pitch from accel
        roll = np.arctan2(acc_board[1], acc_board[2])
        pitch = np.arctan2(-acc_board[0], np.sqrt(acc_board[1] ** 2 + acc_board[2] ** 2))

        # yaw from mag (tilt-compensated) if available; otherwise 0
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        mx, my, mz = mag_board
        if self._use_mag:
            mx2 = mx * cp + mz * sp
            my2 = mx * sr * sp + my * cr - mz * sr * cp
            yaw = np.arctan2(-my2, mx2)
        else:
            yaw = 0.0

        # Euler → quaternion (ZYX)
        cr2, sr2 = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp2, sp2 = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy2, sy2 = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

        w = cr2 * cp2 * cy2 + sr2 * sp2 * sy2
        x = sr2 * cp2 * cy2 - cr2 * sp2 * sy2
        y = cr2 * sp2 * cy2 + sr2 * cp2 * sy2
        z = cr2 * cp2 * sy2 - sr2 * sp2 * cy2

        init_quat = q.quaternion(w, x, y, z).normalized()

        # rotate mag to world frame: q @ [0, mag_board] @ q*
        if self._use_mag:
            mag_board_q = q.quaternion(0.0, *mag_board)
            mag_world_q = init_quat * mag_board_q * init_quat.conjugate()
            mag_world = np.array([mag_world_q.x, mag_world_q.y, mag_world_q.z], dtype=np.float32)
        else:
            mag_world = np.zeros(3, dtype=np.float32)

        return init_quat, mag_world
