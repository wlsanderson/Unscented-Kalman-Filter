"""Minimal Error-State Extended Kalman Filter (vertical position/velocity)."""

import numpy as np
import numpy.typing as npt
import quaternion as q

from UKF.constants import (
    ESKF_NOMINAL_DIM,
    ESKF_ERROR_DIM,
    ESKF_MEASUREMENT_DIM,
    ESKF_PRESSURE_VEL_COUPLING_SPEED,
    ESKF_PRESSURE_VEL_COUPLING_SHARPNESS,
)


class ESKF:

    __slots__ = (
        "x_nom",
        "P",
        "R",
        "nominal_predict_func",
        "error_jacobian_func",
        "process_noise_func",
        "measurement_func",
        "measurement_jacobian_func",
        "_dim_nom",
        "_dim_err",
        "_dim_z",
        "_quat_idx",
        # debug / analysis
        "pred_z",
        "mahalanobis_dist",
        "z_error_score",
    )

    def __init__(
        self,
        dim_nom: int = ESKF_NOMINAL_DIM,
        dim_err: int = ESKF_ERROR_DIM,
        dim_z: int = ESKF_MEASUREMENT_DIM,
    ):
        self._dim_nom = dim_nom
        self._dim_err = dim_err
        self._dim_z = dim_z
        self._quat_idx = 2  # start index of quaternion in nominal state

        # state
        self.x_nom = np.zeros(dim_nom, dtype=np.float32)
        self.x_nom[self._quat_idx] = np.float32(1.0)  # identity quaternion w=1
        self.P = np.eye(dim_err, dtype=np.float32)
        self.R = np.eye(dim_z, dtype=np.float32)

        # injectable function handles (set by State subclasses)
        self.nominal_predict_func = None
        self.error_jacobian_func = None
        self.process_noise_func = None
        self.measurement_func = None
        self.measurement_jacobian_func = None

        # debug
        self.pred_z = np.zeros(dim_z, dtype=np.float32)
        self.mahalanobis_dist = 0.0
        self.z_error_score = np.zeros(dim_z, dtype=np.float32)

    def predict(self, dt: float, u: npt.NDArray):
        """ESKF prediction step."""
        if dt < 1e-12:
            raise ValueError("dt must be positive and non-zero")

        # normalize quaternion
        qi = self._quat_idx
        qn = self.x_nom[qi:qi + 4]
        norm = np.linalg.norm(qn)
        if norm > 1e-10:
            self.x_nom[qi:qi + 4] = qn / norm

        # propagate nominal state
        self.x_nom = self.nominal_predict_func(self.x_nom, u, dt)
        # error-state covariance propagation
        F_d = self.error_jacobian_func(self.x_nom, u, dt)
        Q_d = self.process_noise_func(self.x_nom, u, dt)
        self.P = F_d @ self.P @ F_d.T + Q_d

    def update(self, z: npt.NDArray, init_alt: float, init_mag: npt.NDArray):
        """ESKF measurement update step."""
        # predicted measurement
        z_pred = self.measurement_func(self.x_nom, init_alt, init_mag)
        self.pred_z = z_pred

        # measurement Jacobian (4x5)
        H = self.measurement_jacobian_func(self.x_nom, init_alt, init_mag)

        # innovation
        y = z - z_pred

        # innovation covariance
        S = H @ self.P @ H.T + self.R
        S_inv = np.linalg.inv(S)

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # pressure decoupling: above threshold, pressure stops correcting velocity
        # and quaternion states
        speed = float(abs(self.x_nom[1]))
        coupling = 1.0 / (1.0 + np.exp(
            ESKF_PRESSURE_VEL_COUPLING_SHARPNESS
            * (speed - ESKF_PRESSURE_VEL_COUPLING_SPEED)
        ))
        K[1:5, 0] *= coupling
        # error-state correction
        dx = K @ y

        # diagnostics
        self.mahalanobis_dist = float(y.T @ S_inv @ y)
        S_diag = np.diag(S)
        S_diag_safe = np.where(S_diag > 0, S_diag, 1e-12)
        self.z_error_score = y / np.sqrt(S_diag_safe)

        # inject error into nominal state
        # position and velocity: additive
        self.x_nom[0] += dx[0]
        self.x_nom[1] += dx[1]

        # quaternion: multiplicative update  (dx[2:5] = dtheta)
        dtheta = dx[2:5]
        delta_q = q.from_rotation_vector(dtheta)
        qi = self._quat_idx
        quat = q.quaternion(
            self.x_nom[qi], self.x_nom[qi + 1],
            self.x_nom[qi + 2], self.x_nom[qi + 3],
        )
        q_new = quat * delta_q
        self.x_nom[qi] = q_new.w
        self.x_nom[qi + 1] = q_new.x
        self.x_nom[qi + 2] = q_new.y
        self.x_nom[qi + 3] = q_new.z

        # Covariance Update: P = P - K @ (H @ P)
        self.P = self.P - K @ H @ self.P
        self.P = 0.5 * (self.P + self.P.T) # symmetrize for stability

    @property
    def X(self) -> npt.NDArray:
        """Alias for x_nom to match UKF interface for plotter compatibility."""
        return self.x_nom

