import numpy as np
import numpy.typing as npt

from UKF.sigma_points import SigmaPoints
import quaternion as q
import time


class UKF:

    __slots__ = (
        "X",
        "F",
        "P",
        "Q",
        "H",
        "R",
        "U",
        "_dim_x",
        "_dim_z",
        "_sigma_points_class",
        "_num_sigmas",
        "_sigmas_f",
        "_sigmas_h",
        "_quat_idx",
        "_rotvec_idx",
        "_vec_idx",
        "pred_z",
        "mahalanobis_dist",
        "z_error_score",
    )

    def __init__(self, dim_x: int, dim_z: int, points: SigmaPoints):
        self.X = np.zeros(dim_x)
        self.F = None
        self.P = None
        self.Q = None
        self.H = None
        self.R = None
        self._dim_x = dim_x
        self._dim_z = dim_z
        
        self._sigma_points_class  = points
        self._num_sigmas = self._sigma_points_class.num_sigmas()
        self._sigmas_f = np.zeros([self._num_sigmas, dim_x])
        self._sigmas_h = np.zeros([self._num_sigmas, dim_z])
        self._quat_idx = slice(dim_x - 4, dim_x)
        self._rotvec_idx = slice(dim_x - 4, dim_x - 1)
        self._vec_idx = slice(0, dim_x - 4)

        # debug/analysis stuff
        self.pred_z = None
        self.mahalanobis_dist = None
        self.z_error_score = None


    def predict(self, dt, u):
        if (dt < 1e-12):
            raise ValueError("dt must be positive and non-zero")
        self.X[self._quat_idx] /= np.linalg.norm(self.X[self._quat_idx])
        self.compute_process_sigmas(dt, self.Q(dt), u)
        self.X, self.P = self._unscented_transform_F(
            sigmas = self._sigmas_f,
            Wm = self._sigma_points_class.Wm,
            Wc = self._sigma_points_class.Wc,
            X = self.X,
            u = u,
        )

    def update(self, z, init_pressure, init_mag, init_quat, u):
        sigmas_h = []
        for s in self._sigmas_f:
            sigmas_h.append(self.H(s, init_pressure, init_mag, init_quat, self.X))
        self._sigmas_h = np.array(sigmas_h)
        pred_z, innovation_cov = self._unscented_transform_H(
            self._sigmas_h,
            self._sigma_points_class.Wm,
            self._sigma_points_class.Wc,
            self.R,
            )
        self.pred_z = pred_z
        innovation_cov_inv = np.linalg.inv(innovation_cov)
       
        P_cross_covariance = self._calculate_cross_cov(self.X, pred_z)
        
        kalman_gain = P_cross_covariance @ innovation_cov_inv
        residual = np.subtract(z, pred_z)
        self.mahalanobis_dist = residual.T @ innovation_cov_inv @ residual
        self.z_error_score = (residual**2) / np.diag(innovation_cov)
        delta_x =  np.dot(kalman_gain, residual)
        if u[0] is None:
            delta_x[12:18] = np.zeros(6)
        quat = q.from_float_array(self.X[self._quat_idx])

        delta_q = q.from_rotation_vector(delta_x[self._rotvec_idx])
        
        self.X[self._vec_idx] += delta_x[self._vec_idx]
        self.X[self._quat_idx] = q.as_float_array((delta_q * quat).normalized())
        new_P = self.P - np.dot(np.dot(kalman_gain, innovation_cov), np.transpose(kalman_gain))
        self.P = new_P

        
        

    def _unscented_transform_F(self, sigmas: npt.NDArray[np.float64], Wm, Wc, X, u = None, noise_cov = None):
        # splitting sigma points up into vector states and quaternion states
        vector_sigmas = sigmas[:, self._vec_idx]
        quat_sigmas = q.from_float_array(sigmas[:, self._quat_idx])
        quat_state = q.from_float_array(X[self._quat_idx])
        # small delta quaternions are made by multiplying the quaternion sigmas by the
        # inverse of the current quaternion state
        delta_quats = quat_sigmas * quat_state.conjugate()
        # delta quaternion rotations are converted into delta rotation vectors
        delta_rotvecs = q.as_rotation_vector(delta_quats)
        # the mean of the rotation vector is calculated by multiplying each sigma by each weight
        # this mean cannot be calculated with quaternions directly due to the inability to sum
        # them.
        
        mean_delta_rotvec = np.dot(Wm, delta_rotvecs)
        # mean delta rotation vector is transformed back into a mean delta quaternion and
        # multiplied into the state to get the quaternion prediction
        mean_delta_quat = q.from_rotation_vector(mean_delta_rotvec)
        mean_quat = q.as_float_array((mean_delta_quat * quat_state))
        
        # the vector portion of the sigma points are calculated normally
        vector_mean = np.dot(Wm, vector_sigmas)
        # if there is a control input, then its either standby state or landed state.
        # ideally, this should be handled by checking the flight state.
        x_mean = np.concatenate([vector_mean, mean_quat])

        vec_residual = vector_sigmas - vector_mean[np.newaxis, :]
        full_residuals = np.hstack((vec_residual, delta_rotvecs))
        P_covariance = (full_residuals.T * Wc) @  full_residuals
        if noise_cov is not None:
            P_covariance += noise_cov
        return (x_mean, P_covariance)
    
    @staticmethod
    def _unscented_transform_H(sigmas: npt.NDArray[np.float64], Wm, Wc, noise_cov = None):
        x_mean = np.dot(Wm, sigmas)
        residual = sigmas - x_mean[np.newaxis, :]
        P_covariance = np.dot(residual.T, np.dot(np.diag(Wc), residual))

        if noise_cov is not None:
            P_covariance += noise_cov
        return (x_mean, P_covariance)

    def compute_process_sigmas(self, dt, Q, u):
        # normalize quaternion states
        self.X[self._quat_idx] /= np.linalg.norm(self.X[self._quat_idx])
        sigmas = self._sigma_points_class.calculate_sigma_points(self.X, self.P, Q)
        for i, s in enumerate(sigmas):
            self._sigmas_f[i] = self.F(s, dt, u)

    def _calculate_cross_cov(self, x, z):
        """
        Computes cross covariance between sigma points in state space and measurement space.
        Quaternion state is represented in the tangent space R^3 (rotation vectors).
        """
        P_cross_cov = np.zeros((self._dim_x - 1, self._dim_z))
        n = self._sigmas_f.shape[0]
        for i in range(n):
            dx_vector = np.subtract(self._sigmas_f[i][self._vec_idx], x[self._vec_idx])
            quat_sigmas_f = q.from_float_array(self._sigmas_f[i][self._quat_idx])
            quat_x = q.from_float_array(x[self._quat_idx])
            delta_quats = quat_sigmas_f * quat_x.conjugate()
            delta_rotvecs = q.as_rotation_vector(delta_quats)
            dx = np.concatenate([dx_vector, delta_rotvecs])
            dz = np.subtract(self._sigmas_h[i], z)
            P_cross_cov += self._sigma_points_class.Wc[i] * np.outer(dx, dz)
        return P_cross_cov
