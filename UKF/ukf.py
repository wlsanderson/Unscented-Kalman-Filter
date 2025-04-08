import numpy as np
import numpy.typing as npt
import scipy
import scipy.linalg

from UKF.sigma_points import SigmaPoints
from UKF.quaternion import quat_multiply, quat2rotvec, rotvec2quat, quat_inv


class UKF:

    __slots__ = (
        "X",
        "F",
        "P",
        "Q",
        "H",
        "R",
        "_dim_x",
        "_dim_z",
        "_sigma_points_class",
        "_num_sigmas",
        "_sigmas_f",
        "_sigmas_h",
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

    def predict(self, dt):
        self.compute_process_sigmas(dt, self.Q(dt))
        self.X, self.P = self._unscented_transform_F(
            sigmas = self._sigmas_f,
            Wm = self._sigma_points_class.Wm,
            Wc = self._sigma_points_class.Wc,
            X = self.X,
        )

    def update(self, z, **H_args):
        sigmas_h = []
        for s in self._sigmas_f:
            sigmas_h.append(self.H(s, **H_args))
        self._sigmas_h = np.atleast_2d(sigmas_h)
        pred_z, innovation_cov = self._unscented_transform_H(
            self._sigmas_h,
            self._sigma_points_class.Wm,
            self._sigma_points_class.Wc,
            self.R
            )
        innovation_cov_inv = np.linalg.inv(innovation_cov)
        P_cross_covariance = self._calculate_cross_cov(self.X, pred_z)
        kalman_gain = np.dot(P_cross_covariance, innovation_cov_inv)
        residual = np.subtract(z, pred_z)

        delta_x =  np.dot(kalman_gain, residual)
        delta_q = rotvec2quat(delta_x[6:9])
        self.X[0:6] += delta_x[0:6]
        self.X[6:10] = quat_multiply(delta_q, self.X[6:10])
        self.X[6:10] = self.X[6:10] / np.linalg.norm(self.X[6:10])

        self.P = self.P - np.dot(kalman_gain, np.dot(innovation_cov, kalman_gain.T))

    @staticmethod
    def _unscented_transform_F(sigmas: npt.NDArray[np.float64], Wm, Wc, X=None):
        # splitting sigma points up into vector states and quaternion states
        vector_sigmas = sigmas[:, 0:6]
        quat_sigmas = sigmas[:, 6:10]
        # small delta quaternions are made by multiplying the quaternion sigmas by the
        # inverse of the current quaternion state
        delta_quats = quat_multiply(quat_sigmas, quat_inv(X[6:10]))
        # delta quaternion rotations are converted into delta rotation vectors
        delta_rotvecs = quat2rotvec(delta_quats)
        # the mean of the rotation vector is calculated by multiplying each sigma by each weight
        # this mean cannot be calculated with quaternions directly due to the inability to sum
        # them.
        mean_delta_rotvec = np.dot(Wm, delta_rotvecs)
        # mean delta rotation vector is transformed back into a mean delta quaternion and
        # multiplied into the state to get the quaternion prediction
        mean_quat = quat_multiply(rotvec2quat(mean_delta_rotvec), X[6:10])
        # the vector portion of the sigma points are calculated normally
        vector_mean = np.dot(Wm, vector_sigmas)
        x_mean = np.concatenate([vector_mean, mean_quat])

        quat_covariance = (delta_rotvecs.T * Wc) @ delta_rotvecs
        vec_residual = vector_sigmas - vector_mean[np.newaxis, :]
        vec_covariance = (vec_residual.T * Wc) @  vec_residual
        
        # IMPORTANT: using block_diag means that there will NEVER be cross-covariances
        # between the vector components and quaternion components. This is for simplicity
        # but is not accurate, and should be later reconsidered if inadequate
        P_covariance = scipy.linalg.block_diag(vec_covariance, quat_covariance)
        return (x_mean, P_covariance)
    
    @staticmethod
    def _unscented_transform_H(sigmas: npt.NDArray[np.float64], Wm, Wc, noise_cov = None):
        
        x_mean = np.dot(Wm, sigmas)
        print(Wm)
        residual = sigmas - x_mean[np.newaxis, :]
        P_covariance = np.dot(residual.T, np.dot(np.diag(Wc), residual))

        if noise_cov is not None:
            P_covariance += noise_cov
        return (x_mean, P_covariance)

    def compute_process_sigmas(self, dt, Q, **F_args):
        sigmas = self._sigma_points_class.calculate_sigma_points(self.X, self.P, Q)
        for i, s in enumerate(sigmas):
            self._sigmas_f[i] = self.F(s, dt, F_args)


    def _calculate_cross_cov(self, x, z):
        P_cross_cov = np.zeros((self._sigmas_f.shape[1] - 1, self._sigmas_h.shape[1]))

        n = self._sigmas_f.shape[0]
        for i in range(n):
            dx_vector = np.subtract(self._sigmas_f[i][0:6], x[0:6])
            quat_sigmas_f = self._sigmas_f[i][6:10]
            quat_sigmas_f = quat_sigmas_f / np.linalg.norm(quat_sigmas_f)
            delta_quats = quat_multiply(quat_sigmas_f, quat_inv(x[6:10]))
            delta_rotvecs = quat2rotvec(delta_quats)
            dx = np.concatenate([dx_vector, delta_rotvecs])
            dz = np.subtract(self._sigmas_h[i], z)
            P_cross_cov += self._sigma_points_class.Wc[i] * np.outer(dx, dz)
        return P_cross_cov