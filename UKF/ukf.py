import numpy as np
import numpy.typing as npt

from UKF.sigma_points import SigmaPoints


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
        self.X, self.P = self._unscented_transform(
            sigmas = self._sigmas_f,
            Wm = self._sigma_points_class.Wm,
            Wc = self._sigma_points_class.Wc,
        )

    def update(self, z, **H_args):
        sigmas_h = []
        for s in self._sigmas_f:
            sigmas_h.append(self.H(s, **H_args))
        self._sigmas_h = np.atleast_2d(sigmas_h)

        pred_z, innovation_cov = self._unscented_transform(
            self._sigmas_h,
            self._sigma_points_class.Wm,
            self._sigma_points_class.Wc
            )
        innovation_cov_inv = np.linalg.inv(innovation_cov)
        P_cross_covariance = self._calculate_cross_cov(self.X, pred_z)
        kalman_gain = np.dot(P_cross_covariance, innovation_cov_inv)
        residual = np.subtract(z, pred_z)

        self.X = self.X + np.dot(kalman_gain, residual)
        self.P = self.P - np.dot(kalman_gain, np.dot(innovation_cov, kalman_gain.T))

    @staticmethod
    def _unscented_transform(sigmas: npt.NDArray[np.float64], Wm, Wc):
        
        x_mean = np.dot(Wm, sigmas)
        print(x_mean)
        residual = sigmas - x_mean[np.newaxis, :]
        P_covariance = np.dot(residual.T, np.dot(np.diag(Wc), residual))

        return (x_mean, P_covariance)

    def compute_process_sigmas(self, dt, Q, **F_args):
        sigmas = self._sigma_points_class.calculate_sigma_points(self.X, self.P, Q)
        for i, s in enumerate(sigmas):
            self._sigmas_f[i] = self.F(s, dt, F_args)

    def _calculate_cross_cov(self, x, z):
        P_cross_cov = np.zeros((self._sigmas_f.shape[1], self._sigmas_h.shape[1]))
        n = self._sigmas_f.shape[0]
        for i in range(n):
            dx = np.subtract(self._sigmas_f[i], x)
            dz = np.subtract(self._sigmas_h[i], z)
            P_cross_cov += self._sigma_points_class.Wc[i] * np.outer(dx, dz)
        return P_cross_cov