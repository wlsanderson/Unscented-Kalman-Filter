"""
Based on FilterPy's MerweScaledSigmaPoints class, with changes made to support quaternions from
Edgar Kraft's paper on quaternion UKF's.
"""
import numpy as np
import numpy.typing as npt
import quaternion

class SigmaPoints:
    __slots__ = (
        "_n",
        "_alpha",
        "_beta",
        "_kappa",
        "Wm",
        "Wc",
        "_quat_idx",
        "_rotvec_idx",
        "_vec_idx",

    )

    def __init__(self, n, alpha, beta, kappa):
        self._n = n
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

        self.Wm: npt.NDArray = None
        self.Wc: npt.NDArray = None

        self._compute_weights()
        self._quat_idx = slice(n - 3, n + 1)
        self._rotvec_idx = slice(n - 3, n)
        self._vec_idx = slice(0, n - 3)



    def _compute_weights(self) -> None:
        lambda_ = self._alpha**2 * (self._n + self._kappa) - self._n
        c = 0.5 / (self._n + lambda_)
        self.Wc = np.full(2 * self._n + 1, c)
        self.Wm = np.full(2 * self._n + 1, c)
        self.Wc[0] = lambda_ / (self._n + lambda_) + (1 - self._alpha**2 + self._beta)
        self.Wm[0] = lambda_ / (self._n + lambda_)

    def calculate_sigma_points(self, X, P, Q) -> npt.NDArray:
        """
        Calculates sigma points that will be evaluated in the state transition function. Process
        noise willbe included before the state transition function. Because of this, the scaled
        choleskly square root will be one dimension smaller than the sigma points because the noise
        matrix for quaternions is 3 components, not 4.
        """
        P = np.atleast_2d(P)
        Q = np.atleast_2d(Q)
        state_dim = len(X)
        lambda_ = self._alpha**2 * (self._n + self._kappa) - self._n
        P = 0.5 * (P + P.T)

        scaled_cholesky_sqrt = np.linalg.cholesky((lambda_ + self._n)*(P + Q), upper=True)
        
        sigmas = np.zeros([2 * self._n + 1, state_dim])
        sigmas[0] = X

        for i in range(self._n):
            sigmas[i+1][self._vec_idx] = np.subtract(X[self._vec_idx], -scaled_cholesky_sqrt[i][self._vec_idx])
            sigmas[self._n+i+1][self._vec_idx] = np.subtract(X[self._vec_idx], scaled_cholesky_sqrt[i][self._vec_idx])

            rotvec_sqrt = scaled_cholesky_sqrt[i][6:9]
            quat_sigma = quaternion.from_rotation_vector(rotvec_sqrt).normalized()
            quat_X = quaternion.from_float_array(X[self._quat_idx]).normalized()
            sigmas[i+1][self._quat_idx] = quaternion.as_float_array(quat_X * quat_sigma)
            sigmas[self._n+i+1][self._quat_idx] = quaternion.as_float_array(quat_X * quat_sigma.conjugate())
        return sigmas

    def num_sigmas(self) -> int:
        return 2 * self._n + 1
    
