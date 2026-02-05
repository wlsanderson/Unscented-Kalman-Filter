"""
Based on FilterPy's MerweScaledSigmaPoints class, with changes made to support quaternions from
Edgar Kraft's paper on quaternion UKF's.
"""
import numpy as np
import numpy.typing as npt
import quaternion
from UKF.ukf_functions import print_c_array

class SigmaPoints:
    __slots__ = (
        "Wm",
        "Wc",
        "_n",
        "_alpha",
        "_beta",
        "_kappa",
        "_quat_idx",
        "_rotvec_idx",
        "_vec_idx",
    )

    def __init__(self, n, alpha, beta, kappa):
        # n is the dimension of the state vector minus one due to quaternion being represented
        # as 3 components in the covariance
        self._n = n
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

        # Sigma point weights for means and covariances
        self.Wm: npt.NDArray = None
        self.Wc: npt.NDArray = None

        self._compute_weights()
        self._quat_idx = slice(n - 3, n + 1)
        self._rotvec_idx = slice(n - 3, n)
        self._vec_idx = slice(0, n - 3)

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
        lambda_ = self._compute_lambda_scaling_parameter()
        # this ensures numerical stability in the cholesky square root by making P symmetric
        P = 0.5 * (P + P.T)

        # According to Edgar Kraft's paper on quaternion UKF's, we need to apply the process noise
        # before generating the sigma points, as opposed to a standard UKF where the noise is added
        # after propagating the sigma points through the state transition function.
        try:
            # nugget = 1e-6 * np.trace(P + Q, dtype=np.float32) / self._n
            # P_regularized = P + np.eye(self._n, dtype=np.float32) * nugget
            scaled_cholesky_sqrt = np.linalg.cholesky((lambda_ + self._n) * (P + Q))
        except:
            print(P)
            print(np.linalg.eig(P + Q))
            raise Exception("matrix not positive definite")

        # array to hold the sigma points, first row is the mean state with no perturbation applied.
        sigmas = np.float32(np.zeros([self.num_sigmas(), state_dim]))
        sigmas[0] = X

        # generate the sigma points, the first n sigma points are X + scaled_sqrt, the next n are
        # X - scaled_sqrt.
        for i in range(self._n):
            sigmas[i+1][self._vec_idx] = X[self._vec_idx] + scaled_cholesky_sqrt[:, i][self._vec_idx]
            sigmas[self._n+i+1][self._vec_idx] = X[self._vec_idx] - scaled_cholesky_sqrt[:, i][self._vec_idx]

            # to handle the quaternion part, we convert the rotation vector to a quaternion
            # and apply it to the mean quaternion via quaternion multiplication.
            rotvec_sqrt = scaled_cholesky_sqrt[:, i][self._rotvec_idx]
            quat_sigma = quaternion.from_rotation_vector(rotvec_sqrt)
            quat_X = quaternion.from_float_array(X[self._quat_idx])
            # instead of adding/subtracting, we multiply quaternions to "add", and multiply by
            # the conjucate to "subtract".
            sigmas[i+1][self._quat_idx] = quaternion.as_float_array(quat_X * quat_sigma)
            sigmas[self._n+i+1][self._quat_idx] = quaternion.as_float_array(quat_X * quat_sigma.conjugate())
        return sigmas

    def num_sigmas(self) -> int:
        """ Number of sigma points used by the filter. """
        return 2 * self._n + 1
    
    def _compute_weights(self) -> None:
        """
        Computes the weights for the sigma points. These weights will not change during runtime
        of the filter unless the size of the state vector changes, or alpha/beta/kappa change.
        This is only computed once during initialization.
        Reference: https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf (eq 15)
        """
        lambda_ = self._compute_lambda_scaling_parameter()
        weight_component = 1 / (2*(self._n + lambda_))
        
        # weights for mean and covariance, note that the mean weights must sum to one, but the
        # covariance weights do not have to. Negative weights are also possible.
        self.Wc = np.full(self.num_sigmas(), weight_component)
        self.Wm = np.full(self.num_sigmas(), weight_component)
        self.Wm[0] = lambda_ / (self._n + lambda_)
        self.Wc[0] = lambda_ / (self._n + lambda_) + (1 - self._alpha**2 + self._beta)

    def _compute_lambda_scaling_parameter(self) -> np.float32:
        """ Computes the scaling parameter lambda for the sigma points. """
        return np.float32(self._alpha**2 * (self._n + self._kappa) - self._n)

