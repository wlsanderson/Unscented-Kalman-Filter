"""Largely copied from FilterPy's MerweScaledSigmaPoints class"""
import numpy as np
import numpy.typing as npt

class SigmaPoints:
    __slots__ = (
        "_n",
        "_alpha",
        "_beta",
        "_kappa",
        "Wm",
        "Wc",
    )

    def __init__(self, n, alpha, beta, kappa):
        self._n = n
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

        self.Wm: npt.NDArray = None
        self.Wc: npt.NDArray = None

        self._compute_weights()

        


    def _compute_weights(self) -> None:
        lambda_ = self._alpha**2 * (self._n + self._kappa) - self._n
        c = 0.5 / (self._n + lambda_)
        self.Wc = np.full(2 * self._n + 1, c)
        self.Wm = np.full(2 * self._n + 1, c)
        self.Wc[0] = lambda_ / (self._n + lambda_) + (1 - self._alpha**2 + self._beta)
        self.Wm[0] = lambda_ / (self._n + lambda_)

    def calculate_sigma_points(self, X, P) -> npt.NDArray:
        if self._n != np.size(X):
            raise ValueError("expected size(x) {}, but size is {}".format(self.n, np.size(X)))
        P = np.atleast_2d(P)
        n = len(X)
        lambda_ = self._alpha**2 * (n + self._kappa) - n
        scaled_cholesky_sqrt = np.linalg.cholesky((lambda_ + n)*(P), upper=True)
        sigmas = np.zeros([2 * n + 1, n])
        sigmas[0] = X
        for i in range(n):
            sigmas[i+1] = np.subtract(X, -scaled_cholesky_sqrt[i])
            sigmas[n+i+1] = np.subtract(X, scaled_cholesky_sqrt[i])
        return sigmas

    def num_sigmas(self) -> int:
        return 2 * self._n + 1
    
