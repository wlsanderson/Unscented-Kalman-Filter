from ukf import UKF
from sigma_points import SigmaPoints
from constants import STATE_DIM, ALPHA, BETA, KAPPA, MEASUREMENT_DIM, INITIAL_STATE_ESTIMATE, INITIAL_STATE_COV
from ukf_functions import measurement_function

class Context:
    def __init__(self):
        sigma_points = SigmaPoints(
            n = STATE_DIM,
            alpha = ALPHA,
            beta = BETA,
            kappa = KAPPA,
        )

        self.ukf = UKF(
            dim_x = STATE_DIM,
            dim_z = MEASUREMENT_DIM,
            points = sigma_points,
        )
        self.initialize_filter_settings()
    
    def initialize_filter_settings(self):
        self.ukf.X = INITIAL_STATE_ESTIMATE.copy()
        self.ukf.P = INITIAL_STATE_COV.copy()
        self.ukf.H = measurement_function
