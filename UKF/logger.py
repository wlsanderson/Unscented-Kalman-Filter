import numpy as np
import numpy.typing as npt

class Logger:

    __slots__ = (
        "X_data",
        "P_data",
        "timestamps",
    )

    def __init__(self):
        self.X_data: list = None
        self.P_data: list = None
        self.timestamps: list = None
