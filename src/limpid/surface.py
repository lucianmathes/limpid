import numpy as np

from .layer import Layer
from .fitting import FitParameters


class Surface(Layer):

    def __init__(self, idx: int, precision: int, parameters: FitParameters, pos_aff=-20, temp: float = 293):
        super().__init__(idx, precision, None, parameters, temp=temp, pos_aff=pos_aff)

        # removes fitting parameters from the parent class Layer, which are not needed for the Surface class
        self.parameters.remove('diffusioncoeff_' + self.idx_str)
        self.parameters.remove('diffusionlength_' + self.idx_str)
        self.parameters.remove('thickness_' + self.idx_str)

    def __call__(self, prev_implanted: np.ndarray, energies: np.ndarray) -> (np.ndarray, np.ndarray,
                                                                             np.ndarray, np.ndarray):
        raise AttributeError("This object is not callable.")

    def concentration_left(self, z, u, thickness):
        raise AttributeError("This object has no method 'concentration_left'.")

    def concentration_right(self, z, u, thickness):
        raise AttributeError("This object has no method 'concentration_right'.")
