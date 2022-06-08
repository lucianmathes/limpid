import numpy as np
from src.limpid.layer import Layer
from src.limpid.fitting import FitParameter, FitParameters


class Surface(Layer):

    def __init__(self, idx: int, parameters: FitParameters):
        super().__init__(idx, None, parameters)

        # removes fitting parameters from the parent class Layer, which are not needed for the Surface class
        self.parameters.remove('diffusioncoeff_' + self.idx_str)
        self.parameters.remove('u_' + self.idx_str)
        self.parameters.remove('thickness_' + self.idx_str)

    def __call__(self, prev_implanted: np.ndarray, energies: np.ndarray) -> (np.ndarray, np.ndarray,
                                                                             np.ndarray, np.ndarray):
        raise AttributeError("This object is not callable.")

    def concentration_left(self, z, u, thickness):
        raise AttributeError("This object has no method 'concentration_left'.")

    def concentration_right(self, z, u, thickness):
        raise AttributeError("This object has no method 'concentration_right'.")