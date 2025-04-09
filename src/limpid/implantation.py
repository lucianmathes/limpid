from scipy import integrate
from functools import cache
from numba import njit, float64
import numpy as np
import math
import os
from .c_libraries.compile_c_libs import compile_cmakhov_lib_gcc
from ctypes import *

PATH_TO_CMAKHOV_LIBRARY = os.path.expanduser("~/.limpid/makhov.so")

class ImplantationProfile():
    """A parent class for all types of positron implantation profile.

    Attributes:
      model_function: The continuous positron implantation model function.
      parameters: Model specific parameters for the given type of material.
    """

    def __init__(self, model_function, parameters):
        """Initializes the implantation profile.

        Args:
          model_function: The positron implantation model function. Must be
            continuous.
          parameters: Model specific parameters for the given type of material.
        """

        self.model_function = model_function
        self.parameters = parameters

    def __call__(self, z, energy):
        """Evaluates the implantation profile at the given depth and energy."""

        return self.model_function(z, energy, *self.parameters)

    def get_depth(self, implanted, energy):
        """Calculates the depth for a given fraction of positrons implanted."""

        # TODO for each ImplantationProfile independently, or one could
        # implement a general root-finding algorithm for this parent class
        pass


class MakhovProfile(ImplantationProfile):
    """Wrapper for the Makhov implantation profile model function.

    Attributes:
      model_function: The implantation model function (for electrons) defined
        in [Makhov]_.
      parameters: Model specific parameters for the given type of material.

    References:
      .. [Makhov] A.F. Makhov, "The penetration of electrons into solids. 2.
                  The distribution of electrons in Depth", Sov. Pys. Solid
                  State, Vol. 2, Number 9, pp1942-1944, 1960.

    """

    def __init__(self, parameters):
        super().__init__(self.makhov_profile, parameters)

    @staticmethod
    @cache
    @njit(float64(float64, float64, float64, float64, float64, float64), cache=True)
    def makhov_profile(z, e, rho, a, n, m):
        z_avg = a / rho * e ** n * 10
        z_0 = z_avg / math.gamma(1 / m + 1)

        return (m * z ** (m - 1)) / (z_0 ** m) * np.exp(-(z / z_0) ** m)

    @staticmethod
    @cache
    @njit(float64(float64, float64, float64, float64, float64, float64), cache=True)
    def get_depth(implanted, energy, rho, a, n, m):
        z_avg = a / rho * energy ** n * 10
        z_0 = z_avg / math.gamma(1 / m + 1)

        # computer precision
        if implanted > 1 - 1E-15:
            implanted = 1 - 1E-15

        return z_0 * np.power(-np.log(1 - implanted), 1 / m)


class CMakhovProfile(ImplantationProfile):

    def __init__(self, parameters):
        super().__init__(makhov_profile, parameters)

    def get_depth(self, implanted, energy):
        rho, a, n, m = self.parameters
        return makhov_depth(rho, a, n, m, implanted, energy)

    def solve_first_diffusion_step(self, diffusionlength, thickness, precision, energies, prev_implanted):
        u = 1 / diffusionlength

        prec_exp = 10 ** (-precision)

        if not os.path.exists(PATH_TO_CMAKHOV_LIBRARY):
            compile_cmakhov_lib_gcc()

        liblimpid = CDLL(f"{PATH_TO_CMAKHOV_LIBRARY}")
        ARRAY_POINTER = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")

        liblimpid.makhov_integration_func.argtypes = [c_int, ARRAY_POINTER, ARRAY_POINTER, ARRAY_POINTER, ARRAY_POINTER, ARRAY_POINTER,
                                                      ARRAY_POINTER, ARRAY_POINTER, c_double, c_double, c_double, c_double,
                                                      c_double, c_double, c_double]

        num_of_e = len(energies)

        energies = np.asarray(energies)
        c_left = np.zeros_like(energies)
        c_right = np.zeros_like(energies)
        c_ann = np.zeros_like(energies)
        c_implanted = np.zeros_like(energies)
        offsets = np.zeros_like(energies)

        rho, a, n, m = self.parameters

        liblimpid.makhov_integration_func(num_of_e, c_left, c_right, c_ann, c_implanted, offsets, prev_implanted, energies,
                                          thickness, rho, a, n, m, u, prec_exp)

        return c_left, c_right, c_ann, c_implanted, offsets
