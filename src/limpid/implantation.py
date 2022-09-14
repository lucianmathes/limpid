from scipy import integrate
from functools import lru_cache
from numba import njit, float64
import numpy as np
import math
import os
from .c_libraries.compile_c_libs import compile_cmakhov_lib_gcc
from ctypes import *

PATH_TO_CMAKHOV_LIBRARY = os.path.expanduser("~/.limpid/makhov.so")

class ImplantationProfile(object):

    def __init__(self, func, params):
        self.func = func
        self.params = params

    def __call__(self, z, energy):
        return self.func(z, energy, *self.params)

    def set_params(self, params):
        self.params = params

    def get_depth(self, implanted, energy):
        # TODO for each ImplantationProfile independently, or one could implement a general root-finding algorithm
        #  for this parent class
        pass

    def solve_first_diffusion_step(self, diffusionlength, thickness, precision, energies, prev_implanted):
        u = 1 / diffusionlength

        prec_exp = 10 ** (-precision)

        def integral(f, e):
            return integrate.quad(f, 0, thickness, args=(e,), epsabs=prec_exp, epsrel=prec_exp)[0]

        c_left = np.zeros_like(energies)
        c_right = np.zeros_like(energies)
        c_ann = np.zeros_like(energies)
        c_implanted = np.zeros_like(energies)
        offsets = np.zeros_like(energies)

        for i, energy in enumerate(energies):
            offset = self.get_depth(prev_implanted[i], energy)
            offsets[i] = offset
            c_left[i] = integral(lambda z, e: self.__call__(z + offset, e) *
                                              self.concentration_left(z, u, thickness), energy)
            c_right[i] = integral(lambda z, e: self.__call__(z + offset, e) *
                                               self.concentration_right(z, u, thickness), energy)
            c_implanted[i] = integral(lambda z, e: self.__call__(z + offset, e), energy)
            c_ann[i] = c_implanted[i] - c_left[i] - c_right[i]

        return c_left, c_right, c_ann, c_implanted, offsets

    @lru_cache(maxsize=None)
    def concentration_left(self, z, u, thickness):
        return (np.exp(-u * z) - np.exp(u * (z - 2 * thickness))) / (1 - np.exp(-2 * u * thickness))

    @lru_cache(maxsize=None)
    def concentration_right(self, z, u, thickness):
        return (np.exp(u * z) - np.exp(- u * z)) / (np.exp(u * thickness) - np.exp(- u * thickness))


class MakhovProfile(ImplantationProfile):

    def __init__(self, params):
        super().__init__(makhov_profile, params)

    def get_depth(self, implanted, energy):
        rho, a, n, m = self.params
        return makhov_depth(rho, a, n, m, implanted, energy)


class CMakhovProfile(ImplantationProfile):

    def __init__(self, params):
        super().__init__(makhov_profile, params)

    def get_depth(self, implanted, energy):
        rho, a, n, m = self.params
        return makhov_depth(rho, a, n, m, implanted, energy)

    def solve_first_diffusion_step(self, diffusionlength, thickness, precision, energies, prev_implanted):
        u = 1 / diffusionlength

        prec_exp = 10 ** (-precision)

        if not os.path.exists(PATH_TO_CMAKHOV_LIBRARY):
            compile_cmakhov_lib_gcc()

        liblimpid = CDLL(f"{PATH_TO_CMAKHOV_LIBRARY}")
        ARRAY_POINTER = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")

        liblimpid.makhov_integration_func.argtypes = [c_int, ARRAY_POINTER, ARRAY_POINTER, ARRAY_POINTER, ARRAY_POINTER,
                                                      ARRAY_POINTER,
                                                      ARRAY_POINTER, ARRAY_POINTER, c_double, c_double, c_double,
                                                      c_double,
                                                      c_double, c_double, c_double]

        num_of_e = len(energies)

        energies = np.asarray(energies)
        c_left = np.zeros_like(energies)
        c_right = np.zeros_like(energies)
        c_ann = np.zeros_like(energies)
        c_implanted = np.zeros_like(energies)
        offsets = np.zeros_like(energies)

        rho, a, n, m = self.params

        liblimpid.makhov_integration_func(num_of_e, c_left, c_right, c_ann, c_implanted, offsets, prev_implanted,
                                          energies,
                                          thickness, rho, a, n, m, u, prec_exp)

        return c_left, c_right, c_ann, c_implanted, offsets


@lru_cache(maxsize=None)
@njit(float64(float64, float64, float64, float64, float64, float64), cache=True)
def makhov_profile(z, e, rho, a, n, m):
    z_avg = a / rho * e ** n * 10
    z_0 = z_avg / math.gamma(1 / m + 1)

    return (m * z ** (m - 1)) / (z_0 ** m) * np.exp(-(z / z_0) ** m)


@lru_cache(maxsize=None)
@njit(float64(float64, float64, float64, float64, float64, float64), cache=True)
def makhov_depth(rho, a, n, m, implanted, energy):
    z_avg = a / rho * energy ** n * 10
    z_0 = z_avg / math.gamma(1 / m + 1)

    # deals with infinite offsets, the error made by this implementation is negligible
    if implanted > 1 - 1E-15:
        implanted = 1 - 1E-15

    return z_0 * np.power(-np.log(1 - implanted), 1 / m)
