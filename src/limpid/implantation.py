from src.limpid.implantation_profiles import *
import numpy as np
from functools import lru_cache


class ImplantationProfile(object):

    def __init__(self, func, params):
        self.func = func
        self.params = params

    def __call__(self, z, energy):
        return self.func(z, energy, *self.params)

    def set_params(self, params):
        self.params = params

    def get_offset(self, implanted, energy):
        pass


class MakhovProfile(ImplantationProfile):

    def __init__(self, params):
        super().__init__(makhov_profile, params)

    def get_offset(self, implanted, energy):
        rho, a, n, m = self.params
        return makhov_offset(rho, a, n, m, implanted, energy)
