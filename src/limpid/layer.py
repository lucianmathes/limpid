import numpy as np
from scipy import integrate, constants
from functools import lru_cache

from .implantation import ImplantationProfile
from .fitting import FitParameter, FitParameters


class Layer:

    def __init__(self, idx: int, precision: int, profile: ImplantationProfile or None, parameters: FitParameters,
                 thickness: float = np.inf, pos_aff=-1, temp: float = 293):
        self.idx = idx
        self.idx_str = str(idx)
        self.precision = precision  # floating point precision inn digits, used for numerical integration
        self.parameters = parameters

        # add parameters for each layer
        self.parameters.add(FitParameter(name='lineshape_' + self.idx_str, value=np.inf, vary=True, min=1E-15))
        # the diffusionlength_ is sqrt(D/mu), but often u = 1/diffusionlength is used
        self.parameters.add(FitParameter(name='diffusionlength_' + self.idx_str, value=70, vary=True, min=1E-15))
        # diffusioncoeff_ is only relevant if a significant amount of positrons are able to reach a layer boundary
        self.parameters.add(FitParameter(name='diffusioncoeff_' + self.idx_str, value=1, vary=False, min=1E-15))
        self.parameters.add(FitParameter(name='thickness_' + self.idx_str, value=thickness, vary=False, min=1E-15))

        self.pos_aff = pos_aff
        self.temperature = temp
        self.boltz_stat_factor = np.exp(-self.pos_aff / (constants.physical_constants["Boltzmann constant in eV/K"][0] * self.temperature))
        self.implantation_profile = profile

    def set_lineshape(self, value):
        self.parameters.change_value(name='lineshape_' + self.idx_str, value=value)

    def set_diffusioncoeff(self, value):
        self.parameters.change_value(name='diffusioncoeff_' + self.idx_str, value=value)

    def set_diffusionlength(self, value):
        self.parameters.change_value(name='diffusionlength_' + self.idx_str, value=value)

    def set_thickness(self, value):
        self.parameters.change_value(name='thickness_' + self.idx_str, value=value)

    def get_lineshape(self):
        lineshape = self.parameters['lineshape_' + self.idx_str].value
        return lineshape

    def get_diffusioncoeff(self):
        diffusioncoeff = self.parameters['diffusioncoeff_' + self.idx_str].value
        return diffusioncoeff

    def get_diffusionlength(self):
        diffusionlength = self.parameters['diffusionlength_' + self.idx_str].value
        return diffusionlength

    def get_thickness(self):
        thickness = self.parameters['thickness_' + self.idx_str].value
        return thickness

    def __call__(self, prev_implanted: np.ndarray, energies: np.ndarray) -> (np.ndarray, np.ndarray,
                                                                             np.ndarray, np.ndarray, np.ndarray):

        return self.implantation_profile.solve_first_diffusion_step(self.get_diffusionlength(), self.get_thickness(), self.precision, energies, prev_implanted)

    def get_rates_for_second_part_of_diffusion(self) -> (float, float):
        """
        Second diffusion step.
        Calculates diffsuion and annihilation probabilities for the diffusion
        process between the left and right layer boundary.
        """
        u = 1 / self.get_diffusionlength()
        thickness = self.get_thickness()
        diffusion_coeff = self.get_diffusioncoeff()
        exponential = u * thickness

        # diffusion
        if exponential >= 700:
            # exp(709) ~ 10^308 -> overflow
            diffusion = 0

        elif exponential >= 30:
            # exp(-30) ~ 10^-15 -> truncated, since negligible
            diffusion = 2 * u * diffusion_coeff * np.exp(-u * thickness)

        else:
            diffusion = 2 * u * diffusion_coeff * 1 / (np.exp(-u * thickness) * np.expm1(2 * u * thickness))

        # annihilation
        if exponential >= 30:
            # exp(-30) ~ 10^-15 -> truncated, since negligible
            annihilation = u * diffusion_coeff

        else:
            annihilation = u * diffusion_coeff * (np.exp(u * thickness) + np.exp(-u * thickness) - 2) / \
                           (np.exp(-u * thickness) * np.expm1(2 * u * thickness))

        return diffusion * self.boltz_stat_factor, annihilation * self.boltz_stat_factor
