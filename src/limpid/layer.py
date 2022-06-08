import numpy as np
from scipy import integrate
from src.limpid.implantation import ImplantationProfile
from src.limpid.fitting import FitParameter, FitParameters
from functools import lru_cache


class Layer:

    def __init__(self, idx: int, profile: ImplantationProfile or None, parameters: FitParameters):
        self.idx = idx
        self.idx_str = str(idx)
        self.parameters = parameters

        # add parameters for each layer
        self.parameters.add(FitParameter(name='lineshape_' + self.idx_str, value=0.50, vary=True, min=1E-15))
        self.parameters.add(FitParameter(name='diffusioncoeff_' + self.idx_str, value=100, vary=True, min=1E-15))
        self.parameters.add(FitParameter(name='u_' + self.idx_str, value=0.02, vary=True, min=1E-15))
        self.parameters.add(FitParameter(name='thickness_' + self.idx_str, value=np.inf, vary=False, min=1E-15))

        self.pos_aff = 1

        self.implantation_profile = profile

    def set_lineshape(self, value):
        self.parameters.change_value(name='lineshape_' + self.idx_str, value=value)

    def set_diffusioncoeff(self, value):
        self.parameters.change_value(name='diffusioncoeff_' + self.idx_str, value=value)

    def set_u(self, value):
        self.parameters.change_value(name='u_' + self.idx_str, value=value)

    def set_thickness(self, value):
        self.parameters.change_value(name='thickness_' + self.idx_str, value=value)

    def get_lineshape(self):
        lineshape = self.parameters['lineshape_' + self.idx_str].value
        return lineshape

    def get_diffusioncoeff(self):
        diffusioncoeff = self.parameters['diffusioncoeff_' + self.idx_str].value
        return diffusioncoeff

    def get_u(self):
        annihilationcoeff = self.parameters['u_' + self.idx_str].value
        return annihilationcoeff

    def get_thickness(self):
        thickness = self.parameters['thickness_' + self.idx_str].value
        return thickness

    def __call__(self, prev_implanted: np.ndarray, energies: np.ndarray) -> (np.ndarray, np.ndarray,
                                                                             np.ndarray, np.ndarray):
        u = self.get_u()
        thickness = self.get_thickness()

        def integral(f, e):
            return integrate.quad(f, 0, thickness, args=(e,), epsabs=1e-8, epsrel=1e-8)[0]

        c_left = np.zeros_like(energies)
        c_right = np.zeros_like(energies)
        c_ann = np.zeros_like(energies)
        c_implanted = np.zeros_like(energies)

        for i, energy in enumerate(energies):
            offset = self.implantation_profile.get_offset(prev_implanted[i], energy)
            c_left[i] = integral(lambda z, e: self.implantation_profile(z + offset, e) *
                                              self.concentration_left(z, u, thickness), energy)
            c_right[i] = integral(lambda z, e: self.implantation_profile(z + offset, e) *
                                               self.concentration_right(z, u, thickness), energy)
            c_implanted[i] = integral(lambda z, e: self.implantation_profile(z + offset, e), energy)
            c_ann[i] = c_implanted[i] - c_left[i] - c_right[i]

        return c_left, c_right, c_ann, c_implanted

    @lru_cache(maxsize=None)
    def concentration_left(self, z, u, thickness):
        return (np.exp(-u * z) - np.exp(u * (z - 2 * thickness))) / (1 - np.exp(-2 * u * thickness))

    @lru_cache(maxsize=None)
    def concentration_right(self, z, u, thickness):
        return (np.exp(u * z) - np.exp(- u * z)) / (np.exp(u * thickness) - np.exp(- u * thickness))

    def get_rates_for_second_part_of_diffusion(self) -> (float, float):
        """
        Second diffusion step.
        Calculates diffsuion and annihilation probabilities for the diffusion
        process between the left and right layer boundary.
        """
        u = self.get_u()
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
            annihilation = u * diffusion_coeff * (np.exp(u * diffusion_coeff) + np.exp(-u * diffusion_coeff) - 2) / \
                           (np.exp(-u * thickness) * np.expm1(2 * u * thickness))

        return diffusion, annihilation
