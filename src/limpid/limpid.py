import time
import copy
from functools import cache

import lmfit
import numpy as np
from scipy import integrate

from .implantation import MakhovProfile


BOLTZMANN_CONSTANT = 8.617333262e-05 # in eV/K
_layers_created = 0


class Layer:
    """A digital representation of one layer of your sample.

    The Layer object stores information about the layer material and positrons
    implanted in it.

    Attributes:
      index: Index of the layer. Will be updated to represent ordering when
        when the layer becomes part of a Sample object.
      density: The density of the layer material.
      makhov_parameters: Parameters A, n, m of the layer material used in the
        Makhov implantation profile.
      precision: An integer count of decimals used for computation.
      implantation_profile: Instance of limpid.implantation.ImplantationProfile
        or any child of it. Contains the model function and utilities to
        compute multi-layer implantation profiles.
      parameters: lmfit.Parameters instance containing variables for the fit.
        Will be a shared instance, as soon as the Layer becomes part of a
        Sample object.
      thickness: Thickness of the layer in nm.
      lineshape: Lineshape (e.g., S or W) value of the layer material.
      diffusion_length: Positron diffusion length in the layer.
      diffusion_coefficient: Positron diffusion coefficient for the layer.
      positron_affinity: Positron affinity of the layer material.
      temperature: Temperature of the layer in K.
    """

    def __init__(
        self,
        density: float,
        makhov_parameters: tuple[float, float, float],
        name: str|None = None,
        thickness: float = np.inf,
        lineshape: float = np.inf,
        diffusion_length: float = 70,
        diffusion_coefficient: float = 1,
        positron_affinity: float = -1,
        precision: int = 8,
        index: int|None = None
    ):
        """Initializes the layer based on density and implantation profile.

        Args:
          density: The density of the layer material.
          makhov_parameters: Parameters A, n, m of the layer material used in
            the Makhov implantation profile. Do not confuse A for A_1/2!
          thickness: Thickness of the layer in nm.
          lineshape: Lineshape (e.g., S or W) value of the layer material.
          diffusion_length: Positron diffusion length in the layer.
          diffusion_coefficient: Positron diffusion coefficient for the layer.
          positron_affinity: Positron affinity of the layer material.
          precision: Defines the decimals used for computation, i.e., error
            tolerance of the implantation profile integral and fit tolerance.
            Default is 8, resulting in a tolerance of 10^(-8).
          index: Index of the layer. Will be updated to represent the order of
            layers when Layer becomes part of a Sample object.
        """

        global _layers_created
        _layers_created += 1

        if index is None:
            self.index = _layers_created
        else:
            self.index = index

        if name is None:
            self.name = f'Layer {self.index}'
        else:
            self.name = name

        self.density = density
        self.makhov_parameters = makhov_parameters
        self.precision = precision  # floating point precision in digits
        self.implantation_profile = None

        self.parameters = lmfit.Parameters()
        self.parameters.add(f'thickness_{self.index}', value=thickness,
                            vary=False, min=1E-15)
        self.parameters.add(f'lineshape_{self.index}', value=lineshape,
                            vary=True, min=1E-15)
        self.parameters.add(f'diffusion_length_{self.index}',
                            value=diffusion_length, vary=True, min=1E-15)
        self.parameters.add(f'diffusion_coefficient_{self.index}',
                            value=diffusion_coefficient, vary=False, min=1E-15)

        if positron_affinity > 0:
            wrn = ('The positron affinity must be negative. Using default '
                   'value of -1.')
            print(wrn)
            positron_affinity = -1

        self.positron_affinity = positron_affinity
        self.temperature = 293

    @property
    def thickness(self):
        return self.parameters[f'thickness_{self.index}'].value

    @thickness.setter
    def thickness(self, value):
        self.parameters[f'thickness_{self.index}'].value = value

    @property
    def lineshape(self):
        return self.parameters[f'lineshape_{self.index}'].value

    @lineshape.setter
    def lineshape(self, value):
        self.parameters[f'lineshape_{self.index}'].value = value

    @property
    def diffusion_length(self):
        return self.parameters[f'diffusion_length_{self.index}'].value

    @diffusion_length.setter
    def diffusion_length(self, value):
        self.parameters[f'diffusion_length_{self.index}'].value = value

    @property
    def diffusion_coefficient(self):
        return self.parameters[f'diffusion_coefficient_{self.index}'].value

    @diffusion_coefficient.setter
    def diffusioncoeff(self, value):
        self.parameters[f'diffusion_coefficient_{self.index}'].value = value

    def solve_first_diffusion_step(
        self,
        implantation_energies: np.ndarray,
        previously_implanted: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve the first diffusion step.

        Simulate the positron diffusion to layer boundaries starting from the
        implantation profile.

        Args:
          implantation_energies: Numpy array containing all positron
            implantation energies in keV.
          previously_implanted: Numpy array containing the fraction of
            positrons implanted before this layer.

        Returns:
          Five numpy arrays, i.e.:

          c_left, c_right, c_annihilated, c_implanted, offsets

          They contain the fraction of positrons that diffused to the left and
          right layer boundaries, the fraction of positrons that annihilated
          inside the layer, the fraction of positrons implanted into the layer
          and the offsets for the implantation profile.
        """

        prec_exp = 10 ** (-self.precision)

        @cache
        def concentration_left(z, thickness, u):
            if u * thickness > 100:
                # avoid overflow in np.sinh
                cl = np.exp(-u * z)
            else:
                cl = np.sinh(u * (thickness - z)) / np.sinh(u * thickness)
            return cl

        @cache
        def concentration_right(z, thickness, u):
            if u * thickness > 100:
                cr = np.exp(-u * (thickness - z))
            else:
                cr = np.sinh(u * z) / np.sinh(u * thickness)
            return cr

        def integral(f, max_depth):
            return integrate.quad(f, 0, min(max_depth, self.thickness),
                                  epsabs=prec_exp, epsrel=prec_exp)[0]

        # Offset of the implantation profile. Depth until which the integral is
        # equal to previously_implanted, if the entire sample was made out of
        # this layer's material.
        offsets = np.zeros_like(implantation_energies)
        c_left = np.zeros_like(implantation_energies)
        c_right = np.zeros_like(implantation_energies)
        c_annihilated = np.zeros_like(implantation_energies)
        c_implanted = np.zeros_like(implantation_energies)

        for i, energy in enumerate(implantation_energies):
            offset = self.implantation_profile.get_depth(
                             previously_implanted[i], energy,
                             *self.implantation_profile.parameters)
            max_depth = self.implantation_profile.get_depth(
                             1, energy, *self.implantation_profile.parameters)
            offsets[i] = offset
            c_left[i] = integral(lambda z: self.implantation_profile(z + offset, energy) *
                    concentration_left(z, self.thickness, 1/self.diffusion_length), max_depth)
            c_right[i] = integral(lambda z: self.implantation_profile(z + offset, energy) *
                    concentration_right(z, self.thickness, 1/self.diffusion_length), max_depth)
            c_implanted[i] = integral(lambda z: self.implantation_profile(z + offset, energy), max_depth)
            c_annihilated[i] = c_implanted[i] - c_left[i] - c_right[i]

        return c_left, c_right, c_annihilated, c_implanted, offsets

    def get_rates_for_second_part_of_diffusion(self) -> tuple[np.ndarray, np.ndarray]:
        """Perform second computational step of the diffusion simulation.

        Calculates diffusion and annihilation probabilities for the diffusion
        process between the left and right layer boundary.

        Returns:
          Two numpy arrays, corresponding to the diffusion and annihilation
          rates used for the second part of the simulation.
        """

        exponential = self.thickness / self.diffusion_length
        boltzmann_factor = np.exp(-self.positron_affinity
                            / (BOLTZMANN_CONSTANT * self.temperature))
        # The Boltzmann factor describes density. We need flux.
        boltzmann_factor *= self.diffusion_coefficient / self.diffusion_length

        # diffusion
        if exponential >= 700:
            # exp(709) ~ 10^308 -> overflow
            diffusion = 0

        elif exponential >= 30:
            # exp(-30) ~ 10^-15 -> truncated, since negligible
            diffusion = 2 * np.exp(- exponential) * boltzmann_factor

        else:
            diffusion = 1 / np.cosh(exponential) * boltzmann_factor

        # annihilation
        annihilation = boltzmann_factor - diffusion

        return diffusion, annihilation


class Surface:
    """A digital representation of the surface of your sample.

    The Surface object is almost only used internally by the limpid algorithm.
    It resembles an attractive potential, that drains positrons from the top
    layer of the sample.

    Attributes:
      positron_affinity: Positron affinity of the surface.
      temperature: Temperature of the surface in K.
      parameters: lmfit.Parameters instance containing variables for the fit.
        Usually a shared instance, with all layers of the Sample object.
      precision: An integer count of decimals used for computation.
      index: Index of the surface. Usually zero.
    """

    def __init__(
        self,
        positron_affinity: float = -12,
        parameters: lmfit.Parameters|None = None,
        precision: int = 8,
        index: int = 0
    ):
        """Initializes the surface.

        Args:
          positron_affinity: Positron affinity of the surface. Should be
            smaller than the value of the topmost layer to act as a drain.
          parameters: lmfit.Parameters instance to add variables.
          precision: Defines the decimals used for computation, i.e., error
            tolerance of the implantation profile integral and fit tolerance.
            Default is 8, resulting in a tolerance of 10^(-8).
          index: Index of the surface. Usually zero.
        """

        self.positron_affinity = positron_affinity
        self.temperature = 293
        self.parameters = parameters
        self.precision = precision
        self.index = index

    @property
    def lineshape(self):
        return self.parameters[f'lineshape_{self.index}'].value

    @lineshape.setter
    def lineshape(self, value):
        self.parameters[f'lineshape_{self.index}'].value = value


class Sample:
    """A digital representation of your sample.

    The Sample class makes the LIMPID core functionalities to model
    positron implantation, diffusion and annihilation in your sample
    accessible. It is constructed by one or multiple Layer objects and
    contains all information necessary for the simulations.

    Attributes:
      layers: A list of Layer objects representing the sample. Ordered from
        surface to bulk.
      surface: A Surface object at the top of the sample.
      name: A string containing the name of the sample.
      implantation_model: A string containing the model used for
        positron implantation. Currently only 'makhov' is supported.
      epithermal_correction: A boolean indicating if a correction for
        epithermal positrons is applied.
      temperature: A float containing the sample temperature in K.
      precision: An integer count of decimals used for computation.
      parameters: An lmfit.Parameters object containing all (fixed and
        varied) fit parameters.
      fit_status: An integer representing the fit status. After fitting, its
        value depends on the underlying solver. Initially -2, positive in
        case of success.
    """

    def __init__(
        self,
        layers: tuple[Layer, ...]|Layer,
        name: str = "",
        implantation_model: str = "makhov",
        epithermal_correction: bool = False,
        temperature: float = 293,
        precision: int = 8,
    ):
        """Initializes the sample based on layers.

        Args:
          layers: A single Layer object or list of Layer objects representing
            the sample. Ordered from surface to bulk. Do not provide a layer
            representing the surface, the surface is added at index 0 by the
            algorithm.
          name: Defines the name of the sample.
          implantation_model: Defines the model used for positron
            implantation.
          epithermal_correction: If True a correction for epithermal positrons
            is applied. The simple correction used here was suggested by
            [Britton]_ and is also used by VEPFIT [Veen]_.
          temperature: Defines the sample temperature in K.
          precision: Defines the decimals used for computation, i.e., error
            tolerance of the implantation profile integral and fit tolerance.
            Default is 8, resulting in a tolerance of 10^(-8).

        References:
          .. [Britton] D.T. Britton, "Epithermal effects in positron depth
                       profiling measurements", Phil. Mag. Lett., Vol. 57,
                       Number 3, pp165-169, 1988.
          .. [Veen] A. van Veen, "Analysis of positron profiling data by means
                    of 'VEPFIT'", AIP Conf. Proc., Vol. 218, pp171-198, 1991.
        """

        if isinstance(layers, Layer):
            self.layers = [layers]
        else:
            self.layers = layers
        self.name = name
        self.implantation_model = implantation_model
        self.epithermal_correction = epithermal_correction
        self.precision = precision

        self.parameters = lmfit.Parameters()
        # -2 if the fitting procedure was not yet called, else it contains the status value
        # from the fitting procedure (compare scipy's least_squares status return)
        self.fit_status = -2
        self.measurement_energies = None
        self.measurement_lineshape = None
        self.measurement_lineshape_delta = None
        #self.fit_result = None  # FitResult object from the fitting.py script
        self.annihilation_fractions = None

        if self.precision > 15:
            wrn = ('A precision of more than 15 digits is unachievable. The '
                   'precision has been set to 15.')
            print(wrn)
            self.precision = 15

        self.surface = Surface(positron_affinity=-12,
                               parameters=self.parameters,
                               precision=self.precision,
                               index=0)

        self.temperature = temperature

        # copy of self to save the initial guess
        # Here we use a shallow copy to get parameter updates. We change it to
        # a deep copy right before the fit.
        self.initial_state = copy.copy(self)

        # add fit parameters for the surface
        self.parameters.add('lineshape_0', value=np.inf, vary=True, min=1E-15)

        # for every layer: set the correct index and define the implantation
        # profile
        for i, layer in enumerate(self.layers):
            i += 1 # shift index by one to account for surface layer
            if layer.name == f'Layer {layer.index}':
                layer.name = f'Layer {i}'
            layer.index = i
            for name, p in layer.parameters.items():
                # copy all parameters from the Layer object and assign the
                # correct index to them
                self.parameters.add(f'{name.rsplit("_", maxsplit=1)[0]}_{i}',
                                    value=p.value, vary=p.vary,
                                    min=p.min, max=p.max)

            # make layer.parameters a pointer to Sample.parameters
            layer.parameters = self.parameters

            if implantation_model == 'makhov':
                layer.implantation_profile = MakhovProfile(
                                    [layer.density, *layer.makhov_parameters])
            else:
                err = ( 'Unknown model for positron implantation '
                       f'"{implantation_model}". Currently "makhov" is the'
                        'only model available.')
                raise NotImplementedError(err)

            if i == len(self.layers):
                # last layer defaults to infinite thickness
                if layer.thickness != np.inf:
                    wrn = (f'Invalid thickness ({layer.thickness}) for the '
                           f'last layer ({layer.name}). Thickness set to '
                            'inf.')
                    print(wrn)
                layer.thickness = np.inf
            else:
                if layer.thickness == np.inf:
                    # set a default starting value
                    self.parameters[f'thickness_{i}'].value = 200
                    self.parameters[f'thickness_{i}'].vary = True

        if self.epithermal_correction:
            # Due to the nature of the epithermal correction, we set the lower
            # bound for the "diffusion length" to 1E-5. This makes sure, that
            # the algorithm does not randomly deactivate the epithermal
            # correction by setting it to 1E-15 (effectively 0).
            self.parameters.add('lineshape_epithermal', value=np.inf,
                                vary=True, min=1E-15)
            self.parameters.add('diffusion_length_epithermal', value=1,
                                vary=False, min=1E-5)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.surface.temperature = value
        for layer in self.layers:
            layer.temperature = value

    def calc_implantation_profile(
        self,
        implantation_energy: float,
        num_depth: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates the combined implantation profile.

        Calculates the combined implantation profile for a given sample at a
        given energy. The maximum implantation depth is automatically
        calculated (cut-off @ 99.9%).

        Args:
          implantation_energy: The positron implantation energy in keV for
            which the implantation profile is calculated.
          num_depth: The number of depths evaluated.

        Returns:
          Two numpy arrays containing the implantation depth values and the
          implantation profile evaluated at those depths.
        """

        max_depth = np.inf
        implanted = []
        offsets = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                offset = 0
            else:
                offset = layer.implantation_profile.get_depth(
                            np.sum(implanted), implantation_energy,
                            *layer.implantation_profile.parameters)

            offsets.append(offset)

            max_layer_depth = layer.implantation_profile.get_depth(
                                  1, implantation_energy,
                                  *layer.implantation_profile.parameters)

            if max_layer_depth < layer.thickness + offset:
                # all remaining positrons are implanted into current layer
                max_depth = (np.sum([l.thickness for l in self.layers[:i]])
                             + max_layer_depth - offset)
                implanted.append(1 - np.sum(implanted))
                break
            else:
                prec_exp = 10 ** (-self.precision)
                implanted.append(integrate.quad(
                    lambda z: layer.implantation_profile(z + offset, implantation_energy),
                    0, min(max_depth, layer.thickness), epsabs=prec_exp, epsrel=prec_exp)[0])

        zlist = np.linspace(0, max_depth, num_depth)
        plist = []
        for z in zlist:
            for i in range(len(self.layers)):
                z_is_in_current_layer = (z >= np.sum([l.thickness for l in self.layers[:i]]))
                if i < len(self.layers):
                    z_is_in_current_layer &= (z < np.sum([l.thickness for l in self.layers[:i+1]]))
                
                if z_is_in_current_layer:
                    zz = z + offsets[i] - np.sum([l.thickness for l in self.layers[:i]])
                    plist.append(self.layers[i].implantation_profile(zz, implantation_energy))

        return zlist, plist

    def model_diffusion(
        self,
        implantation_energies: tuple[float, ...]
    ) -> np.ndarray:
        """Simulate positron diffusion and return the resulting depth profile.

        Simluates positron diffusion starting from the implantation profile.
        Evaluates the resulting annihilation profile based on lineshape values
        in sample.Parameters and returns a lineshape parameter for every energy
        in implantation_energies.

        Args:
          implantation_energies: Simulate positron diffusion for the
            implantation energies provided. Energies in keV.

        Returns:
          A list of lineshape parameter values resulting from positron
          annihilation after implantation and diffusion. One lineshape value
          per implantation energy specified.
        """

        # make sure that energies are floats (and not integers)
        implantation_energies = np.array(implantation_energies, dtype=float)

        # Save implantation energies for later use (mainly plotting).
        # Will be overwritten with data when calling fit().
        if self.measurement_energies is None:
            self.measurement_energies = implantation_energies

        n_energies = len(implantation_energies)
        n_layers = len(self.layers)

        implanted = np.zeros(n_energies)
        annihilated = np.zeros((n_energies, n_layers + 1))
        offsets = np.zeros((n_energies, n_layers))
        on_boundaries = np.zeros((n_energies, n_layers + 1))

        diffusion_rate = np.zeros(n_layers)
        annihilation_rate = np.zeros(n_layers)

        lineshapes = np.zeros(n_layers + 2)

        # Get surface lineshape value
        if np.isinf(self.surface.lineshape):
            err = ('Surface lineshape value is set to infinite. Please set a '
                   'sensible value.')
            raise RuntimeError(err)
        lineshapes[0] = self.surface.lineshape
        lineshapes[-1] = 0

        implanted_fractions = []
        for i, layer in enumerate(self.layers):
            # get lineshape values
            lineshapes[i + 1] = layer.lineshape

            c_left, c_right, c_annihilated, c_implanted, offset = \
                layer.solve_first_diffusion_step(implantation_energies,
                                                 implanted)

            on_boundaries[:, i] += c_left
            on_boundaries[:, i + 1] += c_right
            annihilated[:, i+1] += c_annihilated
            offsets[:, i] = offset
            implanted_fractions.append(c_implanted)
            implanted += c_implanted

            # prepare second part of diffusion process
            diffusion_rate[i], annihilation_rate[i] = \
                            layer.get_rates_for_second_part_of_diffusion()

        self.implantation_fractions = implanted_fractions

        # initialize diffusion and annihilation rate arrays
        diffusion_rate_l = np.zeros(n_layers + 1)
        diffusion_rate_r = np.zeros(n_layers + 1)
        annihilation_rate_l = np.ones(n_layers + 1)
        annihilation_rate_r = np.zeros(n_layers + 1)

        # set diffusion and annihilation rates (to non-normalized values)
        diffusion_rate_l[1:-1] = diffusion_rate[:-1]
        diffusion_rate_r[1:-1] = diffusion_rate[1:]
        annihilation_rate_l[1:-1] = annihilation_rate[:-1]
        annihilation_rate_r[1:-1] = annihilation_rate[1:]

        # normalize diffusion and annihilation rates
        sum_of_rates = (diffusion_rate_r + diffusion_rate_l + annihilation_rate_r
                        + annihilation_rate_l)
        diffusion_rate_l /= sum_of_rates
        diffusion_rate_r /= sum_of_rates
        annihilation_rate_l /= sum_of_rates
        annihilation_rate_r /= sum_of_rates

        ### Markov Chain
        number_of_transient_states = len(self.layers) + 1
        transient_states_names = ['Surface', *[f'Interface {i}' for i in range(len(self.layers))]]
        absorbing_states_names = ['Surface', *[l.name for l in self.layers]]
        # Q matrix (transitions between transient states)
        Q = (np.diag(diffusion_rate_r[:-1], 1)
             + np.diag(diffusion_rate_l[1:], -1))

        # R matrix (transitions from transient to absorbing states)
        R = (np.diag(annihilation_rate_r[:-1], 1)
             + np.diag(annihilation_rate_l))
        R[0,0] = 1  ## surface trapped positrons
        R[0,1] = 0  ## annihilate at the surface

        N = np.linalg.inv(np.identity(number_of_transient_states) - Q)

        # Calculate the Absorption Probabilities (B = NR)
        B = np.dot(N, R)

        result = np.dot(on_boundaries, B) + annihilated
        ls_model = np.dot(result, lineshapes[:-1])

        self.annihilation_fractions = result

        if self.epithermal_correction:
            # fraction of epithermal positrons, uses the offset values calculated previously
            epi_frac = np.zeros(n_energies)

            for i, e in enumerate(implantation_energies):
                for j, layer in enumerate(self.layers):
                    func = (lambda z: layer.implantation_profile(z+offsets[i, j], e)
                            * np.exp(-(z+offsets[i, j])
                            / self.parameters["diffusion_length_epithermal"].value))
                    max_depth = layer.implantation_profile.get_depth(
                                  1, e, *layer.implantation_profile.parameters)
                    epi_frac[i] += integrate.quad(func, 0,
                                            min(layer.thickness, max_depth))[0]

            # correct lineshape for epithermal positrons
            ls_model = (ls_model * (1 - epi_frac)
                        + self.parameters["lineshape_epithermal"].value
                        * epi_frac)
            self.annihilation_fractions = np.c_[epi_frac,
                                            result * (1 - epi_frac[:, None])]

        return ls_model

    def fit(
        self,
        implantation_energies: np.ndarray,
        lineshape: np.ndarray,
        lineshape_deltas: np.ndarray|None = None,
        report_to: str = "./limpid-fit-report.txt",
        max_nfev: int = 200,
        verbose: int = 0,
    ) -> lmfit.minimizer.MinimizerResult:
        """Performs a fit to the data provided.

        Hands a residual function of the model_diffusion() function and the
        lineshape data provided to a lmfit.Minimizer() instance.

        Args:
          implantation_energies: A list of positron implantation energies in
            keV, from your data.
          lineshape: A list of measured data, e.g., S or W parameter.
          lineshape_delta: A list of errorbars for the lineshape data.
          report_to: A string containing the desired output filepath. Caution:
            Existing files will be overwritten.
          max_nfev: An integer defining the maximum number of function
            evaluations.
          verbose: An integer representing the amount of information output.
            1 - display a termination report, 2 - display progress during
            iterations.

        Returns:
          A lmfit.minimizer.MinimizerResult instance.

        Raises:
          RuntimeError: Tried to fit the Sample object twice.
        """

        # Define a useful initial guess. Lineshape guesses are only used, if
        # the value is set to inf. This avoids overwriting user defined values.
        if (self.epithermal_correction
            and self.parameters['lineshape_epithermal'].value == np.inf):
            self.parameters['lineshape_epithermal'].value = lineshape[0]

        last_layer = self.layers[-1]
        if not np.isinf(last_layer.thickness):
            wrn = (f'Invalid thickness ({last_layer.thickness}) for the '
                   f'last layer ({last_layer.name}). Thickness set to '
                    'inf.')
            print(wrn)
            last_layer.thickness = np.inf

        self.initial_state = copy.deepcopy(self)

        # Save input data to the sample object for later use (i.e., plots,
        # output, etc.). If the sample object was already used to fit a 
        # dataset, raise an error to avoid problems caused by negligence.
        if self.fit_status > -2:
            err = ('Sample object was already used to fit a dataset. Please '
                   'create a new Sample object.')
            raise RuntimeError(err)
        else:
            self.measurement_energies = implantation_energies.copy()
            self.measurement_lineshape = lineshape.copy()
            if lineshape_deltas is not None:
                self.measurement_lineshape_delta = lineshape_deltas.copy()

        # TODO: Sort data by implantation energy

        index = -1
        stepsize = len(lineshape) // len(self.layers)
        for layer in self.layers[::-1]:
            if layer.lineshape == np.inf:
                layer.lineshape = lineshape[index]
            index -= stepsize

        # Surface lineshape guess for the surface. If epithermal_correction
        # is True, use a value between epithermal and first layer lineshape.
        if self.surface.lineshape == np.inf:
            if self.epithermal_correction:
                self.surface.lineshape = (self.parameters['lineshape_epithermal'].value
                                              + self.parameters['lineshape_1'].value) / 2
            else:
                self.surface.lineshape = lineshape[0]

        def residuals(parameters, energies):
            """Residual function to be minimized by the fit."""
            # update parameters used by the model_diffusion() method
            for p in parameters:
                self.parameters[p].value = parameters[p].value

            diff = (self.model_diffusion(energies) - lineshape)
            if lineshape_deltas is not None:
                diff /= lineshape_deltas
            return diff

        # Perform the actual fit
        start_time = time.time()
        mini = lmfit.Minimizer(residuals, self.parameters,
                               fcn_args=(implantation_energies,),
                               verbose=verbose)

        precision_exp = 10 ** (-self.precision)
        fit_result = mini.least_squares(max_nfev=max_nfev, jac='2-point',
                                        ftol=precision_exp, xtol=precision_exp,
                                        gtol=precision_exp)

        fit_duration = np.round(time.time() - start_time, 5)

        # update parameters object
        self.parameters = fit_result.params

        try:
            self.fit_status = fit_result.status
        except AttributeError:
            # lmfit.minimizer.MinimizerResult has the parameter 'status' only
            # defined in case of a successful fit.
            self.fit_status = -1

        if self.fit_status <= 0:
            wrn = ('Fit did not succeed.')
            print(wrn)
        else:
            msg = (f'\nFit duration: {fit_duration} s'
                    '\nDiffusion length(s):')
            for layer in self.layers:
                diffusion_length_err = fit_result.params[f'diffusion_length_{layer.index}'].stderr
                msg += (f'\n- {layer.name}: '
                        f'({np.round(layer.diffusion_length, self.precision)} '
                        f'+/- {np.round(diffusion_length_err, self.precision)}'
                         ') nm')
            if verbose >= 1:
                print(msg)
            if report_to:
                with open(report_to, 'w') as f:
                    f.write(msg)

        return fit_result
