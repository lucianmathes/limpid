import src.limpid.layer
from src.limpid.layer import Layer
from src.limpid.surface import Surface
from src.limpid.implantation import MakhovProfile
import numpy as np
import time
from src.limpid.fitting import FitParameters, Fit


class Sample:

    def __init__(self, layers, implantation_profile="makhov", sample_name="", mode="fit"):
        self.name = sample_name

        """self.surfaces and self.layers: Lists containing all layers, since surfaces often need special treatments 
        they are separated. The index counting their placement is chosen according the overall structure and not the 
        placement in the respective list. I.e. the index for the first surface layer is zero, the indices for the 
        actual layers are then 1, 2, .. and the index for the last surface (if the sample is not infinitely large) is 
        then N + 1 (assuming N Layers). 
        """

        self.surfaces = []
        self.layers = []

        self.parameters = FitParameters()

        # add first surface
        self.surfaces.append(Surface(0, self.parameters))

        # add layers, index shifted by one to account for surface layer
        for idx, params in enumerate(layers):
            if implantation_profile == "makhov":
                profile = MakhovProfile(params)
            else:
                raise TypeError("Only 'makhov' is supported.")

            # indices shifted by one to account for the first surface interval
            self.layers.append(Layer(idx + 1, profile, self.parameters))

        # add last surface (currently only implemented as a placeholder, since samples are assumed to be
        # infinitely large).
        # self.surfaces.append(Surface(len(self.layers) + 1, self.parameters))

    def fit(self, lineshape: np.ndarray, lineshape_deltas: np.ndarray, energies: np.ndarray, verbose=0):

        # initialize guess
        # surface lineshape guess
        surface = self.surfaces[0]
        surface.set_lineshape(lineshape[0])

        # bulk lineshape guess
        bulk_lineshape_guess = lineshape[-1]
        for layer in self.layers:
            layer.set_lineshape(bulk_lineshape_guess)

        # create residual function
        def residual(params):
            # params are automatically dealt with by the model_diffusion() method
            _ = params

            # returns residuals
            residuals = (self.model_diffusion(energies) - lineshape) / lineshape_deltas
            return residuals

        # fit
        start = time.time()
        fitting = Fit(residual, self.parameters)

        fitting.least_squares(method="dogbox", verbose=verbose)
        stop = time.time()

        if verbose >= 1:
            print(fitting.result)
            D = self.parameters["diffusioncoeff_1"].value
            u = self.parameters["u_1"].value
            print("\nDiffusionlength: ", np.sqrt(D * 1/(u**2 * D)))

        # caching
        if verbose >= 2:
            print("\nTime: ", stop-start)
            for layer in self.layers:
                print("conc_left for layer ", layer.idx, ": ", layer.concentration_left.cache_info())
                print("conc_right for layer ", layer.idx, ": ", layer.concentration_right.cache_info())
            print("makhov profile: ", src.limpid.implantation.makhov_profile.cache_info())
            print("makhov profile offset: ", src.limpid.implantation.makhov_offset.cache_info())

    def model_diffusion(self, energies):

        implanted = np.zeros_like(energies)
        annihilated = np.zeros((len(energies), len(self.layers) + 1))
        on_boundaries = np.zeros((len(energies), len(self.layers) + 1))

        diffusion_rate = np.zeros(len(self.layers))
        annihilation_rate = np.zeros(len(self.layers))

        lineshapes = np.zeros(len(self.layers) + 2)

        """
        Currently only the lineshapes are fitted for the surface layer, all other parameters are assumed to have certain
        properties, which is further elaborated down below.
        """

        # Set surface lineshape
        lineshapes[0] = self.surfaces[0].get_lineshape()

        # Set second surface lineshape
        """
        The second surface would be necessary for a finite sized last layer. 
        This is for now not enabled.
        """
        lineshapes[-1] = 0

        for i, layer in enumerate(self.layers):
            c_left, c_right, c_ann, c_implanted = layer(implanted, energies)
            on_boundaries[:, i] += c_left
            on_boundaries[:, i + 1] += c_right
            annihilated[:, i] += c_ann
            implanted += c_implanted

            # prepare second part of diffusion process
            diffusion_rate[i], annihilation_rate[i] = layer.get_rates_for_second_part_of_diffusion()

            # get lineshape values
            lineshapes[i + 1] = layer.get_lineshape()

        # initialize diffusion and annihilation rate arrays
        diffusion_rate_l = np.zeros(len(self.layers) + 1)
        diffusion_rate_r = np.zeros(len(self.layers) + 1)
        annihilation_rate_l = np.ones(len(self.layers) + 1)
        annihilation_rate_r = np.zeros(len(self.layers) + 1)

        # set diffusion and annihilation rates to non-normalized values
        """
        The following is assumed in the values set below:
        1) all positrons located on the first interval boundary annihilate into 
        the surface region, hence the left and right side diffusion_rate and the 
        right side annihilation rates are zero. Only the left side annihilation 
        rate is 1.
        
        2) For now it is assumed that the last interval is infinitely large, 
        hence no second surface after the last layer. The left side annihilation 
        is 1, with all other rates being 0. From the point of view of the 
        simulation this enforces that all positrons annihilate within the 
        defined layers. However, this definition has no effect, since no 
        positrons will be able to reach the last boundary anyway (it is 
        infinitely large). This was done in preparation for a finite sized 
        layer, which would introduce a second surface at the end. To account for 
        this consideration, only the left and right side annihilation rates have 
        to be swapped: i.e. the right side annihilation rate is 1, with all 
        other rates being 0.
        """
        diffusion_rate_l[1:-1] = diffusion_rate[:-1]
        diffusion_rate_r[1:-1] = diffusion_rate[1:]
        annihilation_rate_l[1:-1] = annihilation_rate[:-1]
        annihilation_rate_r[1:-1] = annihilation_rate[1:]

        # normalize diffusion and annihilation rates
        norm = diffusion_rate_r + diffusion_rate_l + annihilation_rate_r + annihilation_rate_l
        diffusion_rate_l = diffusion_rate_l / norm
        diffusion_rate_r = diffusion_rate_r / norm
        annihilation_rate_l = annihilation_rate_l / norm
        annihilation_rate_r = annihilation_rate_r / norm

        # iterate over boundaries
        ls_model = np.zeros(len(energies))
        ls_eff = np.zeros(len(energies))
        r = 0

        for i in range(len(self.layers) + 1):
            """
            Iterating through layers. The lineshape index is shifted by one, 
            since the first value is used for the surface.
            """
            c = on_boundaries[:, i] + on_boundaries[:, i - 1] * r
            j_left_eff = diffusion_rate_l[i] * (1 - r)

            ls_model += lineshapes[i + 1] * annihilated[:, i] + (lineshapes[i] * annihilation_rate_l[i] +
                                                                 lineshapes[i + 1] * annihilation_rate_r[i] +
                                                                 ls_eff * j_left_eff) * c / (1 - j_left_eff * r)

            # prep for next step
            ls_eff = (lineshapes[i] * annihilation_rate_l[i] + lineshapes[i + 1] * annihilation_rate_r[i] +
                      ls_eff * j_left_eff) / (annihilation_rate_l[i] + annihilation_rate_r[i] + j_left_eff)

            r = diffusion_rate_r[i] / (1 - diffusion_rate_l[i] * r)

        return ls_model

    def plot(self):
        # TODO

        # copy ?
        pass

    def save(self):
        # TODO

        # copy?
        pass
