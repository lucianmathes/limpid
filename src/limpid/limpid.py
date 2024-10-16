import time
import warnings
import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from .layer import Layer
from .surface import Surface
from .implantation import MakhovProfile, CMakhovProfile
from .fitting import FitParameters, FitParameter, Fit


class Sample:

    def __init__(self, layers, implantation_profile="makhov", sample_name="", epithermal=False, precision: int = 8):
        self.name = sample_name

        """self.surfaces and self.layers: Lists containing all layers, since surfaces often need special treatments
        they are separated. The index counting their placement is chosen according the overall structure and not the
        placement in the respective list. I.e. the index for the first surface layer is zero, the indices for the
        actual layers are then 1, 2, .. and the index for the last surface (if the sample is not infinitely large) is
        then N + 1 (assuming N Layers).
        """

        self.surfaces = []
        self.layers = []
        self.precision = precision
        self.epithermal = epithermal
        self.temperature = 293

        # markov process
        self.markov_vector = False
        self.used_markov_to_fit = False

        if self.precision > 15:
            self.precision = 15
            warnings.warn("A precision of more than 15 digits is unachievable. The precision has been set to 15.")

        self.parameters = FitParameters()

        if self.epithermal:
            self.parameters.add(FitParameter(name='lineshape_epi', value=np.inf, vary=True, min=1E-15))
            # Due to the nature of the epithermal correction, the lower bound is set 1E-5. This makes sure,
            # that the algorithm does not randomly deactivate the epithermal correction by setting it to 1E-15 (
            # effectively 0).
            self.parameters.add(FitParameter(name='l_epi', value=1, vary=True, min=1E-15))


        # add first surface
        self.surfaces.append(Surface(0, self.precision, self.parameters, temp=self.temperature, pos_aff=-12))

        # add layers, index shifted by one to account for surface layer
        for idx, params in enumerate(layers):
            if implantation_profile == "makhov":
                profile = MakhovProfile(params[:-1])
            elif implantation_profile == "cmakhov":
                profile = CMakhovProfile(params[:-1])
            else:
                raise TypeError("Only 'makhov' and 'cmakhov' are currently implemented.")

            # indices shifted by one to account for the first surface interval
            if idx < len(layers) - 1:
                self.layers.append(Layer(idx + 1, self.precision, profile, self.parameters, 230, pos_aff=params[-1], temp=self.temperature))
            else:
                # last layer defaults to infinite thickness
                self.layers.append(Layer(idx + 1, self.precision, profile, self.parameters, np.inf, pos_aff=params[-1], temp=self.temperature))

        # add last surface (currently only implemented as a placeholder, since samples are assumed to be
        # infinitely large).
        # self.surfaces.append(Surface(len(self.layers) + 1, self.parameters))

        # below here are the parameters which are set after the execution
        """
        -2 if the fitting procedure was not yet called, else it contains the status value
        from the fitting procedure (compare scipy's least_squares status return)
        """
        self.fit_status = -2
        self.measurement_energies = None
        self.measurement_lineshape = None
        self.measurement_lineshape_delta = None
        self.fit_result = None  # FitResult object from the fitting.py script

    def fit(self, lineshape: np.ndarray, lineshape_deltas: np.ndarray, energies: np.ndarray, verbose=0, report_to="./limpid_out.txt", markov_chain=True, max_nfev=100):

        # save input data to the sample object for later use (i.e. plots, output, etc.)
        # if the sample object was already used to fit a dataset it will throw an error
        # this is done to reduce the potential problems caused by negligence
        if self.fit_status > -2:
            raise UserWarning("This Sample object was already used to fit a dataset. Please create a new Sample object.")
        else:
            self.measurement_energies = energies.copy()
            self.measurement_lineshape = lineshape.copy()
            self.measurement_lineshape_delta = lineshape_deltas.copy()
            # self.fit_status = 2 for testing plot, also comment everything below as well

        # the lineshape guess is only used, if the initialisation value is still set to infinite, this avoids
        # overwriting values set by the user.
        # initialize epithermal guess
        if self.epithermal and self.parameters['lineshape_epi'].value == np.inf:
            self.parameters.change_value(name='lineshape_epi', value=lineshape[0])

        # bulk lineshape guess
        number_of_layers = len(self.layers)
        number_of_energies = len(lineshape)

        # used to shift the guess for the bulk, is set to one if a second surface exists. This shifts (if shift is 1)
        # the guess from the last lineshape value to lineshape[-1-stepsize], ensuring a different guess than the
        # second surface
        shift = 0
        if len(self.surfaces) == 2:
            shift = 1

        stepsize = number_of_energies//(number_of_layers + shift)

        index = -1 - (shift * stepsize)
        for layer in self.layers[::-1]:
            if layer.get_lineshape() == np.inf:
                layer.set_lineshape(lineshape[index])
            index -= stepsize


        # initialize guess
        # Surface lineshape guess for first surface. If epithermal correction is activated,
        # it is set between the epithermal and first layer lineshape value
        if self.surfaces[0].get_lineshape() == np.inf:
            if self.epithermal:
                self.surfaces[0].set_lineshape((self.parameters['lineshape_epi'].value + self.parameters['lineshape_1'].value)/2)
            else:
                self.surfaces[0].set_lineshape(lineshape[0])

        # surface lineshape guess for last surface (if it exists)
        if len(self.surfaces) == 2 and self.surfaces[1].get_lineshape() == np.inf:
            self.surfaces[1].set_lineshape(lineshape[-1])

        # create residual function
        def residual(params):
            # params are automatically dealt with by the model_diffusion() method
            _ = params

            # returns residuals
            residuals = (self.model_diffusion(energies, markov_chain=markov_chain) - lineshape) / lineshape_deltas
            return residuals

        # fit
        start = time.time()
        fitting = Fit(residual, self.parameters)

        fitting.least_squares(method="dogbox", verbose=verbose, precision=self.precision, max_nfev=max_nfev)
        self.fit_result = fitting.result
        stop = time.time()

        self.fit_status = fitting.result.status

        if verbose >= 1:
            print(fitting.result)
            print("\nDiffusionlength(s):")
            for layer in self.layers:
                diffusionlength = self.fit_result["diffusionlength_"+layer.idx_str].value
                diffusionlength_err = self.fit_result["diffusionlength_" + layer.idx_str].stderr
                print("- Layer "+layer.idx_str+": ", str(round(diffusionlength, self.precision)) + " nm", f"(uncertainty: {round(diffusionlength_err, self.precision)} nm)")
            print(f"\nruntime: {round(stop-start, 5)} s")
        if report_to:
            with open(report_to, 'w') as f:
                f.write(str(fitting.result))
                f.write("\nDiffusionlength(s):")
                for layer in self.layers:
                    diffusionlength = self.fit_result["diffusionlength_"+layer.idx_str].value
                    diffusionlength_err = self.fit_result["diffusionlength_" + layer.idx_str].stderr
                    f.write("- Layer "+layer.idx_str+": " + str(round(diffusionlength, self.precision)) + " nm"+ f" (uncertainty: {round(diffusionlength_err, self.precision)} nm)")
                f.write(f"\nruntime: {round(stop-start, 5)} s")


    def model_diffusion(self, energies, markov_chain=False):

        implanted = np.zeros_like(energies)
        annihilated = np.zeros((len(energies), len(self.layers) + 1))
        offsets = np.zeros((len(energies), len(self.layers)))
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
        """
        if len(self.surfaces) == 2:
            lineshapes[-1] = self.surfaces[1].get_lineshape()
        else:
            lineshapes[-1] = 0

        implanted_fractions = []
        for i, layer in enumerate(self.layers):
            c_left, c_right, c_ann, c_implanted, offset = layer(implanted, energies)
            on_boundaries[:, i] += c_left
            on_boundaries[:, i + 1] += c_right
            annihilated[:, i] += c_ann
            offsets[:, i] = offset
            implanted_fractions.append(c_implanted)
            implanted += c_implanted

            # prepare second part of diffusion process
            diffusion_rate[i], annihilation_rate[i] = layer.get_rates_for_second_part_of_diffusion()

            # get lineshape values
            lineshapes[i + 1] = layer.get_lineshape()
        self.implantation_fractions = implanted_fractions
        # initialize diffusion and annihilation rate arrays
        diffusion_rate_l = np.zeros(len(self.layers) + 1)
        diffusion_rate_r = np.zeros(len(self.layers) + 1)
        annihilation_rate_l = np.ones(len(self.layers) + 1)
        annihilation_rate_r = np.zeros(len(self.layers) + 1)

        # set diffusion and annihilation rates to non-normalized values
        """
        The following is assumed in the values set below:
        1) (For now) all positrons located on the first interval boundary annihilate into
        the surface region, hence the left and right side diffusion_rate and the
        right side annihilation rates are zero. Only the left side annihilation
        rate is 1.

        2) For now it is assumed that the last interval is infinitely large,
        hence no second surface after the last layer. The left side annihilation
        is 1, with all other rates being 0. This enforces that all positrons annihilate within the
        defined layers. However, this definition has no effect, since no
        positrons will be able to reach the last boundary anyway (it is
        infinitely far away). This was done in preparation for a finite sized
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

        # markov chain approach
        if markov_chain:
            self.used_markov_to_fit = True
            # building markov process matrix M
            number_of_final_states = 1 + len(self.layers) + 1  # these are surface, layers, surface
            number_of_inter_states = len(self.layers) + 1  # these are the separating boundaries
            dimension = number_of_final_states + number_of_inter_states

            # create initial population matrix after first diffusion step
            population_matrix = np.zeros((len(energies), dimension))
            # column 0 (annihilated in first surface) is 0

            # columns for layer are set by the already annihilated positron fraction
            # matrix has one column to many for indexing reasons in the other algorithm
            population_matrix[:, 1:1+len(self.layers)] = annihilated[:, :-1]

            # column 1+len(self.layers) (annihilated in first surface) is 0

            # columns 2+len(self.layers): are set to the populations on the booundaries
            population_matrix[:, 2 + len(self.layers):] = on_boundaries

            M = np.zeros((dimension, dimension))
            M[:number_of_final_states, :number_of_final_states] = np.eye(number_of_final_states)

            for i in range(number_of_inter_states):
                row = np.zeros(dimension)
                row[i:i+2] = annihilation_rate_l[i], annihilation_rate_r[i]
                if i == 0:
                    row[number_of_final_states + 1 + i] = diffusion_rate_r[i]
                elif i == number_of_inter_states - 1:
                    row[number_of_final_states - 1 + i] = diffusion_rate_l[i]
                else:
                    row[number_of_final_states - 1 + i] = diffusion_rate_l[i]
                    row[number_of_final_states + 1 + i] = diffusion_rate_r[i]

                M[number_of_final_states + i, :] = row

            # equals M^(2^n) for the markov process, which is more than enough for n=6
            for i in range(6):
                M = M @ M

            result = population_matrix @ M

            residuals_of_markov = np.sum(result[:, number_of_final_states:], axis=1)
            if np.amax(residuals_of_markov) > 10**(-self.precision):
                raise Warning("Markov process did not converge, use the legacy method by setting 'markov_chain=False'.")

            self.markov_vector = result[:, :len(self.layers)+len(self.surfaces)].copy()

            ls_model = result[:, :number_of_final_states] @ lineshapes

        # legacy approach
        else:
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

        if self.epithermal:
            # fraction of epithermal positrons, uses the offset values calculated previously
            epi_frac = np.zeros_like(energies)

            for i, e in enumerate(energies):
                for j, layer in enumerate(self.layers):
                    epi_frac[i] += integrate.quad(lambda z: layer.implantation_profile(z+offsets[i, j], e) *
                                                            np.exp(-(z+offsets[i, j])/self.parameters["l_epi"].value),
                                                  0, layer.get_thickness())[0]

            if self.used_markov_to_fit:
                markov_vector_old = self.markov_vector.copy()
                self.markov_vector = np.zeros((len(energies), len(self.layers)+len(self.surfaces) + 1))
                self.markov_vector[:, 0] = epi_frac
                self.markov_vector[:, 1:] = markov_vector_old * np.tile((1 - epi_frac), (len(self.layers)+len(self.surfaces), 1)).T

            # correct lineshape for epithermal positrons
            ls_model = ls_model * (1 - epi_frac) + self.parameters["lineshape_epi"].value * epi_frac

        return ls_model

    def show_imp_ann_fracs(self, save=True, show=False, names_layer=None, 
                           taglines=None, figtype="pdf",
                           fontsize=11,
                           figsize=(5.8476, 2.5),
                           channel_colors=['lightsteelblue', 'darkseagreen', 'midnightblue', "sienna"],
                           savename="imp_ann_frac_plot", axis=None):
        """Display Layer distribution of implanted and annihilated positrons.

        Parameters
        ----------
        show : bool
            Show plot to user (default is False).
        save : bool
            Save figutr to file (default is True).
        names_layer : list(str)
            Labels of the Layers.
        taglines : list(list(int(), float()))
            List of vertical lines to draw. Each line is described by a list
            containing an integer desgination the layer and a float (in units 
            of keV) giving the energy at which it is to be drawn.
        fontsize : int
            (default is 11)
        figsize : tuple(float(), float())
            Figuresize (default is (5.8476, 2.5)).
        channel_colors : list(str()), optional
            (Default is ['lightsteelblue', 'darkseagreen'])
        savename : str, optional
            (default is "imp_ann_frac_plot.pdf"
        """

        if type(self.markov_vector) == bool:
            err = ("Cannot create plot of implanted and annihilated fractions "
                   "without modelling diffusion. Call Sample.fit() or "
                   "Sample.model_diffusion() first.")
            raise TypeError(err)

        imp_energy = self.measurement_energies
        #annihilation fractions as a function of energy (result of LIMPID fit)
        f_channel = []
        names_channel = []
        if self.epithermal:
            epi = 1
            f_epithermal = self.markov_vector[:,0]
            f_channel.append(f_epithermal)
            f_surface = self.markov_vector[:,1]
            names_channel.append("epithermal")
        else:
            epi = 0
            f_surface = self.markov_vector[:,0]
        f_channel.append(f_surface)
        names_channel.append("surface")
        for i,layer in enumerate(self.layers):
            f_channel.append(self.markov_vector[:,epi+1+i])
            makhov_params = layer.implantation_profile.params
            if names_layer:
                names_channel.append(names_layer[i])
            else:
                names_channel.append(f"layer{i}") 
        
        # configure matplotlib
        ml = MultipleLocator(0.1)
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['figure.figsize'] = figsize
        if not show:
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })
        
        # make plots
        if not axis is None:
            axs = axis
        else:
            fig, axs = plt.subplots(2, 1, sharex=True)
        # implantation fractions
        f_imp = self.implantation_fractions
        f = np.zeros(np.size(f_imp[0]))
        for f_lay,tag,col in zip(f_imp, names_channel, channel_colors):
            f += f_lay 
            axs[0].plot(imp_energy, f, linestyle='-', marker='', color='black')
            axs[0].fill_between(x=imp_energy, y1 =f-f_lay, y2=f, color=col, label=tag)
        axs[0].set_ylabel('Implantation \n fractions', fontsize=9)
        axs[0].set_xlim(min(imp_energy), max(imp_energy))
        axs[0].set_ylim(0.0, 1.01)
        axs[0].xaxis.set_ticks_position('top')
        axs[0].yaxis.set_minor_locator(ml)
        axs[0].grid(which='both', linestyle='--')
        # annihilation fractions
        f = np.zeros(np.size(f_channel[0]))
        channel_colors.insert(0, "grey")
        n_l = 0
        for f_ch, tag, col in zip(f_channel, names_channel, channel_colors):
            f += f_ch
            axs[1].plot(imp_energy, f, color='black')
            axs[1].fill_between(x=imp_energy, y1=f-f_ch, y2=f, color=col, label=tag)
            if taglines:
                for l in taglines:
                    if n_l == l[0]:
                        e = l[1]
                        lymin = np.interp(e, imp_energy, f-f_ch)
                        lymax = np.interp(e, imp_energy, f)
                        axs[1].vlines(e, ymin=lymin, ymax=lymax, color='steelblue')
                        axs[1].text(e, 0.62, f'{e:.1f} keV', horizontalalignment='center', fontsize = 8) #, backgroundcolor='lightsteelblue')
            n_l += 1
        axs[1].set_xlabel('Implantation energy / keV')
        axs[1].set_ylabel('Annihilation\nfractions', fontsize=9)
        axs[1].set_ylim(1.01, 0)
        axs[1].yaxis.set_minor_locator(ml)
        axs[1].grid(which='both', linestyle='--')
        if axis is None:
            plt.tight_layout()
            plt.legend()
            fig.subplots_adjust(hspace=.0)
        if save and (axis is None):
            if figtype == "pgf":
                matplotlib.use("pgf")
            savename += "." + figtype
            print(savename)
            plt.savefig(savename)
        if show and not (figtype == "pgf"):
            plt.show()
        return axs
