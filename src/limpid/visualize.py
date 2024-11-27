import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from .limpid import Sample
from .utils import calc_implantation_profile


def check_sample_status(sample: Sample):
    """
    Checks if the sample contains a valid fitting result, expressed by the sample.status flag,
    which is set by the fitting procedure.
    """

    if sample.fit_status < -1:
        err = ('This Sample object contains no valid fitting result.')
        raise RuntimeError(err)
    elif sample.fit_status < 1:
        print('The fit terminated before any fitting condition was satisfied.')


def fit_result(sample: Sample):
    """
    Simple plot of the fit result.
    """
    check_sample_status(sample)

    plt.errorbar(sample.measurement_energies, sample.measurement_lineshape, sample.measurement_lineshape_delta,
                 ls='', capsize=3, label="data")
    energies = np.linspace(sample.measurement_energies[0], sample.measurement_energies[-1], 120)
    plt.plot(energies, sample.model_diffusion(energies), label="fit")
    plt.legend()
    plt.xlabel("Energy (keV)")
    plt.ylabel("Lineshape")
    plt.title(sample.name)
    plt.tight_layout()
    plt.show()

def detailed_fit_result(sample: Sample, output_dir="", profile_energies=("mid", "high"), show=True):
    """
    More detailed plots of the fit result.
    """
    check_sample_status(sample)

    assert len(profile_energies) == 2
    if profile_energies[0] == "mid":
        e_0 = sample.measurement_energies[len(sample.measurement_energies) // 2]
    elif isinstance(profile_energies[0], int) or isinstance(profile_energies[0], float):
        e_0 = profile_energies[0]
    else:
        raise TypeError(f"{profile_energies[0]} is neither float nor int.")

    if profile_energies[1] == "high":
        e_1 = sample.measurement_energies[-1]
    elif isinstance(profile_energies[1], int) or isinstance(profile_energies[0], float):
        e_1 = profile_energies[1]
    else:
        raise TypeError(f"{profile_energies[1]} is neither float nor int.")

    fig_main, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
    fig_main.suptitle(sample.name)

    energies = np.linspace(sample.measurement_energies[0], sample.measurement_energies[-1], 120)

    # fit result
    ax1.errorbar(sample.measurement_energies, sample.measurement_lineshape, sample.measurement_lineshape_delta,
                 ls='', capsize=3, label="data")
    ax1.plot(energies, sample.model_diffusion(energies), label="fit")
    ax1.set(ylabel="S parameter")
    ax1.legend()

    # residuals
    residuals = sample.model_diffusion(sample.measurement_energies) - sample.measurement_lineshape
    cumres = np.cumsum(residuals)
    res_and_cumres = np.concatenate((residuals, cumres))
    ax2.plot(sample.measurement_energies, np.zeros_like(sample.measurement_energies), color="black")
    ax2.plot(sample.measurement_energies, cumres, color='orange', linestyle='--', label='cumulative sum')
    ax2.scatter(sample.measurement_energies, residuals, label='residuals')
    ymax = 1.15 * np.max(np.abs(res_and_cumres))
    ax2.set_ylim(-ymax, ymax)
    ax2.legend()
    ax2.set(xlabel="E (keV)")

    # mid energy profile
    z_mid, p_mid = calc_implantation_profile(sample, e_0, 100)
    ax3.plot(z_mid, p_mid, label=str(round(e_0, 3)) + " keV")
    ax3.legend()
    ax3.yaxis.set_label_position("right")
    ax3.set(ylabel="Implanted fraction")
    ax3.yaxis.tick_right()

    z_high, p_high = calc_implantation_profile(sample, e_1, 100)
    ax4.plot(z_high, p_high, label=str(round(e_1, 3)) + " keV")
    ax4.legend()
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax4.set(ylabel="Implanted fraction")
    ax4.set(xlabel="Depth (nm)")

    #fig_main.tight_layout()
    savename = sample.name.split(".")[0]
    plt.savefig(output_dir + f"limpid_out_detailed_{savename}")
    if show:
        plt.show()
    plt.close()

def show_imp_ann_fracs(sample, save=True, show=False, names_layer=None, 
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

    if type(sample.markov_vector) == bool:
        err = ("Cannot create plot of implanted and annihilated fractions "
               "without modelling diffusion. Call Sample.fit() or "
               "Sample.model_diffusion() first.")
        raise TypeError(err)

    imp_energy = sample.measurement_energies
    #annihilation fractions as a function of energy (result of LIMPID fit)
    f_channel = []
    names_channel = []
    if sample.epithermal_correction:
        epi = 1
        f_epithermal = sample.markov_vector[:,0]
        f_channel.append(f_epithermal)
        f_surface = sample.markov_vector[:,1]
        names_channel.append("epithermal")
    else:
        epi = 0
        f_surface = sample.markov_vector[:,0]
    f_channel.append(f_surface)
    names_channel.append("surface")
    for i,layer in enumerate(sample.layers):
        f_channel.append(sample.markov_vector[:,epi+1+i])
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
    f_imp = sample.implantation_fractions
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
