import numpy as np
import matplotlib.pyplot as plt

import limpid
from .limpid import Sample


def check_sample_status(sample: limpid.Sample):
    """Checks if the sample contains a valid fitting result.

    Args:
      sample: The Sample object containing all the data.

    Raises:
      RuntimeError: The Sample object has not been fit (successfully).
    """

    if sample.fit_status < -1:
        err = ('This Sample object contains no valid fitting result.')
        raise RuntimeError(err)
    elif sample.fit_status < 1:
        print('The fit terminated before any fitting condition was satisfied.')

def plot_initial_guess(
    sample: Sample,
    energies: np.ndarray|None = None,
    lineshape: np.ndarray|None = None,
    lineshape_delta: np.ndarray|None = None,
    n_energies: int = 150,
    e_min: float = 0.01,
    e_max: float = 50,
):
    """Shows a plot of the data and initial guess of the fit.

    Args:
      sample: The Sample object containing all the data.
      energies: Positron implantation energies of the measurement data in keV.
      lineshape: Measured lineshape parameters.
      lineshape_delta: Errors of the measured lineshape parameters.
      n_energies: Number of samples between min and max energy, if no data
        provided.
      e_min: Min energy in keV, if no data provided.
      e_max: Max energy in keV, if no data provided.

    Returns:
      A tuple of the matplotlib Figure and Axis objects.
    """

    # Set the epithermal initial guess otherwise done in Sample.fit().
    if (sample.epithermal_correction
        and sample.parameters['lineshape_epithermal'].value == np.inf):
        sample.parameters['lineshape_epithermal'].value = lineshape[0]

    fig, ax = plt.subplots()

    if energies is not None:
        e_min = energies[0]
        e_max = energies[-1]
    if lineshape_delta is not None:
        ax.errorbar(energies, lineshape, lineshape_delta, ls='', capsize=3,
                    label="data")
    elif lineshape_delta is None:
        ax.scatter(energies, lineshape, label="data")
        # manually step forward in the color cycle, because scatter doesn't
        ax.plot([])
    elif sample.measurement_energies is not None:
        e_min = sample.measurement_energies[0]
        e_max = sample.measurement_energies[-1]
        ax.errorbar(sample.measurement_energies, sample.measurement_lineshape,
                    sample.measurement_lineshape_delta, ls='', capsize=3,
                    label="data")

    energies = np.linspace(np.power(e_min, 1/1.6),
                           np.power(e_max, 1/1.6), n_energies) ** 1.6

    ax.plot(energies, sample.initial_state.model_diffusion(energies),
            label='initial guess')
    ax.legend()
    ax.set_xlabel('Positron implantation energy / keV')
    ax.set_ylabel('Lineshape parameter')
    fig.suptitle(sample.name)
    plt.show()

    return fig, ax

def plot_result(sample: Sample, show_init: bool = True, show: bool = True):
    """Shows a plot of the data and fit result.

    Args:
      sample: The Sample object containing all the data.
      show_init: Show the initial guess.
      show: Show the matplotlib pop-up window.

    Returns:
      A tuple of the matplotlib Figure and Axis objects.
    """

    check_sample_status(sample)

    fig, ax = plt.subplots()
    if sample.measurement_lineshape_delta is None:
        ax.scatter(sample.measurement_energies, sample.measurement_lineshape,
                   label="data")
        # manually step forward in the color cycle, because scatter doesn't
        ax.plot([])
    else:
        ax.errorbar(sample.measurement_energies, sample.measurement_lineshape,
                    sample.measurement_lineshape_delta, ls='', capsize=3,
                    label="data")
    energies = np.linspace(sample.measurement_energies[0],
                           sample.measurement_energies[-1], 120)
    if show_init:
        ax.plot(energies, sample.initial_state.model_diffusion(energies),
                label='initial guess')
    ax.plot(energies, sample.model_diffusion(energies), label='fit')
    ax.legend()
    ax.set_xlabel('Positron implantation energy / keV')
    ax.set_ylabel('Lineshape parameter')
    fig.suptitle(sample.name)
    if show:
        plt.show()

    return fig, ax

def plot_detailed_result(
    sample: limpid.Sample,
    output_dir: str = '',
    profile_energies: tuple = ('mid', 'high'),
    show: bool = True,
):
    """Shows detailed plots of the fit result.

    Creates a matplotlib figure containing four plots: The input data with the
    best fit obtained, the fit residuals, and the implantation profiles of two
    selected energies.

    Args:
      sample: The Sample object containing all the data.
      output_dir: The directory to save the figure in.
      profile_energies: A tuple of two selected implantation energies in keV.
        The figure will contain the corresponding implantation profiles. Other
        possible values (for auto-selection): 'mid', 'high'.
      show: Show a popup window containing the plot.

    Returns:
      A tuple of the matplotlib Figure and Axes objects.
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

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(sample.name)

    energies = np.linspace(sample.measurement_energies[0], sample.measurement_energies[-1], 120)

    # fit result
    if sample.measurement_lineshape_delta is None:
        axs[0,0].scatter(sample.measurement_energies,
                         sample.measurement_lineshape, label='data')
        # manually step forward in the color cycle, because scatter doesn't
        axs.plot([])
    else:
        axs[0,0].errorbar(sample.measurement_energies,
                          sample.measurement_lineshape,
                          sample.measurement_lineshape_delta,
                          ls='', capsize=3, label='data')
    axs[0,0].plot(energies, sample.model_diffusion(energies), label='fit')
    axs[0,0].set_ylabel('S parameter')
    axs[0,0].set_xlabel('Implantation energy / keV')
    axs[0,0].legend()

    # residuals
    residuals = sample.model_diffusion(sample.measurement_energies) - sample.measurement_lineshape
    cumres = np.cumsum(residuals)
    res_and_cumres = np.concatenate((residuals, cumres))
    axs[0,1].plot(sample.measurement_energies, np.zeros_like(sample.measurement_energies), color='black')
    axs[0,1].plot(sample.measurement_energies, cumres, color='orange', linestyle='--', label='cumulative sum')
    axs[0,1].scatter(sample.measurement_energies, residuals, label='residuals')
    ymax = 1.15 * np.max(np.abs(res_and_cumres))
    axs[0,1].set_ylim(-ymax, ymax)
    axs[0,1].legend()
    axs[0,1].set_xlabel('Implantation energy / keV')

    # mid energy profile
    z_mid, p_mid = sample.calc_implantation_profile(e_0)
    axs[1,0].plot(z_mid, p_mid, label=str(round(e_0, 3)) + ' keV')
    axs[1,0].legend()
    axs[1,0].set_ylabel('Implantation profile')
    axs[1,0].set_xlabel('Depth / nm')

    z_high, p_high = sample.calc_implantation_profile(e_1)
    axs[1,1].plot(z_high, p_high, label=str(round(e_1, 3)) + ' keV')
    axs[1,1].legend()
    axs[1,1].set_xlabel('Depth / nm')

    savename = sample.name.split(".")[0]
    plt.savefig(output_dir + f'limpid_out_detailed_{savename}')
    if show:
        plt.show()

    return fig, axs

def plot_fractions(
    sample: limpid.Sample,
    cumulative: bool = True,
    colors: list[str]|None = None,
    save: bool = True,
    show: bool = True,
    savename: str = 'positron_fractions.pdf',
    fig: plt.Figure|None = None,
):
    """Displays the distribution of implanted and annihilated positrons.

    Derives the positron implantation and annihilation fractions from a
    limpid.Sample object and plots them in a cumulative style.

    Args:
      sample: The Sample object containing all the data.
      cumulative: Choose between cumulative and normal plotting.
      colors: List of colors recognized by matplotlib.
      save: Save the figure to a file.
      show: Show a popup window containing the plot.
      savename : Filepath to save the figure at.
      fig: matplotlib.pyplot.Figure instance used for plotting.

    Returns:
      A tuple of the matplotlib Figure and Axes objects.
    """

    if sample.annihilation_fractions is None:
        err = ('Cannot create plot of implanted and annihilated fractions '
               'without modelling diffusion. Call Sample.fit() or '
               'Sample.model_diffusion() first.')
        raise TypeError(err)

    energies = np.linspace(sample.measurement_energies[0],
                           sample.measurement_energies[-1], 120)
    sample.model_diffusion(energies)
    annihilation_fractions = sample.annihilation_fractions
    annihilation_channels = ['surface', *[l.name for l in sample.layers]]

    if sample.epithermal_correction:
        annihilation_channels.insert(0, 'epithermal')
        skip_colors = 2
    else:
        skip_colors = 1

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    while len(colors) < len(annihilation_channels):
        colors += colors

    if fig is not None:
        axs = fig.get_axes()
    else:
        fig, axs = plt.subplots(2, 1, sharex=True)

    # implantation fractions
    if cumulative:
        cumsum_implantation = np.zeros_like(sample.implantation_fractions[0])
        for i, layer in enumerate(sample.layers):
            cumsum_implantation += sample.implantation_fractions[i]
            if i < len(annihilation_channels) - 1:
                axs[0].plot(energies, cumsum_implantation, color='black',
                            linewidth=1)
            axs[0].fill_between(x=energies,
                        y1=cumsum_implantation-sample.implantation_fractions[i],
                        y2=cumsum_implantation, color=colors[i+skip_colors])

        # annihilation fractions
        cumsum_annihilation = np.zeros_like(sample.implantation_fractions[0])
        for i, channel_name in enumerate(annihilation_channels):
            cumsum_annihilation += annihilation_fractions.T[i]
            if i < len(annihilation_channels) - 1:
                axs[1].plot(energies, cumsum_annihilation, color='black',
                            linewidth=1)
            axs[1].fill_between(x=energies,
                    y1=cumsum_annihilation-annihilation_fractions.T[i],
                    y2=cumsum_annihilation, color=colors[i], label=channel_name)
        axs[0].xaxis.set_ticks_position('top')
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(1, 0)
        fig.subplots_adjust(hspace=.0)
    else:
        for i, layer in enumerate(sample.layers):
            axs[0].plot(energies, sample.implantation_fractions[i], linestyle='-',
                        marker='', color=colors[i+skip_colors])
        for i, channel_name in enumerate(annihilation_channels):
            axs[1].plot(energies, annihilation_fractions.T[i], color=colors[i], label=channel_name)
        axs[0].set_ylim(-0.01, 1.01)
        axs[1].set_ylim(-0.01, 1.01)

    axs[0].set_ylabel('Implantation\nfractions')
    axs[0].set_xlim(min(energies), max(energies))
    axs[1].set_xlabel('Positron implantation energy / keV')
    axs[1].set_ylabel('Annihilation\nfractions')
    axs[1].legend()

    if save:
        plt.savefig(savename)
    if show:
        plt.show()

    return fig, axs
