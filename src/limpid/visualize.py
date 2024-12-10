import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy import integrate

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

def calc_implantation_profile(
    sample: Sample,
    implantation_energy: float,
    num_depth: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the combined implantation profile.

    Calculates the combined implantation profile for a given sample at a given
    energy. The maximum implantation depth is automatically calculated
    (cut-off @ 99.9%).

    Args:
      sample: The Sample object containing all the data.
      implantation_energy: The positron implantation energy for which the
        implantation profile is calculated.
      num_depth: The number of depths evaluated.

    Returns:
      Two numpy arrays containing the implantation depth values and the
      implantation profile evaluated at those depths.
    """

    layers = sample.layers
    max_depth = np.inf
    implanted = [0]
    offsets = []

    for i, layer in enumerate(layers):

        if i == 0:
            offset = 0
        else:
            offset = layer.implantation_profile.get_depth(sum(implanted), implantation_energy)

        offsets.append(offset)

        max_layer_depth = layer.implantation_profile.get_depth(0.999, implantation_energy)

        if max_layer_depth < layer.thickness + offset:
            max_depth = max_layer_depth - offset
            implanted.append(0.999 - implanted[-1])

        else:
            implanted.append(integrate.quad(lambda z: layer.implantation_profile(z + offset, implantation_energy), 0,
                                            layer.thickness, epsabs=1e-5, epsrel=1e-5)[0] - implanted[-1])

    def combined_implantation_profile(z):

        lower_bound = 0
        total_depth = 0

        for offset, layer in zip(offsets, layers):
            total_depth += layer.thickness
            if lower_bound <= z <= total_depth:
                return layer.implantation_profile(z + offset, implantation_energy)
            else:
                lower_bound = total_depth

    z = np.linspace(0, max_depth, num_depth)

    return z, [combined_implantation_profile(z_val) for z_val in z]

def fit_result(sample: Sample):
    """Shows a plot of the data and fit result.

    Args:
      sample: The Sample object containing all the data.

    Returns:
      A tuple of the matplotlib Figure and Axis objects.
    """

    check_sample_status(sample)

    fig, ax = plt.subplots()
    ax.errorbar(sample.measurement_energies, sample.measurement_lineshape,
                sample.measurement_lineshape_delta, ls='', capsize=3,
                label="data")
    energies = np.linspace(sample.measurement_energies[0],
                           sample.measurement_energies[-1], 120)
    ax.plot(energies, sample.model_diffusion(energies), label='fit')
    ax.legend()
    ax.set_xlabel('Energy / keV')
    ax.set_ylabel('Lineshape')
    fig.suptitle(sample.name)
    plt.show()

    return fig, ax

def detailed_fit_result(
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
      profile_energies: A tuple of two selected implantation energies. The
        figure will contain the corresponding implantation profiles.
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
    axs[0,0].errorbar(sample.measurement_energies, sample.measurement_lineshape, sample.measurement_lineshape_delta,
                 ls='', capsize=3, label="data")
    axs[0,0].plot(energies, sample.model_diffusion(energies), label="fit")
    axs[0,0].set(ylabel="S parameter")
    axs[0,0].legend()

    # residuals
    residuals = sample.model_diffusion(sample.measurement_energies) - sample.measurement_lineshape
    cumres = np.cumsum(residuals)
    res_and_cumres = np.concatenate((residuals, cumres))
    axs[0,1].plot(sample.measurement_energies, np.zeros_like(sample.measurement_energies), color="black")
    axs[0,1].plot(sample.measurement_energies, cumres, color='orange', linestyle='--', label='cumulative sum')
    axs[0,1].scatter(sample.measurement_energies, residuals, label='residuals')
    ymax = 1.15 * np.max(np.abs(res_and_cumres))
    axs[0,1].set_ylim(-ymax, ymax)
    axs[0,1].legend()
    axs[0,1].set(xlabel="E (keV)")

    # mid energy profile
    z_mid, p_mid = calc_implantation_profile(sample, e_0)
    axs[1,0].plot(z_mid, p_mid, label=str(round(e_0, 3)) + " keV")
    axs[1,0].legend()
    axs[1,0].yaxis.set_label_position("right")
    axs[1,0].set(ylabel="Implanted fraction")
    axs[1,0].yaxis.tick_right()

    z_high, p_high = calc_implantation_profile(sample, e_1)
    axs[1,1].plot(z_high, p_high, label=str(round(e_1, 3)) + " keV")
    axs[1,1].legend()
    axs[1,1].yaxis.set_label_position("right")
    axs[1,1].yaxis.tick_right()
    axs[1,1].set(ylabel="Implanted fraction")
    axs[1,1].set(xlabel="Depth (nm)")

    savename = sample.name.split(".")[0]
    plt.savefig(output_dir + f"limpid_out_detailed_{savename}")
    if show:
        plt.show()

    return fig, axs

def plot_fractions(
    sample: limpid.Sample,
    save: bool = True,
    show: bool = True,
    savename: str = "positron_fractions.pdf",
    fig: plt.Figure|None = None,
):
    """Displays the distribution of implanted and annihilated positrons.

    Derives the positron implantation and annihilation fractions from a
    limpid.Sample object and plots them in a cumulative style.

    Args:
      sample: The Sample object containing all the data.
      show: Show a popup window containing the plot.
      save: Save the figure to a file.
      savename : Filepath to save the figure at.
      fig: matplotlib.pyplot.Figure instance used for plotting.

    Returns:
      A tuple of the matplotlib Figure and Axes objects.
    """

    if type(sample.markov_vector) == bool:
        err = ("Cannot create plot of implanted and annihilated fractions "
               "without modelling diffusion. Call Sample.fit() or "
               "Sample.model_diffusion() first.")
        raise TypeError(err)

    implantation_energies = sample.measurement_energies
    annihilation_fractions = sample.markov_vector.transpose()
    annihilation_channels = ['surface', *[l.name for l in sample.layers]]

    if sample.epithermal_correction:
        annihilation_channels.insert(0, 'epithermal')
        skip_colors = 2
    else:
        skip_colors = 1

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    while len(colors) < len(annihilation_channels):
        colors += colors

    if not fig is None:
        axs = fig.get_axes()
    else:
        fig, axs = plt.subplots(2, 1, sharex=True)

    # implantation fractions
    cumsum_implantation = np.zeros_like(sample.implantation_fractions[0])
    for i, layer in enumerate(sample.layers):
        cumsum_implantation += sample.implantation_fractions[i]
        axs[0].plot(implantation_energies, cumsum_implantation, linestyle='-',
                    marker='', color='black')
        axs[0].fill_between(x=implantation_energies,
                    y1=cumsum_implantation-sample.implantation_fractions[i],
                    y2=cumsum_implantation, color=colors[i+skip_colors])
    axs[0].set_ylabel('Implantation\nfractions')
    axs[0].set_xlim(min(implantation_energies), max(implantation_energies))
    axs[0].set_ylim(0.0, 1.01)
    axs[0].xaxis.set_ticks_position('top')

    # annihilation fractions
    cumsum_annihilation = np.zeros_like(sample.implantation_fractions[0])
    for i, channel_name in enumerate(annihilation_channels):
        cumsum_annihilation += annihilation_fractions[i]
        axs[1].plot(implantation_energies, cumsum_annihilation, color='black')
        axs[1].fill_between(x=implantation_energies,
                y1=cumsum_annihilation-annihilation_fractions[i],
                y2=cumsum_annihilation, color=colors[i], label=channel_name)
    axs[1].set_xlabel('Positron implantation energy / keV')
    axs[1].set_ylabel('Annihilation\nfractions')
    axs[1].set_ylim(1.01, 0)
    axs[1].legend()
    fig.subplots_adjust(hspace=.0)

    if save:
        plt.savefig(f'{savename}.pdf')
    if show:
        plt.show()

    return fig, axs
