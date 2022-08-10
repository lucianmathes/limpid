import matplotlib.pyplot as plt
import warnings
import numpy as np

from .limpid import Sample
from .utils import calc_implantation_profile


def check_sample_status(sample: Sample):
    """
    Checks if the sample contains a valid fitting result, expressed by the sample.status flag,
    which is set by the fitting procedure.
    """

    if sample.fit_status < -1:
        raise UserWarning("This Sample object contains no valid fitting result.")
    elif sample.fit_status < 1:
        warnings.warn("The fit terminated before any fitting condition was satisfied.")


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


def detailed_fit_result(sample: Sample, profile_energies=("mid", "high")):
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
    ax1.plot(energies, sample.model_diffusion(energies, markov_chain=sample.used_markov_to_fit), label="fit")
    ax1.legend()

    # residuals
    residuals = sample.model_diffusion(sample.measurement_energies, markov_chain=sample.used_markov_to_fit) - sample.measurement_lineshape
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
    ax3.yaxis.tick_right()

    z_high, p_high = calc_implantation_profile(sample, e_1, 100)
    ax4.plot(z_high, p_high, label=str(round(e_1, 3)) + " keV")
    ax4.legend()
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax4.set(xlabel="Depth (nm)")

    fig_main.tight_layout()
    plt.show()
