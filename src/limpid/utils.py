import numpy as np
from scipy import integrate

from .limpid import Sample


def calc_implantation_profile(sample: Sample, energy: float, N: int):
    """
    Evaluates the combined implantation profile at N points for a given sample at a given energy. The maximum
    implantation depth is automatically calculated (cut-off @ 99.9%).
    """
    layers = sample.layers
    max_depth = np.inf
    implanted = [0]
    offsets = []

    for i, layer in enumerate(layers):

        if i == 0:
            offset = 0
        else:
            offset = layer.implantation_profile.get_depth(sum(implanted), energy)

        offsets.append(offset)

        max_layer_depth = layer.implantation_profile.get_depth(0.999, energy)

        if max_layer_depth < layer.thickness + offset:
            max_depth = max_layer_depth - offset
            implanted.append(0.999 - implanted[-1])

        else:
            implanted.append(integrate.quad(lambda z: layer.implantation_profile(z + offset, energy), 0,
                                            layer.thickness, epsabs=1e-5, epsrel=1e-5)[0] - implanted[-1])

    def combined_implantation_profile(z):

        lower_bound = 0
        total_depth = 0

        for offset, layer in zip(offsets, layers):
            total_depth += layer.thickness
            if lower_bound <= z <= total_depth:
                return layer.implantation_profile(z + offset, energy)
            else:
                lower_bound = total_depth

    z = np.linspace(0, max_depth, N)

    return z, [combined_implantation_profile(z_val) for z_val in z]
