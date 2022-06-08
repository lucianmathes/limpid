import numpy as np
from scipy.special import gamma
from functools import lru_cache


@lru_cache(maxsize=None)
def makhov_profile(z, e, rho, a, n, m):
    z_avg = a / rho * e ** n * 10
    z_0 = z_avg / gamma(1 / m + 1)

    return (m * z ** (m - 1)) / (z_0 ** m) * np.exp(-(z / z_0) ** m)

@lru_cache(maxsize=None)
def makhov_offset(rho, a, n, m, implanted, energy):
    z_avg = a / rho * energy ** n * 10
    z_0 = z_avg / gamma(1 / m + 1)

    # deals with infinite offsets, the error made by this implementation is negligible
    if implanted > 1 - 1E-15:
        implanted = 1 - 1E-15

    return z_0 * np.power(-np.log(1 - implanted), 1 / m)