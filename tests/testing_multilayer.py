import numpy as np
import matplotlib.pyplot as plt

from src.limpid.limpid import Sample
from src.limpid.figures import fit_result, detailed_fit_result


# parameters: (rho, a, n, m)
# tungsten
w = (19.25, 4.51, 1.498, 1.621, 0)
# si
si = (2.33, 2.48, 1.792, 1.99, -6.95)
# cu
cu = (8.96, 2.84, 1.67, 1.73, -5.81)


# select data file
filepath = "data/CuSi688.csv"

# read data from file, then load the columns into numpy arrays
dataset = np.transpose(np.genfromtxt(filepath, delimiter=","))
# create the necessary arrays for the energy, s and ds
e_data = dataset[0]/1000  # dataset is in eV, but limpid assumes keV !!!!
s_data = dataset[1]
ds_data = dataset[2]

# epithermal is set to false, fit converges nicely without
# precision of six digits is sufficient
sample = Sample(layers=[cu, si], sample_name=filepath, epithermal=False, precision=13, implantation_profile="cmakhov")

# guess for layer thickness based on sample preparation, allow fitting of parameter
sample.parameters["thickness_1"].value = 400
sample.parameters["thickness_1"].vary = True

#sample.parameters["lineshape_1"].value = 0.58
#sample.parameters["lineshape_2"].value = 0.64

# set the parameters for the diffusion process to fixed values, since the diffusionlength can't be successfully fitted,
# if the relevant implantation energies are too high
sample.parameters["diffusionlength_2"].vary = False  # the fitting parameter 'u' is sqrt(mu/D)
sample.parameters["diffusionlength_2"].value = 100  # set to some arbitrary  value (trial and error), will be investigated and fixed (?)
sample.parameters["diffusioncoeff_2"].vary = False

# the argument markov_chain enables/disables a different approach for diffusion
# max_nfev limits the fitting iterations.
sample.fit(s_data, ds_data, e_data, verbose=2, markov_chain=True, max_nfev=100)

detailed_fit_result(sample)

# fit_result(sample)
if sample.used_markov_to_fit:
    plt.plot(e_data, sample.markov_vector)
    plt.legend(["surface", "cu", "si"])
    plt.show()


# general information about sample parameters:
"""
In limpid all sample parameters are collected in the parameters object. The individual parameters can be addressed by 
the same way one would use a dictionary, i.e. sample.parameters["thickness_1"] returns the thickness parameter of the 
first layer. Each parameter has multiple variables: 

- value: the value, can be set to some value as the starting guess for the fit
- vary: boolean flag, which can be used to fix a parameter (i.e. to NOT fit it) with the flag set to 'False'
- min: lower fitting bound
- max: upper fitting bound

The line 'sample.parameters["thickness_1"].value = 400' will hence set the starting guess to 400 nm, and similarly
the line 'sample.parameters["thickness_1"].vary = True' will make sure that the parameter is fitted (layer thickness 
will no be fitted by default!).

Typical parameters within the parameter objects are:
- 'lineshape_0': surface S parameter
- 'lineshape_1', 'lineshape_2', etc.: S parameters for first, second layer and so on
- 'thickness_1, 'thickness_2', etc.: thickness of first, second layer and so on
- 'diffusionlength_1', 'diffusionlength_2', etc.: diffusionlength for first, second layer and so on.
- 'diffusioncoeff_1', 'diffusioncoeff_2', etc.: the diffusioncoeff for first, second layer and so on. This parameter is
    fitted on its own. Typically this parameter can be fixed, since it does not seem to have a big influence on the fit.
    This however still needs further investigation, especially for multilayer systems.
    
And if enabled:
- 'lineshape_epi': epithermal S parameter
- 'l_epi': epithermal scattering length

Note: For systems with a large first layer (>300 nm) it can be helpful to fix the diffusion parameter and the 
u parameter for the following layer(s ? only tested for two layer systems), since the model will struggle to converge 
on a value. Example:

sample.parameters["diffusionlength_2"].vary = False
sample.parameters["diffusionlength_2"].value = 100
sample.parameters["diffusioncoeff_2"].vary = False
"""
