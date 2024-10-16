import numpy as np
from limpid.limpid import Sample
from limpid.figures import fit_result, detailed_fit_result


# parameters: (rho, a, n, m, affinity)
# si
si = (2.33, 2.48, 1.792, 1.99, -6.95)
# cu
cu = (8.96, 2.84, 1.67, 1.73, -4.81)
# chromium
cr = (7.15, 2.74, 1.6735, 1.7595, -2.62)

# select data file
filepath = "./CuSi_data.csv"

# read data from file, then load the columns into numpy arrays
dataset = np.transpose(np.genfromtxt(filepath, delimiter=","))
# create the necessary arrays for the energy, s and ds
e_data = dataset[0]/1000  # dataset is in eV, but limpid assumes keV !!!!
s_data = dataset[1]
ds_data = dataset[2]

# epithermal is set to false, fit converges nicely without

sample = Sample(layers=[cu, si], sample_name=filepath, epithermal=False, precision=15)

sample.parameters["lineshape_0"].value = 0.65
sample.parameters["lineshape_0"].vary = False

sample.parameters["thickness_1"].value = 450
sample.parameters["thickness_1"].vary = True
sample.parameters["lineshape_1"].value = 0.58
#sample.parameters["lineshape_1"].vary = True
sample.parameters["diffusioncoeff_1"].vary = False

sample.parameters["lineshape_2"].value = 0.666345280728452
sample.parameters["lineshape_2"].vary = False
sample.parameters["diffusionlength_2"].value = 456.58819585204645
sample.parameters["diffusionlength_2"].vary = False
sample.parameters["diffusioncoeff_2"].vary = False


print(s_data[0])

print(sample.parameters["lineshape_0"])
sample.fit(s_data, ds_data, e_data, verbose=2, markov_chain=False, max_nfev=100)



detailed_fit_result(sample)


#array = np.array([e_data, sample.model_diffusion(e_data, markov_chain=False)]).T
#np.savetxt("CuSi688fit_lineshape0fixed.csv", array, delimiter=",")

#if sample.used_markov_to_fit:
#    array2 = np.array([e_data, sample.markov_vector.T[0], sample.markov_vector.T[1], sample.markov_vector.T[2]]).T
#    np.savetxt("CuSi688frac_lineshape0fixed.csv", array2, delimiter=",")
