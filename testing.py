import numpy as np
import matplotlib.pyplot as plt

from src.limpid.limpid import Sample
from src.limpid.figures import fit_result, detailed_fit_result


# parameters: (rho, a, n, m)
# tungsten
w = (19.25, 4.51, 1.498, 1.621, -1)
# si
si = (2.33, 2.48, 1.792, 1.99, -6.95)

# select data file
filepath = "data/2020-12-27_W-SC_SC44_REM0.csv"

# read data from file, then load the columns into numpy arrays
dataset = np.transpose(np.genfromtxt(filepath, delimiter=","))
# create the necessary arrays for the energy, s and ds
# toss unneeded columns 2 and 4
e_data = dataset[0][dataset[0]>0]
s_data = dataset[1][-len(e_data):]
ds_data = dataset[3][-len(e_data):]

sample = Sample(layers=[w], sample_name=filepath, epithermal=True, precision=13, implantation_profile="cmakhov")

# the argument markov_chain enables/disables a different approach for diffusion model
sample.fit(s_data, ds_data, e_data, verbose=2, markov_chain=True)

detailed_fit_result(sample)

# fit_result(sample)

if sample.used_markov_to_fit:
    plt.plot(e_data, sample.markov_vector)
    plt.show()


