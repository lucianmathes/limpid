import matplotlib.pyplot as plt

from src.limpid.layer import Layer
from src.limpid.implantation import MakhovProfile
from src.limpid.limpid import Sample
import numpy as np


# parameters: (rho, a, n, m)
# tungsten
w = (19.25, 4.51, 1.498, 1.621)
# si
si = (2.33, 2.48, 1.792, 1.99)

# select data file
filepath = "data/2021-01-06_Wsinglecrystal_SC59_REM0.csv"

# read data from file, then load the columns into numpy arrays
dataset = np.transpose(np.genfromtxt(filepath, delimiter=","))
# create the necessary arrays for the energy, s and ds
# toss unneeded columns 2 and 4
e_data = dataset[0][dataset[0]>2]
s_data = dataset[1][-len(e_data):]
ds_data = dataset[3][-len(e_data):]

energy = np.linspace(e_data[0], e_data[-1], 100)

sample = Sample(layers=[w])

sample.fit(s_data, ds_data, e_data, verbose=2)

result = sample.model_diffusion(e_data)

plt.plot(e_data, s_data)
plt.plot(e_data, result)
plt.show()

