import numpy as np
import matplotlib.pyplot as plt
from limpid import Sample, Layer
from limpid.visualize import detailed_fit_result, plot_fractions


si = Layer(density=2.33,
           makhov_parameters=(2.48, 1.792, 1.99))
cu = Layer(8.96, (2.84, 1.67, 1.73))
cr = Layer(7.15, (2.74, 1.67, 1.76))
s = Sample([cu, si], epithermal_correction=False)

filepath = "./data/CuSi_data.csv"
dataset = np.transpose(np.genfromtxt(filepath, delimiter=","))
e_data = dataset[0]/1000  # dataset is in eV, but limpid assumes keV !!!!
s_data = dataset[1]
ds_data = dataset[2]

#plt.plot(e_data, s_data)
#plt.show()

s.parameters["lineshape_0"].value = 0.65
s.parameters["lineshape_0"].vary = True

#s.parameters["thickness_1"].value = 450
s.parameters["thickness_1"].vary = True
s.parameters["lineshape_1"].value = 0.58
#s.parameters["lineshape_1"].vary = True
s.parameters["diffusion_coefficient_1"].vary = False

s.parameters["lineshape_2"].value = 0.666345280728452
s.parameters["lineshape_2"].vary = False
s.parameters["diffusion_length_2"].value = 456.58819585204645
s.parameters["diffusion_length_2"].vary = False
s.parameters["diffusion_coefficient_2"].vary = False


out = s.fit(e_data, s_data, ds_data, verbose=1, markov_chain=True, max_nfev=100)
print(out.init_values)
print(out.params)

#res = s.model_diffusion([1, 2, 3, 4, 5], markov_chain=True)
#print(res)
plot_fractions(s)
#detailed_fit_result(s)
