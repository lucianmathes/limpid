import limpid

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.792, 1.99))
cu = limpid.Layer(density=8.96, makhov_parameters=(2.84, 1.67, 1.73))
sample = limpid.Sample([cu, si])

e, s, ds = limpid.load_example_data('cu-si')

sample.parameters["lineshape_0"].value = 0.63
sample.parameters["lineshape_1"].value = 0.58
sample.parameters["lineshape_2"].value = 0.64
sample.parameters["diffusion_length_1"].value = 50
sample.parameters["diffusion_length_2"].value = 500

limpid.initial_guess(sample, e, s, ds)

out = sample.fit(e, s, ds)
limpid.fit_result(sample)

limpid.plot_fractions(sample)
