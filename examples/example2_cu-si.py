import limpid

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.792, 1.99), name='Si')
cu = limpid.Layer(density=8.96, makhov_parameters=(2.84, 1.67, 1.73), name='Cu')
sample = limpid.Sample([cu, si], epithermal_correction=True)

e, s, ds = limpid.load_example_data('cu-si')

sample.parameters["lineshape_0"].value = 0.62
sample.parameters["lineshape_1"].value = 0.57
sample.parameters["lineshape_2"].value = 0.65
sample.parameters["diffusion_length_1"].value = 30
sample.parameters["diffusion_length_2"].value = 165
sample.parameters["diffusion_length_2"].vary = False
sample.parameters["thickness_1"].value = 350
sample.parameters["thickness_1"].vary = True

sample.fit(e, s, ds, verbose=True)
limpid.fit_result(sample, show_init=True)

sample.parameters.pretty_print()
