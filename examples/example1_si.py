import limpid


si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.729, 1.99), name='Si')
sample = limpid.Sample(si, epithermal_correction=True)

e, s, ds = limpid.load_example_data('si')

sample.parameters["lineshape_0"].value = 0.635
sample.parameters["lineshape_1"].value = 0.666

limpid.plot_initial_guess(sample, e, s)

sample.fit(e, s, verbose=True)
print(si.diffusion_length)

limpid.plot_result(sample, show_init=True)
limpid.plot_fractions(sample)
