import limpid


si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.792, 1.99), name='Si')
sample = limpid.Sample(si, epithermal_correction=True)

e, s = limpid.load_example_data('si')

sample.parameters["lineshape_0"].value = 0.96
sample.parameters["lineshape_1"].value = 1

limpid.plot_initial_guess(sample, e, s)

sample.fit(e, s, verbose=True)

limpid.plot_result(sample, show_init=True)
limpid.plot_fractions(sample)
