import limpid

# 40 nm Al, 180 nm SiO2, Si substrate

for voltage in [0, 6]:
    al = limpid.Layer(
        density=2.70,
        makhov_parameters=(2.53, 1.748, 2.02),
        name='Al',
        thickness=40,
        potential=0,
    )
    sio2 = limpid.Layer(
        density=2.65,
        makhov_parameters=(2.32, 1.774, 2.65),
        name='SiO2',
        thickness=180,
        relative_permittivity=7,
    )
    si = limpid.Layer(
        density=2.33,
        makhov_parameters=(2.48, 1.73, 1.99),
        name='Si',
        potential=voltage,
    )
    sample = limpid.Sample([al, sio2, si], epithermal_correction=True)

    e, s, ds = limpid.load_example_data(f'al-sio2-si_{voltage}v')

    sample.parameters["lineshape_0"].value = 0.643
    sample.parameters["lineshape_1"].value = 1.159
    sample.parameters["lineshape_2"].value = 0.593
    sample.parameters["lineshape_3"].value = 0.630
    sample.parameters["lineshape_epithermal"].value = 0.506
    sample.parameters["diffusion_length_1"].value = 80.0
    sample.parameters["diffusion_length_2"].value = 41.6
    sample.parameters["diffusion_length_3"].value = 454.4
    sample.parameters["diffusion_length_epithermal"].value = 1

    sample.fit(e, s, ds, verbose=True)
    limpid.plot_result(sample, show_init=True)

    limpid.plot_fractions(sample, savename=f"positron_fractions_{voltage}V.pdf")

    sample.parameters.pretty_print()
