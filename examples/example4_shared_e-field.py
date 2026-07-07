import limpid
import lmfit
import matplotlib.pyplot as plt

# 40 nm Al, 180 nm SiO2, Si substrate

samples = []
energies = []
lineshapes = []

# fig, ax = plt.subplots()

for voltage in [0, 6]:
    al = limpid.Layer(
        density=2.70,
        makhov_parameters=(2.53, 1.748, 2.02),
        name='Al',
        thickness=40,
        potential=voltage,
    )
    sio2 = limpid.Layer(
        density=2.65,
        makhov_parameters=(2.32, 1.774, 2.65),
        name='SiO2',
        thickness=180,
        relative_permittivity=7,
    )
    interface = limpid.Layer(
        density=2.33,
        makhov_parameters=(2.48, 1.73, 1.99),
        name='Si',
        potential=0,
        thickness=0.1,
    )
    si = limpid.Layer(
        density=2.33,
        makhov_parameters=(2.48, 1.73, 1.99),
        name='Si',
        thickness=1000,
        relative_permittivity=1,
    )
    sub = limpid.Layer(
        density=2.33,
        makhov_parameters=(2.48, 1.73, 1.99),
        name='Substrate',
        potential=-1+voltage,
    )
    sample = limpid.Sample((al, sio2, interface, si, sub))

    samples.append(sample)

    e, s, _ = limpid.load_example_data(f'al-sio2-si_{voltage}v')

    sample.parameters["lineshape_0"].set(0.6211)
    sample.parameters["lineshape_1"].set(0.7441)
    sample.parameters["lineshape_2"].set(0.6082)
    sample.parameters["lineshape_3"].set(0.625, False)
    sample.parameters["lineshape_4"].set(0.625, False)
    sample.parameters["lineshape_5"].set(0.625, False)
    sample.parameters["diffusion_length_1"].set(25.)
    sample.parameters["diffusion_length_2"].set(1.)
    sample.parameters["diffusion_length_3"].set(140, False)
    sample.parameters["diffusion_length_4"].set(140, False)
    sample.parameters["diffusion_length_5"].set(140, False)
    sample.parameters["electrical_mobility_2"].set(1e10)
    sample.parameters["electrical_mobility_4"].set(1e10)

    # limpid.plot_initial_guess(sample, e, s)

    # sample.model_diffusion(e)
    # limpid.plot_fractions(sample, show=True)

    # sample.fit(e, s, max_nfev=10000)

    # sample.parameters.pretty_print()
    # limpid.plot_result(sample, show_init=False)

    energies.append(e)
    lineshapes.append(s)

# plt.show()
# quit()

shared_params = lmfit.Parameters()

shared_params.add("lineshape_0", 0.6211)
shared_params.add("lineshape_1", 0.7441)
shared_params.add("lineshape_2", 0.6082)
shared_params.add("lineshape_3", 0.625, False)
shared_params.add("lineshape_4", 0.625, False)
shared_params.add("lineshape_5", 0.625, False)
# shared_params.add("lineshape_epithermal", 0.5386)
shared_params.add("diffusion_length_1", 464, min=0)
shared_params.add("diffusion_length_2", 5., min=0)
shared_params.add("diffusion_length_3", 140, False)
shared_params.add("diffusion_length_4", 140, False)
shared_params.add("diffusion_length_5", 140, False)
shared_params.add("electrical_mobility_2", 1e10, min=0)
# shared_params.add("electrical_mobility_4", 1e10, min=0)
# shared_params.add("diffusion_length_epithermal", 1, False)

limpid.limpid.shared_fit(
    samples, energies, lineshapes, shared_params=shared_params, max_nfev=10000
)

for sample in samples:
    sample.parameters.pretty_print()
    limpid.plot_result(sample, show_init=False)
