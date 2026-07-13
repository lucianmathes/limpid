import limpid
import lmfit

# Silicon single crystal

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.729, 1.99), name='Si')
sample1 = limpid.Sample(si, epithermal_correction=True)

e1, s1, ds1 = limpid.load_example_data('si')

sample1.parameters["lineshape_0"].value = 0.635
sample1.parameters["lineshape_1"].value = 0.666

# Cu on Si

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.73, 1.99), name='Si', positron_affinity=-6.95)
cu = limpid.Layer(density=8.96, makhov_parameters=(2.84, 1.67, 1.73), name='Cu', positron_affinity=-4.81)
sample2 = limpid.Sample([cu, si], epithermal_correction=True)

e2, s2, ds2 = limpid.load_example_data('cu-si')

sample2.parameters["diffusion_length_epithermal"].value = 1
sample2.parameters["diffusion_length_epithermal"].vary = False
sample2.parameters["lineshape_0"].value = 0.62
sample2.parameters["lineshape_1"].value = 0.57
sample2.parameters["lineshape_2"].value = 0.6659
sample2.parameters["diffusion_length_1"].value = 30
sample2.parameters["diffusion_length_2"].value = 372
sample2.parameters["diffusion_length_2"].vary = False
sample2.parameters["thickness_1"].value = 350
sample2.parameters["thickness_1"].vary = True

shared_params = lmfit.Parameters()

shared_params.add("lineshape_0", 0.63)

limpid.shared_fit(
    [sample1, sample2], [e1, e2], [s1, s2], shared_params=shared_params
)

for sample in [sample1, sample2]:
    sample.parameters.pretty_print()
    limpid.plot_result(sample, show_init=True)
