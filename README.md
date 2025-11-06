    ██╗     ██╗███╗   ███╗██████╗ ██╗██████╗ 
    ██║     ██║████╗ ████║██╔══██╗██║██╔══██╗
    ██║     ██║██╔████╔██║██████╔╝██║██║  ██║
    ██║     ██║██║╚██╔╝██║██╔═══╝ ██║██║  ██║
    ███████╗██║██║ ╚═╝ ██║██║     ██║██████╔╝
    ╚══════╝╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═════╝ 

# LIMPID -- Layer-wise Investigation of Measurements on Positron Implantation and Diffusion

LIMPID helps you with analyzing your positron annihilation depth profiles.
It will fit the solution of the diffusion equation to your measurement data and thereby determine the positron diffusion length in your sample.
For an explanation of the LIMPID algorithm and for citation purposes, please refer to [our paper](https://arxiv.org/abs/2511.02889).

# Installation
```
pip install --upgrade pip
pip install limpid
```

# Getting Started

Learn how to use LIMPID by following these example applications.

## Example 1: A Si Monocrystal

Create a limpid Sample object containing the Si-specific parameters needed ([Dryzek, 2008](#references)).
```python
import limpid

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.73, 1.99), name='Si')
sample = limpid.Sample(si)
```

Load the example data.
`e` is the positron implantation energy, `s` is the lineshape parameter of the Doppler-broadened 511 keV peak.
```python
e, s = limpid.load_example_data('si')
```

Provide a reasonable first guess and check using a plot.
```python
sample.parameters["lineshape_0"].value = 0.635
sample.parameters["lineshape_1"].value = 0.666

limpid.plot_initial_guess(sample, e, s)
```

Perform the fit and plot the result.
```python
sample.fit(e, s)
limpid.plot_result(sample)
```

Let's print the resulting diffusion length of positrons in Si.
```python
print(si.diffusion_length)
 >> 390.6668158143418
```

If you look closely, the fit does not match the data at implantation energies below 2 keV.
Let's try again with an epithermal correction.
```python
sample = limpid.Sample(si, epithermal_correction=True)
sample.fit(e, s, verbose=True)
 >> `gtol` termination condition is satisfied.
 >> Function evaluations 7, initial cost 2.0814e-03, final cost 1.0762e-05, first-order optimality 2.48e-09.
 >> 
 >> Fit duration: 2.49621 s
 >> Diffusion length(s):
 >> - Si: (372.49612987 +/- 10.21031932) nm
```

Plot the result (including initial guess).
And also plot the implantation and annihilation fractions where you can see the percentage of epithermal positrons.
```python
limpid.plot_result(sample, show_init=True)
limpid.plot_fractions(sample)
```

You can find the entire script in `examples/example1_si.py`.


## Example 2: A Thin Cu Layer on a Si Substrate

Recreate the physical layers of the sample.
Makhov parameters for a lot of materials have been calculated by ([Dryzek, 2008](#references)).
```python
import limpid

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.73, 1.99), name='Si')
cu = limpid.Layer(density=8.96, makhov_parameters=(2.84, 1.67, 1.73), name='Cu')
sample = limpid.Sample([cu, si])
```

Load the example data.
`ds` is the standard deviation of the lineshape parameter `s`.
```python
e, s, ds = limpid.load_example_data('cu-si')
```

Provide a reasonable first guess.
Use accurate estimates where possible and fix known parameters (like the diffusion length of positrons in the Si substrate from Example 1) for a more stable fitting result.
Set `sample.parameters["thickness_{i}"].vary = True`, if you want the fit to determine a layer thickness.
```python
sample.parameters["lineshape_0"].value = 0.62
sample.parameters["lineshape_1"].value = 0.57
sample.parameters["lineshape_2"].value = 0.65
sample.parameters["diffusion_length_1"].value = 30
sample.parameters["diffusion_length_2"].value = 372
sample.parameters["diffusion_length_2"].vary = False
sample.parameters["thickness_1"].value = 350
sample.parameters["thickness_1"].vary = True
```

Perform the fit and plot the result.
```python
sample.fit(e, s, ds, verbose=True)
limpid.plot_result(sample, show_init=True)
```

Print a table of all parameters.
```python
sample.parameters.pretty_print()
```

Now retry with an epithermal correction and see, if there are any differences.
Also try to fix the Si lineshape parameter to the value obtained by Example 1 and see if that makes the fit more stable (faster convergence/smaller errors).
You can find the entire script in `examples/example2_cu-si.py`.

Note further, that different materials have different positron affinities ([Puska, 1989](#references)).
In layered systems this can result in preferred diffusion directions.
Try it and compare the positron annihilation fractions with and without affinities.
```python
si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.73, 1.99), positron_affinity=-6.95, name='Si')
cu = limpid.Layer(density=8.96, makhov_parameters=(2.84, 1.67, 1.73), positron_affinity=-4.81, name='Cu')

 [...]

limpid.plot_fractions(sample)
```

## References

- J. Dryzek (2008), "GEANT4 simulation of slow positron beam implantation profiles", Nucl. Instrum. Methods Phys. Res. B, Vol. 266, Number 18, pp4000
- M. Puska (1989), "Positron affinities for elemental metals", J. Phys.: Condens. Matter, Vol. 1, Number 35, pp6081
