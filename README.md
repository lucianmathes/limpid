    ██╗     ██╗███╗   ███╗██████╗ ██╗██████╗ 
    ██║     ██║████╗ ████║██╔══██╗██║██╔══██╗
    ██║     ██║██╔████╔██║██████╔╝██║██║  ██║
    ██║     ██║██║╚██╔╝██║██╔═══╝ ██║██║  ██║
    ███████╗██║██║ ╚═╝ ██║██║     ██║██████╔╝
    ╚══════╝╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═════╝ 

# LIMPID -- Layer-wise Investigation of Measurements on Positron Implantation and Diffusion

LIMPID helps you with analyzing your positron annihilation depth profiles.
It will fit the solution of the diffusion equation to your measurement data and thereby determine the positron diffusion length in your sample.

# Installation

## For Users
Via Package index of gitlab.lrz.de 
```
pip install --upgrade pip
pip install limpid --index-url https://gitlab+deploy-token-2044:gldt-mTjywbYYyhsXerAJys29@gitlab.lrz.de/api/v4/projects/113374/packages/pypi/simple
```

## For Developers
Clone the repository then inside the directory run
```
pip install --upgrade pip
pip install --editable .
```

# Getting Started
Learn how to use LIMPID by following these example applications.

## Example 1: A Si Monocrystal
Create a limpid Sample object containing the Si-specific parameters needed [[1]](#1).
```python
import limpid

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.792, 1.99), name='Si')
sample = limpid.Sample(si)
```

Load the example data.
`e` is the positron implantation energy, `s` is the lineshape parameter of the Doppler-broadened 511 keV peak.
```python
e, s = limpid.load_example_data('si')
```

Provide a reasonable first guess and check using a plot.
Note that `s` here is normalized to its last value (S<sub>bulk</sub> = 1).
```python
sample.parameters["lineshape_0"].value = 0.96
sample.parameters["lineshape_1"].value = 1

limpid.initial_guess(sample, e, s)
```

Perform the fit and plot the result.
```python
sample.fit(e, s)
limpid.fit_result(sample)
```

Let's print the resulting diffusion length of positrons in Si.
```python
print(si.diffusion_length)
 >> 21.280334976471796
```

A diffusion length of 21 nm seems way to low for monocrystalline Si.
Let's try again with an epithermal correction.
```python
sample = limpid.Sample(si, epithermal_correction=True)
sample.fit(e, s, verbose=True)
 >> `gtol` termination condition is satisfied.
 >> Function evaluations 35, initial cost 7.5210e-05, final cost 1.0650e-06, first-order optimality 9.75e-09.
 >> 
 >> Fit duration: 3.06078 s
 >> Diffusion length(s):
 >> - Layer 1: (165.18248443 +/- 115.03995652) nm
```

Plot the result (including initial guess).
And also plot the implantation and annihilation fractions where you can see the percentage of epithermal positrons.
```python
limpid.fit_result(sample, show_init=True)
limpid.plot_fractions(sample)
```

You can find the entire script in `examples/example1_si.py`.


## Example 2: A Thin Cu Layer on a Si Substrate
Recreate the physical layers of the sample.
Makhov parameters for a lot of materials have been calculated by [[Dryzek and Horodek, 2008]](#1).
```python
import limpid

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.792, 1.99), name='Si')
cu = limpid.Layer(density=8.96, makhov_parameters=(2.84, 1.67, 1.73), name='Cu')
sample = limpid.Sample([cu, si])
```

Load the example data.
`ds` is the standard deviation of the lineshape parameter `s`.
```python
e, s, ds = limpid.load_example_data('cu-si')
```

Provide a reasonable first guess.
Use accurate estimates where possible and fix known parameters (like the diffusion length of positrons in Si from Example 1) for a more stable fitting result.
Set `sample.parameters["thickness_{i}"].vary = True`, if you want the fit to determine a layer thickness.
```python
sample.parameters["lineshape_0"].value = 0.62
sample.parameters["lineshape_1"].value = 0.57
sample.parameters["lineshape_2"].value = 0.65
sample.parameters["diffusion_length_1"].value = 30
sample.parameters["diffusion_length_2"].value = 165
sample.parameters["diffusion_length_2"].vary = False
sample.parameters["thickness_1"].value = 350
sample.parameters["thickness_1"].vary = True
```

Perform the fit and plot the result.
```python
sample.fit(e, s, ds, verbose=True)
limpid.fit_result(sample, show_init=True)
```

Print a table of all parameters.
```python
sample.parameters.pretty_print()
```

Now retry with an epithermal correction and see, if there are any differences.
You can find the entire script in `examples/example2_cu-si.py`.

## References
<a id="1">[1]</a>
J. Dryzek, "GEANT4 simulation of slow positron beam implantation profiles",
Nucl. Instrum. Methods Phys. Res. B, Vol. 266, Number 18, pp4000.
