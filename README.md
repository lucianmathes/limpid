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

# Example 1: A Thin Cu Layer on a Si Substrate
Recreate the physical layers of the sample.
```python
import limpid

si = limpid.Layer(density=2.33, makhov_parameters=(2.48, 1.792, 1.99))
cu = limpid.Layer(density=8.96, makhov_parameters=(2.84, 1.67, 1.73))
sample = limpid.Sample([cu, si])
```

Load the example data.
```python
e, s, ds = limpid.load_example_data('cu-si')
```

Provide a reasonable first guess and check using a plot.
```python
sample.parameters["lineshape_0"].value = 0.63
sample.parameters["lineshape_1"].value = 0.58
sample.parameters["lineshape_2"].value = 0.64
sample.parameters["diffusion_length_1"].value = 50
sample.parameters["diffusion_length_2"].value = 500

limpid.initial_guess(sample, e, s, ds)
```

Perform the fit. Plot the result.
```python
out = sample.fit(e, s, ds)
limpid.fit_result(sample)
```

Plot implantation (stopping) and annihilation fractions.
```python
limpid.plot_fractions(sample)
```
You can find the entire script in `examples/example1_cu-si.py`.
