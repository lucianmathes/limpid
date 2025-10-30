# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-10-30
### Added
- Warning if positive positron affinities are provided.
- Colors option for `plot_fractions`.

### Changed
- Use analytically correct matrix description of the Markov process.
- Use hyperbolic functions in first diffusion step to avoid numerical cancellation.
- Use correct hyperbolic functions in second diffusion step and reduce calculations.
- Cut unnecessary (u x D) factor from diffusion and annihilation rates.
- Add necessary (u x D) factor to the Boltzmann factor.
- Fix bug where data was missing in `plot_initial_guess`.
- Correct (non-cumulative) fraction plots for the case of epithermal correction.
- Cleaner appearance of fraction plots.
- Fix epithermal scattering length to 1 nm by default.

### Removed
- Legacy algorithm.
- Broken C implementation for faster implantation profile integration.

## [0.3.2] - 2025-07-21
### Added
- Option to suppress pop-up image of the `plot_result` function.

### Changed
- More detailed axis labelling in plots.

## [0.3.1] - 2025-05-22
### Added
- Clearly indicate units of implantation energy in docstrings.

### Changed
- Fix bug with reporting to file, if fit did not converge.
- Fix bug in plotting multi-layer implantation profiles.

## [0.3.0] - 2025-04-09
### Added
- Citations regarding the epithermal correction method.
- A warning is now printed even in non-verbose mode, if the fit did not succeed.

### Changed
- Layer thickness is now varied by default, if no value is specified.
- More detailed Sample class docstring.
- Renamed functions for visualization. All plotting functions now start with `plot_`.
- Fix epithermal fractions at very low energies by cutting the integration range.
- Parameters object now contains Stderr values after fitting.

### Removed
- Unused tests.

## [0.2.2] - 2025-01-09
### Added
- Second example with dataset.
- Dryzek citation for Makhov parameters in the README.
- Option to fit data without delta/standard deviation.
- Option to pass a single Layer object to Sample.

### Changed
- Added missing import to `__init__.py`.
- Avoid division inf / inf when calculating diffusion to layer boundaries.
- Increase max. number of function evaluations to 200.
- Fix missing initial guess in plots of fits using epithermal correction.
- Use finely-spaced energy for fraction plots instead of the energy provided by the data.

### Removed
- Unused data in the examples folder.

## [0.2.1] - 2025-01-08
### Added
- Script for Example 1 referenced in the README.

## [0.2.0] - 2025-01-08
### Added
- Option to plot the initial guess.
- Option to plot implantation and annihilation fractions non-cumulative.
- Example data that can be loaded using the name of the dataset.
- README section on a simple application using example data.

### Changed
- Fix diffusion rates at very low energies by cutting the integration range.
- Include the initial guess in the result plot by default.
- All functions of `visualize.py` and the `example_data` submodule can now be accessed like top-level functions of limpid.

## [0.1.5] - 2024-12-10
### Added
- Documentation for every class and function in limpid.py
- Installation instructions for development.

### Changed
- Fix the handling of unsuccessful fits.
- Move `data/` folder into `examples/` directory.
- Version is now only defined in `__init__.py`. Setuptools can read it from there.
- Fix a bug in the calculation of diffusion rates/fractions that would result in negative concentrations.

### Removed
- Deploy of branches other than main

## [0.1.1] - 2022-10-10
### Added
- Python packaging stuff.
- Continuous integration on GitLab.
- README section on using the C library.
- A unittest that actually tests nothing.
- Include version number in `__init__.py`.
- Gitignore file.
- Usage example including data of a Cu layer on Si.

### Changed
- Move testing scripts to separate folder.
- Fix a bug with `pkg_resources`.

## [0.1.0] - 2022-06-08
### Added
- Initital commit.
