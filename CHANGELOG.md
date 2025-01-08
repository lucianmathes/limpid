# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-08
### Added
- Option to plot the initial guess.
- Option to plot implantation and annihilation fractions non-cumulative.
- Example data that can be loaded using the name of the dataset.
- README section on a simple application using example data.

### Changed
- Fix diffusion rates at very low energies by cutting the integration range.
- Include the initial guess in the result plot by default.
- All functions of visualize.py and the example_data submodule can now be accessed like top-level functions of limpid.

## [0.1.5] - 2024-12-10
### Added
- Documentation for every class and function in limpid.py
- Installation instructions for development.

### Changed
- Fix the handling of unsuccessful fits.
- Move data/ folder into examples/ directory.
- Version is now only defined in __init__.py. Setuptools can read it from there.
- Fix a bug in the calculation of diffusion rates/fractions that would result in negative concentrations.

### Removed
- Deploy of branches other than main

## [0.1.1] - 2022-10-10
### Added
- Python packaging stuff.
- Continuous integration on GitLab.
- README section on using the C library.
- A unittest that actually tests nothing.
- Include version number in __init__.py
- Gitignore file.
- Usage example including data of a Cu layer on Si.

### Changed
- Move testing scripts to separate folder.
- Fix a bug with pkg_resources.

## [0.1.0] - 2022-06-08
### Added
- Initital commit.
