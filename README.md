# **SigmaEpsilon.Solid.Fourier** - Fourier solutions of some plate and beam bending problems in Python

![ ](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/blob/main/logo.png?raw=true)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sigma-epsilon/sigmaepsilon.solid.fourier/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/sigma-epsilon/sigmaepsilon.solid.fourier/tree/main)
[![codecov](https://codecov.io/gh/sigma-epsilon/sigmaepsilon.solid.fourier/graph/badge.svg?token=7JKJ3HHSX3)](https://codecov.io/gh/sigma-epsilon/sigmaepsilon.solid.fourier)
[![Documentation Status](https://readthedocs.org/projects/sigmaepsilonsolidfourier/badge/?version=latest)](https://sigmaepsilonsolidfourier.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/sigmaepsilon.solid.fourier)](https://pypi.org/project/sigmaepsilon.solid.fourier/)
[![Python](https://img.shields.io/badge/python-3.10|3.11|3.12-blue)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The `sigmaepsilon.solid.fourier` library provides semi-analytic solutions for certain beam and plate bending problems, where boundary conditions are inherently satisfied through the careful choice of approximating functions. While the calculations are limited to a few boundary conditions, they are considerably faster than, for example, a finite element solution when applicable. This makes the library particularly useful for several purposes:

- experimentation
- verification
- concept validation
- education
- publication

## Highlights

- Semi-analytic solutions of beam and plate problems.
- Easy to use, high level interface to define various kinds of loads.
- Support for arbitrary loads using Monte-Carlo based coefficient determination.
- Industry-grade performance based on highly parallel, performant code.
- Tight integration with popular Python libraries like NumPy, SciPy, xarray, etc.
- A gallery of examples for plotting with Matplotlib for all types of problems.
- A collection of downloadable Jupyter Notebooks ready for execution covering all available functionality.
- Getting Started, User Guide and API Reference in the documentation.
- The library is intensively tested on CircleCI and has a high coverage level (read more about testing below).

## Documentation

The [documentation](https://sigmaepsilonsolidfourier.readthedocs.io/en/latest/) is built with [Sphinx](https://www.sphinx-doc.org/en/master/) using the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html) and hosted on [ReadTheDocs](https://readthedocs.org/).

Check it out for the user guide, installation instructions, an ever growing set of examples, and API Reference.

## Running the Applications

The repository contains a set of applications, where typically a third party library is used to create something interactive and deployable involving some calculations and reporting. Of course, before you can run these examples on your local machine, you have to go through first the installation instructions below.

## Installation

For installation instructions, please refer to the [documentation](https://sigmaepsilonsolidfourier.readthedocs.io/en/latest/).

## How to contribute?

Contributions are currently expected in any the following ways:

- **finding bugs**
  If you run into trouble when using the library and you think it is a bug, feel free to raise an issue.
- **feedback**
  All kinds of ideas are welcome. For instance if you feel like something is still shady (after reading the user guide), we want to know. Be gentle though, the development of the library is financially not supported yet.
- **feature requests**
  Tell us what you think is missing (with realistic expectations).
- **examples**
  If you've done something with the library and you think that it would make for a good example, get in touch with the developers and we will happily inlude it in the documention.
- **sharing is caring**
  If you like the library, share it with your friends or colleagues so they can like it too.

In all cases, read the [contributing guidelines](CONTRIBUTING.md) before you do anything.

## Acknowledgements

**Many of the packages referenced in this document and in the introduction have corresponding research papers that can be cited. If you use them in your work through `sigmaepsilon.solid.fourier`, please take a moment to review their documentation and cite their papers accordingly.**

Additionally, the funding for these libraries often depends on the size of their user base. If your work heavily relies on these libraries, consider showing your support by clicking the :star: button.

## License

This package is licensed under the [MIT license](LICENSE.txt).
