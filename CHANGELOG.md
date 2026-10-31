# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., best follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.3.0] - 2026-10-31

### Changed

- Relaxed the dependency bounds, so that the package installs next to preinstalled packages (e.g. on Databricks): no upper caps, and the minimums are the oldest versions the test suite passes with (checked by a lowest-direct CI job). NumPy 2 is supported, which no earlier version could be installed with.
- Python 3.13 is supported, and `requires-python` is `>=3.10` without an upper bound.
- Requires `sigmaepsilon.deepdict>=3.1.0`, `sigmaepsilon.math>=2.6.0`, `sigmaepsilon.mesh>=3.3.0` and `sigmaepsilon.solid.material>=1.2.0`. The directly imported NumPy and Numba are declared as dependencies.
- `sigmaepsilon.solid.material` 1.2.0 uses the plane stress reduced stiffness for plates, so stiffness matrices of plate sections built with it are smaller than before (by 18% for an isotropic material with a Poisson's ratio of 0.3), and so are the results that depend on them.
- Plates are postprocessed with dense matrix products (BLAS) instead of the Numba kernels, about 2× faster on scattered points. If the points form a full or partial tensor-product grid, in any order, the results are evaluated on the grid, which is 20× faster or more. The Numba kernels remain available with `Config.POSTPROC_METHOD = "numba"`.

### Added

- `Config.POSTPROC_METHOD` and `Config.POSTPROC_BUFFER_SIZE`.

## [2.2.0] - 2026-10-26

**Results for moment loads were wrong in earlier versions.** If you applied non-zero moment loads (`mx`/`my` on plates, `M` on beams), recompute your results.

### Fixed

- Kirchhoff-Love plates: moment loads used the wrong sign for `mx` and the wrong harmonic index for both `mx` and `my`.
- Euler-Bernoulli beams: the sign of moment loads was flipped, both in the solution and in the shear force.
- Plates: concentrated moments (`PointLoad` with non-zero `mx`/`my`) were expanded in the wrong basis functions.
- All models: the constant part of moment loads (the zero-order term of their cosine series) was dropped, and with it the shear forces that balance it (and the related rotations for Timoshenko-Ehrenfest beams and Uflyand-Mindlin plates).
- Results were matched to load cases by position. Items of a `LoadGroup` that are not load cases caused errors or results being assigned to the wrong load case.
- Sign of the moment loads in the Kirchhoff-Love plate theory guide.
- `LoadCase.eval_approx` left out the zero-order terms of moment loads, e.g. it returned zero for a uniform moment load.

### Added

- `LoadCase.rhs0`, returning the zero-order coefficients of moment loads.
- `LoadGroup.cases` accepts `return_address=True` to yield `(address, case)` pairs.
- Accuracy tests against closed-form solutions and between the thin/slender and shear-deformable models.

## [2.0.0] - 2024-10-9

Version 2.0.0 comes with a completely renewed documentation.

### Added

- Protocols to avoid circular references and improve typing.
- Dedicated classes `BeamLoadCaseResultLinStat` and `PlateLoadCaseResultLinStat` to store results from linear static analysis. Instances can be turned into `xarray` instances.
- Support for Python 3.12.
- 100% coverage of the user facing part of the API.
- Lots of examples as interactive Jupyter Notebooks (see the documentation).

### Changed

- `xarray` is no longer a direct dependency, but it is still supported.
- Changed definition of load cases.
- Changes definition of beam and plate problems.
- Renamed `RectangularPlate` to `NavierPlate`.
- Renamed `Problem.solve` to `Problem.linear_static_analysis`, also changed the order of arguments.

## [1.0.0] - 2023-09-27

This is the first release.

### Added

- Linear elastic analysis of Euler-Bernoulli and Timoshenko-Ehrenfest beams
- Linear elastic analysis of Kirchhoff-Love and Uflyand-Mindlin plates
- Support for concentrated and distributed loads
