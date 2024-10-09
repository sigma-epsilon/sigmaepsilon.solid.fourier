# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., best follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
