# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

Never read the .env file in the repository.

The commit messages shouldn't include any details on ownership at all.

Never add "Co-Authored-By: Claude..." or similar AI contributor lines to commit messages. Commit messages are not the place to make ownership statements — ownership of code is governed by company policies and agreements between employer and employee, or by the individual repository owner if no organization is involved.

## Project

`sigmaepsilon.solid.fourier` computes semi-analytic Navier (Fourier series) solutions for simply supported beams (Euler-Bernoulli, Timoshenko) and rectangular plates (Kirchhoff-Love, Uflyand-Mindlin). It is part of the `sigmaepsilon.*` namespace package family and depends heavily on `sigmaepsilon.deepdict`, `sigmaepsilon.math`, `sigmaepsilon.mesh` and `sigmaepsilon.solid.material`. Python 3.10–3.12, managed with uv (hatchling build backend). Source lives under `src/sigmaepsilon/solid/fourier/` (namespace package, no `__init__.py` above `fourier/`).

## Commands

```shell
uv sync --group dev --group test --group docs   # dev setup (no group is installed by default)
uv run pytest                                 # all tests
uv run pytest tests/test_plates.py::TestMindlinPlate::test_mindlin_plate_smoke   # single test
uv run pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.solid.fourier   # coverage (CI uses --cov-report=xml)
./run_tests_with_coverage.sh NUMBA_DISABLE_JIT   # coverage with Numba JIT off, so jitted code is measured (uses .coveragerc_nojit)
cd docs && make html                          # Sphinx docs (needs the docs group)
```

- Formatting: `black` (in the dev group). No linter is configured.
- `NUMBA_DISABLE_JIT=1` is also useful when debugging inside jitted functions. The default `.coveragerc` excludes `@njit` code from coverage.
- To work against sibling `sigmaepsilon.*` checkouts in editable mode: `uv pip install -e ../sigmaepsilon.<pkg>` (undone by the next `uv sync`).
- CI (CircleCI) runs tests on Python 3.12; pushes to `main` publish to PyPI and pushes to `test` publish to TestPyPI, so the version in `pyproject.toml` matters on those branches.

## Architecture

The solution pipeline is identical for beams and plates and is spread across stage modules; `NavierBeam.linear_static_analysis` ([beam.py](src/sigmaepsilon/solid/fourier/beam.py)) and `NavierPlate.linear_static_analysis` ([plate.py](src/sigmaepsilon/solid/fourier/plate.py)) are near-mirrors that drive it:

1. **LHS** – `preproc.lhs_Navier` builds per-harmonic stiffness coefficients.
2. **RHS** – each `LoadCase.rhs(problem)` returns Fourier coefficients of the load, shape `(nMN, nComponent)`; these are stacked over `loads.cases()`.
3. **Solve** – `proc.linsolve_*` (diagonal per harmonic, no global system). Bernoulli/Kirchhoff first rescale the RHS via `rhs_Bernoulli`/`rhs_Kirchhoff`.
4. **Postprocess** – `postproc.postproc` evaluates all quantities at the requested points, returning `(nLHS, nRHS, nPoints, nComponents)`.
5. Results are wrapped per load case in `BeamLoadCaseResultLinStat`/`PlateLoadCaseResultLinStat` ([result.py](src/sigmaepsilon/solid/fourier/result.py)) and placed in a locked `DeepDict` that mirrors the `LoadGroup` tree.

Key conventions that span files:

- **Model type is implied by arguments, not a flag.** Supplying shear stiffness (`GA` for beams, `S` for plates) switches to Timoshenko/Mindlin; omitting it gives Bernoulli/Kirchhoff. The low-level dispatchers (`lhs_Navier`, `postproc`) further distinguish beam vs. plate by whether `size`/`shape` are scalars or iterables. `MechanicalModelType` ([enums.py](src/sigmaepsilon/solid/fourier/enums.py)) exposes `is_1d`/`is_2d`, which loads use to pick their RHS routine.
- **Result component order is positional.** Column indices are defined by the `components` lists in `result.py` and must match the constants at the top of `postproc.py` (plates: `UZ, ROTX, ROTY, CX, CY, CXY, EXZ, EYZ, MX, MY, MXY, QX, QY`; beams: `UY, ROTZ, CZ, EXY, MZ, SY`). Changing one requires changing the other. Sign conventions are documented in `docs/source/theory_guide/sign_conventions.md`.
- **Loads** ([loads/](src/sigmaepsilon/solid/fourier/loads/)): `LoadGroup` is a `DeepDict` of nested groups/cases; `LoadCase` subclasses (`PointLoad`, `LineLoad`, `RectangleLoad`, `DiskLoad`) implement `rhs(problem)`. Load values are per DOF (plates: `[F, Mx, My]`; beams: `[F, M]`). If any value is a string expression (e.g. `"x*y"`, parsed with `sigmaepsilon.math.function.Function`), the RHS is computed by Monte Carlo integration ([mc.py](src/sigmaepsilon/solid/fourier/mc.py)) instead of closed form; sample counts and batch sizes come from `Config` ([config.py](src/sigmaepsilon/solid/fourier/config.py)) unless `num_mc` is passed. `DiskLoad` always uses Monte Carlo.
- **Protocols** ([protocols.py](src/sigmaepsilon/solid/fourier/protocols.py)) decouple loads from problems (`NavierProblemProtocol`, `LoadCaseProtocol`, `LoadGroupProtocol`) and are used in `isinstance` checks, e.g. to tell apart loads and points in the positional `*args` of `linear_static_analysis`.
- **Numerics** are Numba `@njit(nogil=True, parallel=True, cache=True)` kernels in `preproc.py`, `postproc.py`, `proc.py` and `mc.py`. Keep them Numba-compatible (plain NumPy, explicit array shapes; inputs are normalized with `atleast*d` and cast to a consistent float/int dtype before the call).

## Docs and tests

- Docstrings use NumPy style. Docs are Sphinx (pydata theme, nbsphinx, myst, bibtex); the example notebooks in `docs/source/examples/` are rendered into the gallery, and `docs/source/theory_guide/` holds the derivations.
- Tests build plate sections with `sigmaepsilon.solid.material` and use `sigmaepsilon.core.testing.SigmaEpsilonTestCase`; most are smoke tests plus input-validation checks.
- `apps/` contains standalone demo applications (FastAPI, Dash, a papermill/PyVista/Azure pipeline) with their own optional dependency groups; they are not part of the package.
