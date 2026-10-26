# Audit of `sigmaepsilon.solid.fourier`

**Date:** 2026-10-26 · **Version audited:** 2.1.3 (`pyproject.toml`), branch `feature/agent_setup` at `7e30bfd`

## Summary

Results for transverse loads (`fz` on plates, `F` on beams) are correct. They match the closed-form beam solutions to 9 digits and the tabulated plate coefficients to all three published figures, and the thin-plate and slender-beam models agree with their shear-deformable counterparts to machine precision.

**Results for moment loads are wrong.** This applies to any load with a non-zero `mx`/`my` on a Kirchhoff plate, or a non-zero `M` on a Bernoulli beam. The sign is flipped, and for plates the magnitude is also wrong, by up to ~5× in the cases tested. The code contradicts the library's own theory guide in these places. In addition, all four models drop the constant part of the shear force caused by moment loads.

Beyond that, the main risks are:

- Memory use grows with *harmonics × points*. Moderate problems need gigabytes.
- The Numba parallelism is mostly ineffective.
- The test suite contains no accuracy checks, so none of the above was caught.
- Several invalid inputs return plausible-looking numbers (NaN, zeros or partial results) without an error.

| Severity | Count | Meaning |
|---|---|---|
| Critical | 4 (all solved in PR #37) | Wrong engineering results without any error being raised |
| High | 6 (all solved in PRs #40 and #42) | Wrong or misleading results in edge cases, or serious scalability limits |
| Medium | 9 | Robustness, API and test-suite problems |
| Low | 14 | Documentation, packaging, CI and repository upkeep |

### How this audit was done

- Read every module in `src/`, all tests, the CI/packaging configuration, the theory guide's derivations and the demo apps.
- Ran the test suite: 27 passed in 26 s.
- Wrote scripts (kept outside the repo) that compare the library against:
  - closed-form solutions (simply supported beam; Timoshenko & Woinowsky-Krieger square-plate coefficients);
  - the slender/thin limits between model pairs: Bernoulli vs Timoshenko with `GA = 1e12`, and Kirchhoff vs Mindlin with `S = 1e9·I`;
  - corrected versions of the suspect functions, patched in at runtime, to confirm that each proposed fix is complete.
- Measured peak memory and thread scaling on an 11-core macOS machine.

Every finding marked **Confirmed** was reproduced by running code. **Unverified** items are flagged as such.

---

## Critical

### C1. Kirchhoff plate: moment loads use the wrong sign and the wrong harmonic index — ✅ Solved

> **Status: Solved** in [PR #37](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/37). Originally: Confirmed.

[`preproc.py:262-266`](src/sigmaepsilon/solid/fourier/preproc.py#L262-L266)

```python
res[i, n] = (
    coeffs[i, n, 0]
    - coeffs[i, n, 1] * cy * (n + 1)   # should be + ... * cy * n_y
    - coeffs[i, n, 2] * cx * (n + 1)   # should be - ... * cx * m_x
)
```

There are two defects:

1. **Wrong index.** `n` here is the *flattened* harmonic index `iMN = (m-1)*N + n-1`, not the mode number in y or x. The multiplier should be the y mode number for `mx` and the x mode number for `my`. The two coincide only when `M = 1` (for `mx`) or `N = 1` (for `my`).
2. **Wrong sign for `mx`.** The theory guide ([`kirchhoff_plates_navier.md:69`](docs/source/theory_guide/kirchhoff_plates_navier.md)) gives an equivalent load of `p_z + ∂p_yy/∂x − ∂p_xx/∂y`. With the cosine-in-y basis that `mx` uses, this becomes `+ mx·(π n/Ly)`.

**Evidence.** Kirchhoff and Mindlin (`S = 1e9·I`) were compared on a 2×3 plate with 12×12 harmonics under a rectangular patch load. The table shows the maximum relative mismatch over the output components.

| Load | Current code | With the fix below |
|---|---|---|
| `fz` | all components match | all match |
| `mx` | 170–350 % | all 13 components match |
| `my` | 440–18 000 % | all 13 components match |

The postprocessing in `postproc_Kirchhoff` is correct once the RHS is fixed.

**Fix:**

```python
m = np.repeat(np.arange(1, M + 1), N); n = np.tile(np.arange(1, N + 1), M)
rhs = f + mx * np.pi * n / Ly - my * np.pi * m / Lx
```

`shape` must then be passed into `rhs_Kirchhoff`.

### C2. Bernoulli beam: the moment-load sign is flipped in both the RHS and the shear force — ✅ Solved

> **Status: Solved** in [PR #37](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/37). Originally: Confirmed.

- [`preproc.py:280`](src/sigmaepsilon/solid/fourier/preproc.py#L280) computes `f − m·c·n`. The theory guide ([`bernoulli_beams_navier.md`, eq. `eq_bernoulli_beamXY_v_coefficient_formula`](docs/source/theory_guide/bernoulli_beams_navier.md)) gives `q_j + r_j·πj/L`.
- [`postproc.py:133`](src/sigmaepsilon/solid/fourier/postproc.py#L133) computes `V = (EI·v·c³ + r)·cos`. The guide gives `(EI·v·c³ − r)·cos`.

**Evidence.** Under a point moment and a distributed moment, Bernoulli returns `UY`, `ROTZ`, `CZ` and `MZ` with exactly the opposite sign to Timoshenko with `GA = 1e12`. The relative mismatch is 2.0 on every one of these components.

The two errors cancel in the shear force, so today `SY` happens to agree with Timoshenko. **Fixing only the RHS makes `SY` wrong** (it oscillates as ±2r). Both lines have to be fixed together. With both fixed, all components match Timoshenko apart from C3 below.

### C3. Every model drops the constant shear force caused by moment loads — ✅ Solved

> **Status: Solved** in [PR #37](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/37). Originally: Confirmed for beams, likely for plates.

Moment loads are expanded in `cos(kπx/L)` for `k ≥ 1`. That basis has no `k = 0` term, so the mean of the moment load, `m̄ = (1/L)∫m dx`, is discarded.

For a simply supported beam, a uniform moment load causes no deflection and no bending moment. It does cause a constant shear force (balanced by the support reactions), and in Timoshenko theory a constant rotation offset of `−m̄/GA`.

**Evidence.** A couple `M0 = 1` was applied at `L/3` on a beam with `L = 10`. The exact shear is `|V| = 0.1` along the whole beam. The library returns `SY ≈ 0` for both Bernoulli and Timoshenko.

**Fix.** Add the analytic `k = 0` correction in postprocessing: `V += m̄`, plus `θ += −m̄/GA` for Timoshenko, with the sign following the guide's convention. The plate cosine bases have the same gap in each direction, so `QX`/`QY` under `mx`/`my` loads are probably also missing a term that is constant in one direction. This needs its own check.

### C4. Results are matched to load cases by position, and the two sequences can differ — ✅ Solved

> **Status: Solved** in [PR #37](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/37). Originally: Confirmed.

[`plate.py:128`](src/sigmaepsilon/solid/fourier/plate.py#L128) / [`plate.py:147-148`](src/sigmaepsilon/solid/fourier/plate.py#L147-L148) (and [`beam.py:170`](src/sigmaepsilon/solid/fourier/beam.py#L170), [`beam.py:189-190`](src/sigmaepsilon/solid/fourier/beam.py#L189-L190)):

- The RHS is built from `loads.cases()`, which yields **only `LoadCase` instances**.
- Results are assigned while iterating `loads.items(deep=True)`, which yields **every leaf**. `LoadGroup` is typed to accept `Any` values.

`LoadGroup(meta="note", A=PointLoad(...), B=PointLoad(...))` currently raises `IndexError`. If a non-load leaf appears *after* the cases, the extra entry is skipped silently. If one appears in the *middle*, cases receive another case's results.

**Fix.** Collect `(address, case)` pairs once, then build the RHS and assign the results from that same list.

---

## High

### H1. Postprocessing memory grows with harmonics × points — ✅ Solved

> **Status: Solved** in [PR #40](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/40). Originally: Confirmed. The kernels now accumulate into the `(nLHS, nRHS, nP, ncomp)` output directly. On the 60×60 / 10 000-point Mindlin problem, the whole process now peaks at 176 MiB of RSS.

Every `postproc_*` kernel allocates `(M·N, nLHS, nRHS, nP, 13)` and only then sums over the harmonic axis ([`postproc.py:70`](src/sigmaepsilon/solid/fourier/postproc.py#L70)).

| Harmonics | Points | Peak memory growth | Size of the returned result |
|---|---|---|---|
| 40×40 | 5 000 | 773 MiB | 0.5 MiB |
| 60×60 | 10 000 | 3.4 GiB | 1.0 MiB |

The FastAPI demo evaluates 30 000 points with harmonic counts chosen by the client, so a request with 100×100 harmonics would try to allocate about 31 GiB.

**Fix.** Accumulate into the `(nLHS, nRHS, nP, ncomp)` output inside the kernel. This also removes the separate `np.sum` pass.

### H2. Numba parallelism is mostly ineffective — ✅ Solved

> **Status: Solved** in [PR #40](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/40). Originally: Confirmed. Every kernel now has a single `prange`: over points in postprocessing, and over harmonics in the pre-processing, solve and Monte Carlo kernels. On the 40×40 / 5 000-point Mindlin problem, the full analysis takes 0.019 s on 11 threads and 0.108 s on 1 thread. Before the change it took 0.241 s and 0.836 s. Part of the single-thread gain comes from computing the y-direction sines and cosines once per point. The results are bit-identical to the previous implementation.

Every kernel nests `prange` loops. Numba parallelises only the outermost one and runs the inner ones serially. The outermost loop is usually `nRHS`, the number of load cases, which is often 1. Measured result: 11 threads take 0.81 s and 1 thread takes 0.93 s on the same 40×40 / 5 000-point problem, a 1.15× speedup.

**Fix.** Put the single `prange` over points (postprocessing) or harmonics (pre- and processing).

### H3. Invalid inputs produce plausible-looking output instead of errors — ✅ Solved

> **Status: Solved** in [PR #42](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/42). Originally: Confirmed. `NavierBeam` and `NavierPlate` now validate `length`/`size`, `N`/`shape`, `EI`, `GA`, `D` and `S` when they are set, and raise for non-scalar beam stiffnesses, for non-zero `D16`/`D26`/`S12`, and for non-symmetric or non-positive stiffness matrices. `linear_static_analysis` requires the points and checks that they lie on the structure. Every load checks that its domain lies within the structure, and beam line loads must not be reversed.

| Input | Behaviour |
|---|---|
| `points` omitted, beam | Returns a `(1, 6)` array of `NaN` |
| `points` omitted, plate | Returns a `(1, 13)` array of zeros |
| `EI=[1000, 2000]` (array) | Uses only the first value; no warning |
| `D` with non-zero `D16`/`D26` | Coupling terms ignored silently; identical result to `D16 = 0`. The Navier solution is only valid for specially orthotropic plates. |
| `S` with off-diagonal terms | Ignored silently |
| Load outside the span, e.g. `PointLoad(2L, …)` | Accepted; returns the odd-periodic extension (here ≈ 0) |
| `size`, `shape`, `D`, `S` shapes and signs | Never validated |

**Fix.** Validate in the constructors and in `linear_static_analysis`. Raise when `D16`, `D26` or `S12` is non-zero, or at least warn.

### H4. Monte Carlo drops a final batch of exactly one sample — ✅ Solved

> **Status: Solved** in [PR #42](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/42), together with [issue #39](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/issues/39). Originally: Confirmed. Monte Carlo is now an opt-in method, and its sampler (`mc.monte_carlo_samples`) always emits the remaining samples as a last batch.

[`mc.py:77`](src/sigmaepsilon/solid/fourier/mc.py#L77) and [`mc.py:177`](src/sigmaepsilon/solid/fourier/mc.py#L177) use `if remaining_MC_points > 1`, but the sum is still divided by the full `num_MC_samples`. With `num_mc=5001`, the first coefficient came out 3.0 % low (1.572 vs exact 1.621). The comparison should be `> 0`.

### H5. Monte Carlo results are not reproducible and use the global RNG — ✅ Solved

> **Status: Solved** in [PR #42](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/42), together with [issue #39](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/issues/39). Originally: Confirmed. Load coefficients are now integrated with composite Gauss–Legendre quadrature by default (tensor-product on rectangles, polar coordinates on disks), with node counts derived from the number of harmonics, and projected onto the Fourier basis with matrix products. They match the closed forms to machine precision, and on 40×40 harmonics they are about 60× faster than the previous 100 000-sample Monte Carlo default. Monte Carlo remains available (`integration="mc"`, or by passing `num_mc`), and takes an `rng` seed or `numpy.random.Generator`; the global random state is no longer used.

`utils.generate_random_points_*` call `np.random.uniform`. Two identical calls gave 1.6220 and 1.6199. This also changes the caller's global random state.

The integration domains are all rectangles, line segments or disks, so Gauss–Legendre quadrature (polar coordinates for disks) would give deterministic and far more accurate coefficients at lower cost. Short of that, accept a `seed` or a `numpy.random.Generator`.

### H6. Beam line loads ignore the beam Monte Carlo settings — ✅ Solved

> **Status: Solved** in [PR #42](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/42). Originally: Confirmed by reading. Beam line loads now use `Config.NUM_MC_SAMPLES_BEAM` and `Config.MC_BATCH_SIZE_BEAM`.

[`loads/lineload.py:57`](src/sigmaepsilon/solid/fourier/loads/lineload.py#L57) uses `Config.NUM_MC_SAMPLES_PLATE` for beams as well, so `Config.NUM_MC_SAMPLES_BEAM` is never used for line loads.

---

## Medium

- **M1. No accuracy tests.** All 27 tests are smoke tests, shape checks or input-validation tests. None compares a result with a reference value, which is why C1–C3 went unnoticed. Recommended additions:
  - closed-form beam and plate cases;
  - parametrised slender/thin-limit consistency tests (the comparison used in this audit), covering every load type and DOF;
  - convergence as harmonics are added;
  - Monte Carlo vs closed-form coefficients for constant loads.
- **M2. Input validation uses `assert`.** See [`lineload.py:60-72`](src/sigmaepsilon/solid/fourier/loads/lineload.py#L60-L72), [`rectangleload.py:62`](src/sigmaepsilon/solid/fourier/loads/rectangleload.py#L62) and [`diskload.py:47`](src/sigmaepsilon/solid/fourier/loads/diskload.py#L47). The checks disappear under `python -O`, and the tests rely on `AssertionError`. Use `ValueError`/`TypeError`.
- **M3. NumPy integer load values crash.** In [`rectangleload.py:68`](src/sigmaepsilon/solid/fourier/loads/rectangleload.py#L68) (and `utils.is_scalar`), `np.int64` is not an `int` subclass, so `np.array([1, 0, 0])` is treated as a symbolic expression and raises `ValueError: Invalid value 1`. Check against `numbers.Real` instead.
- **M4. Integer evaluation points are rejected on beams.** [`beam.py:153`](src/sigmaepsilon/solid/fourier/beam.py#L153) checks for `(float, Iterable)`, so `linear_static_analysis(5, loads)` raises `TypeError`.
- **M5. Abstract methods are not enforced.** `LoadCase` and `NavierProblem` use `@abstractmethod` without `ABC`, so incomplete subclasses can be instantiated.
- **M6. ✅ Solved in [PR #42](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/42). Constant-value Monte Carlo closures work only by accident.** In `mc.py:56` and `:156`, `def function(points): ... values[i]` captures the loop variable `i` late. It gives the right value only because the later loop that calls these functions reuses the same variable name `i`. Bind it explicitly (`lambda p, v=values[i]: ...`).
- **M7. The multi-stiffness capability is half exposed.** Every kernel supports `nLHS > 1`, but the public API always takes `res[0, ...]`. Either expose it properly (useful for parameter sweeps) or remove it from the kernels.
- **M8. `linear_static_analysis` argument parsing is duplicated** between `NavierBeam` and `NavierPlate`. It belongs in `NavierProblem`, along with the shared RHS/solve/result-assembly code.
- **M9. The order of shear strains in `strains` is unverified.** The result's `strains` slice returns `[CX, CY, CXY, EXZ, EYZ]` and the FastAPI app passes it to `MindlinPlateSection.utilization`. `sigmaepsilon.solid.material` orders shear components as `yz, xz` elsewhere. For isotropic single-layer sections the order makes no difference. For orthotropic or layered sections it could swap the shear-stress profiles. **Unverified**; worth a dedicated test.

---

## Low

### Documentation and docstrings

- [`beam.py:22`](src/sigmaepsilon/solid/fourier/beam.py#L22): `NavierBeam` is described as handling "simply-supported **plates**".
- [`preproc.py:43`](src/sigmaepsilon/solid/fourier/preproc.py#L43): says shear stiffness applies to "Euler-Bernoulli beams"; it means Timoshenko.
- `postproc_Kirchhoff` docstring lists index 8 as "moment y" (and 9 as "moment y" again).
- `postproc_Bernoulli` docstring describes `EI` as a 3d `(nLHS, 3, 3)` array; it is 1d.
- The `postproc_Mindlin`/`Kirchhoff` docstrings give the solution shape as `(nRHS, nLHS, …)`; the code uses `(nLHS, nRHS, …)`.
- The `rhs()` docstrings for `PointLoad` and `LineLoad` say the return shape is `(H, 3)`; beams return `(N, 2)`. They also mention "the attached problem" when `problem` is required.
- `RectangleLoad.domain` is documented as "Default is `None`" but is required.
- Spelling: `num_MC_bacthes`, "ponts", "axpis", "rectagle", "one ore more".

### API tidiness

- `LoadGroup.cooperative` is stored but never read anywhere.
- `NavierProblemProtocol.shape` is a property declared with a `value` parameter, and `result_class` is annotated `ClassVar[LoadCaseResultLinStat]` instead of `type[...]`.
- `LoadCaseResultLinStat.__slots__` lists an unused `_components`. The subclasses don't declare `__slots__`, so they get a `__dict__` anyway.
- `lhs_Navier(**kw)` silently swallows unknown keyword arguments.

### Packaging, CI and repository

- Several upkeep issues:
    - **Changelog.** `CHANGELOG.md` stops at 2.0.0, while the package is at 2.1.3.
    - **Broken license links.** README and CONTRIBUTING link to `LICENSE.txt`; the file is `LICENSE`.
    - **Stale version in editable installs.** `__version__` comes from installed metadata, so an editable install that hasn't been reinstalled reports an old version. The local environment reports 2.1.0. `docs/source/conf.py` uses this value too.
    - **CI test coverage.** CircleCI tests only Python 3.12, although the classifiers claim 3.10–3.12.
    - **CI test results.** CircleCI stores `test-results`, but pytest writes no JUnit XML there.
    - **No linting.** CI runs no lint, format or type checks.
    - **Python 3.13+ excluded.** `python = ">=3.10,<3.13"` blocks 3.13+; check whether Numba and the `sigmaepsilon.*` dependencies still need this cap.
    - **Dependabot.** `.github/dependabot.yml` is copied boilerplate that does nothing useful:
      - it allows `sigmaepsilon.material`, which is not a dependency (the real one is `sigmaepsilon.solid.material`);
      - it treats `xarray` as a production dependency;
      - `reviewers: "Bence Balogh"` is not a GitHub login;
      - the comments mention Django, npm and Japan.
- Demo apps in `apps/` (not part of the package, but published in the repo):
    - `apps/fem_dash/app.py` is empty.
    - `apps/fastapi/main.py` builds file paths from the client-supplied `job_id` without sanitising it, which allows path traversal (`"../../x"`). It also accepts unbounded `nx`/`ny`, which combined with H1 lets one request exhaust memory, and has no authentication. That's acceptable for a local demo, but it should say so in its README.

---

## Verified as correct

These parts were checked and need no change:

- **Bernoulli beam, transverse loads.** Midspan deflection under a point load and under a UDL matches `PL³/48EI` and `5qL⁴/384EI` to 9 digits with 400 terms. Moments match `PL/4` and `qL²/8` (the negative sign is consistent with the guide's convention `M_z = EI·v''`).
- **Kirchhoff square plate, UDL, ν = 0.3.** `w = 0.004062·qa⁴/D` and `Mx = 0.04789·qa²`, matching Timoshenko & Woinowsky-Krieger (0.00406 and 0.0479).
- **Kirchhoff postprocessing** (all 13 components, including the `QX`/`QY` moment-load terms) agrees with Mindlin in the thin limit once C1 is fixed.
- **Mindlin and Timoshenko kernels.** The stiffness matrices are symmetric and agree with the postprocessing formulas. A hand derivation of the transverse equilibrium row matches `lhs_Navier_Mindlin`. The order of `S44`/`S55` matches the `yz, xz` convention of `sigmaepsilon.solid.material`.
- **Closed-form load coefficients** (point, constant line on beams, constant rectangle) agree with their Monte Carlo counterparts within sampling error.

## Recommended order of work

1. ✅ **Fix C1, C2 and C4.** *(Done in PR #37.)* Each is a few lines. Add the slender/thin-limit consistency tests from M1 at the same time so they cannot come back.
2. **Release a patch version** whose changelog warns that results for moment loads on Kirchhoff plates and Bernoulli beams were wrong in earlier versions.
3. ✅ **Fix C3 and verify the plate equivalent of the missing shear term.** *(Done in PR #37.)*
4. ✅ **Fix H1 and H2 together**, by rewriting the postprocessing kernels to accumulate in place with one parallel loop over points. *(Done in PR #40.)*
5. ✅ **Fix H3–H6:** input validation and the Monte Carlo fixes, ideally replacing Monte Carlo with quadrature. *(Done in [PR #42](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/pull/42); Monte Carlo is kept as an option.)*
6. **Fix the medium and low items** opportunistically.
