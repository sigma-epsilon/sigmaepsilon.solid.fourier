
# Navier's Solution for Simply-Supported Timoshenko-Ehrenfest Beams

The method presented here closely follows the solution for Euler-Bernoulli beams, with differences arising solely from the distinct assumptions underlying each beam theory. It is assumed that the reader is already familiar with the Euler-Bernoulli beam solution.

## Behavior of Timoshenko-Ehrenfest beams under static loads

The Timoshenko-Ehrenfest beam theory {cite:p}`enwiki:1321454082, doi:10.1177/1081286519856931` extends classical beam theory, originally developed by Stephen Timoshenko and Paul Ehrenfest in the early 20th century, by accounting for both shear deformation and rotational bending effects. Unlike the simpler Euler-Bernoulli theory—which assumes that plane sections remain plane and perpendicular to the neutral axis—Timoshenko's theory allows for shear strains. This makes it more accurate for short or deep beams, or for beams made from materials with a low shear modulus. The theory provides a more comprehensive description of beam behavior, especially when shear deformation effects are significant, and is widely used in engineering applications requiring higher-fidelity structural modeling.

The Timoshenko-Ehrenfest beam theory is based on the following assumptions:

1. Small strains and small displacements
2. Linear elastic material behavior
3. Constant cross-section along the beam axis
4. No coupling between bending and torsion

### Equilibrium equations

Considering an infinitesimal beam segment in the $x$-$y$ plane and applying the equations of static equilibrium, we obtain

```{math}
:label: eq_mz_beamXY_timoshenko
\sum M_z: \, \frac{d M_z(x)}{dx} + V_y(x) + p_{zz}(x) = 0
```

for moment equilibrium about the $z$-axis at position $x$, and

```{math}
:label: eq_fy_beamXY_timoshenko
\sum F_y: \, \frac{d V_y(x)}{dx} + p_{y}(x) = 0
```

for vertical equilibrium along the $y$-axis, where

- $M_z$ is the moment acting counter-clockwise about $+x$,
- $V_y$ is the shear force acting along $+y$,
- $p_{zz}$ is the distributed moment load,
- $p_{y}$ is the distributed vertical load.

The stress resultants $M_z$ and $V_y$ are defined as

```{math}
:label: eq_def_Mz_beamXY_timoshenko
M_z(x) = \int_A -y \, \sigma_x(x) dA
```

and

```{math}
:label: eq_def_Vy_beamXY_timoshenko
V_y(x) = \int_A \tau_{xy}(x) dA.
```

```{note}
In the Euler-Bernoulli beam theory, equations {eq}`eq_mz_beamXY_timoshenko` and {eq}`eq_fy_beamXY_timoshenko` are typically combined into a single expression. Here, however, we retain them as two separate equations.
```

### Geometric equations

Beam theory describes the motion of a fundamentally three-dimensional object using a one-dimensional domain. This requires a mapping between the two systems. Defining a rotation $\vartheta_z$ such that a positive value corresponds to a counter-clockwise rotation about $+x$, we have

```{math}
:label: eq_geom_timoshenko_beamXY
u(x, y, z) = -y \, \vartheta_z(x).
```

Assuming small strains, we obtain

```{math}
:label: eq_geom_epsxx_timoshenko_beamXY
\varepsilon_x (x,y,z) = \frac{d u(x,y,z)}{dx} = -y \, \frac{d \vartheta_z(x)}{dx}
```

and

```{math}
:label: eq_geom_epsxy_timoshenko_beamXY
\gamma_{xy}(x,y,z) = \frac{d u(x,y,z)}{dy} + \frac{d v(x,y,z)}{dx} = \frac{d v(x,y,z)}{dx} - \vartheta_z(x)
```

for the nonzero engineering strain components of the small-strain tensor, where equation {eq}`eq_geom_timoshenko_beamXY` is used on the right side of {eq}`eq_geom_epsxy_timoshenko_beamXY`.

```{note}
In the Euler-Bernoulli approach, the assumption that cross-sections remain perpendicular to the neutral axis leads to setting {eq}`eq_geom_epsxy_timoshenko_beamXY` to zero. Since the Timoshenko-Ehrenfest theory relaxes this assumption, we do not make this simplification here.
```

### Material equations

Assuming a linear elastic material without initial stresses (the undeformed configuration is stress free), the material equations are as follows

```{math}
:label: eq_mat_timoshenko_beamXY
\begin{align*}
\varepsilon_x (x,y,z) &= E \, \sigma_x (x,y,z), \\
\gamma_{xy} (x,y,z) &= \overline{G} \, \tau_{xy} (x,y,z)
\end{align*}
```

where $E$ is Young's modulus of elasticity and $\overline{G}$ is the shear modulus, with the effect of the appropriate shear correction factor included. If we combine {eq}`eq_def_Mz_beamXY_timoshenko`, {eq}`eq_def_Vy_beamXY_timoshenko` with the material equations {eq}`eq_mat_timoshenko_beamXY` and the geometric relationships {eq}`eq_geom_epsxx_timoshenko_beamXY` and {eq}`eq_geom_epsxy_timoshenko_beamXY`, we obtain the relationships between the generalized internal dynams $M_z$ and $V_y$ and the generalized strain components $\kappa_x$ and $\gamma_{xy}$:

```{math}
:label: eq_timoshenko_beamXY_generalized
\begin{align*}
M_z(x) = E I_z \kappa_x(x), \\
V_y(x) = \overline{G} A \gamma_{xy}
\end{align*}
```

where $I_z = \int_A y^2 \\, dA$ is the second moment of inertia about $z$ and $\kappa_x(x)=\frac{d \vartheta_z(x)}{dx}$ is the curvature about $z$.

### Putting it all together

If we take the first derivatives of the equations of {eq}`eq_timoshenko_beamXY_generalized` wrt. $x$ and substitute into {eq}`eq_mz_beamXY_timoshenko` and {eq}`eq_fy_beamXY_timoshenko`, we arrive at two a $2^{nd}$ order DEs, that fully describe the behaviour of a Timoshenko-Ehrenfest beam under the effect of static loads, within the bounds of the assumptions we've made.

```{math}
:label: eq_timoshenko_beamXY_final
\begin{align*}
\sum F_y&: \quad \overline{G} A \left( \frac{d^2 v(x)}{dx^2} - \frac{d \vartheta_z(x)}{dx} \right) + p_{y}(x) &= 0, \\
\sum M_z&: \quad  E I_z \frac{d^2 \vartheta_z(x)}{dx^2} + \overline{G} A \left( \frac{d v(x)}{dx} - \vartheta_z(x) \right)  + p_{zz}(x) &= 0.
\end{align*}
```

These DEs alongside with the sufficient number of boundary conditions, completes the boundary-value problem (BVP) of a Timoshenko-Ehrenfest beam.

### BVP of simply-supported Timoshenko-Ehrenfest beams

Equation {eq}`eq_timoshenko_beamXY_final` governs the behaviour of all Euler-Bernoulli beams under the effects of static loads -within the bounds of the assumptions we've made-, regardless to the beam being a console, a continous multi-span beam, or something as simple as a simply-supported beam. The difference between these structures comes down to different boundary-conditions, which are constraints posed on the unknowns, at the boundaries. For a simply-supported beam, in the simplest case, these conditions are

```{math}
:label: eq_timoshenko_beamXY_BC
v(0)=0, \quad v(L)=0, \quad \kappa_x(0)=0, \quad \kappa_x(L)=0 
```

specifying zero vertical displacements and curvatures at the boundaries. Equations {eq}`eq_timoshenko_beamXY_final` and {eq}`eq_timoshenko_beamXY_BC` together consitute the BVP of a simply-supported Timoshenko-Ehrenfest beam.

## Navier's solution of simply-supported Timoshenko-Ehrenfest beams

The solution is identical in approach to the one for Euler-Bernoulli beams described {ref}`here <Navier_solution_simply_supported_bernoulli>`. Now we have two unknown displacement functions to determine, but the logic is the same. The series expansions  are

```{math}
:label: eq_timoshenko_beamXY_displacement_series_expansion
\begin{align*}
v(x) \approx \overline{v}(x) &= \sum_{k=1}^{n} v^{(k)} \, \sin \left( \frac{\pi k x}{L} \right), \\
\vartheta_z(x) \approx \overline{\vartheta}_z(x) &= \sum_{k=1}^{n} \vartheta_z^{(k)} \, \cos \left( \frac{\pi k x}{L} \right)
\end{align*}
```

for the displacements and

```{math}
:label: eq_timoshenko_beamXY_load_series_expansion
\begin{align*}
q(x) \approx \overline{q}(x) &= \sum_{k=1}^{n} q^{(k)} \, \sin \left( \frac{\pi k x}{L} \right), \\
p_{zz}(x) \approx \overline{p}_{zz}(x) &= \sum_{k=1}^{n} p_{zz}^{(k)} \, \cos \left( \frac{\pi k x}{L} \right)
\end{align*}
```

for the distributed forces. Upon substitution of {eq}`eq_timoshenko_beamXY_displacement_series_expansion` and {eq}`eq_timoshenko_beamXY_load_series_expansion` into {eq}`eq_timoshenko_beamXY_final` and utilizing a few trigonometric identities, the solutions for the different modes decouple. The solution for the unknowns of mode $k$ comes down to the solution of the following linear system:

```{math}
\begin{bmatrix}
\overline{G} A \frac{\pi^2 k^2}{L^2} & - \overline{G} A \frac{\pi k}{L} \\
\text{sym.} & \overline{G} A + EI_z \frac{\pi^2 k^2}{L^2}
\end{bmatrix}
\begin{bmatrix}
v^{(k)} \\
\vartheta_z^{(k)}
\end{bmatrix}
=
\begin{bmatrix}
q^{(k)} \\
p_{zz}^{(k)}
\end{bmatrix}.
```

If all $v^{(k)}$ and $\vartheta_z^{(k)}$ has been calculated, we can compute all quantities of interest:

```{math}
:label: eq_timoshenko_beamXY_summary
\begin{align*}
v(x) \approx \overline{v}(x) &= \sum_{k=1}^{n} v^{(k)} \, \sin \left( \frac{\pi k x}{L} \right) \\
\vartheta_z(x) \approx \overline{\vartheta}_z(x) &= \sum_{k=1}^{n} \vartheta_z^{(k)} \, \cos \left( \frac{\pi k x}{L} \right) \\
\kappa_z(x) \approx \overline{\kappa}_z(x) &= - \sum_{k=1}^{n} \vartheta_z^{(k)} \, \frac{\pi k}{L} \, \sin \left( \frac{\pi k x}{L} \right) \\
\gamma_{xy}(x) \approx \overline{\gamma}_{xy}(x) &= \sum_{k=1}^{n} \left( v^{(k)} \frac{\pi k}{L} - \vartheta_z^{(k)}\right) \, \cos \left( \frac{\pi k x}{L} \right) \\
M_z(x) \approx \overline{M}_z(x) &= - \sum_{k=1}^{n} EI_z \vartheta_z^{(k)} \, \frac{\pi k}{L} \, \sin \left( \frac{\pi k x}{L} \right) \\
V_y(x) \approx \overline{V}_y(x) &= \sum_{k=1}^{n} \overline{G} A \left( v^{(k)} \frac{\pi k}{L} - \vartheta_z^{(k)}\right) \, \cos \left( \frac{\pi k x}{L} \right)
\end{align*}
```

### Series representation of arbitrary loads

This is identical to the procedure described for Euler-Bernoulli beams {ref}`here <Navier_series_representation_of_beam_loads>`.
