# Navier's Solution of Simply-Supported Euler-Bernoulli Beams

At the core of all solutions provided in this package lies Navier's classical approach for simply supported Euler-Bernoulli beams {cite:p}`navier1826resume`. This method forms the foundation for analyzing beam deflections and internal forces under various loading conditions. By leveraging the power of Fourier series expansions, Navier's solution enables the representation of arbitrary load distributions and corresponding beam responses in a systematic and mathematically rigorous way. This framework not only simplifies the solution process for simply supported beams but also serves as a stepping stone for more advanced structural analysis techniques.

## The behaviour of Euler-Bernoulli beams under the effect of static loads

The Euler-Bernoulli beam theory {cite:p}`enwiki:1329231351` builds on the following set of assumptions:

1. Small strains and small displacements.
2. Linear elastic material behaviour.
3. Plane sections remain plane and perpendicular to the neutral axis.

On top of these fundamental assumptions, we use the following simplifications:

- Constant cross section along the beam axis.
- No coupling between bending and torsion effects.

### Equilibrium equations

If we take an infinitesimal beam segment in the $x-y$ plane, and write the equations of static equilibrium, we arrive to the equation

```{math}
:label: eq_mz_beamXY
\sum M_z: \, \left. \frac{d M_z}{dx} \right|_x + V_y(x) + p_{zz}(x) = 0
```

for the moment equilibrium about $z$ at $x$ and

```{math}
:label: eq_fy_beamXY
\sum F_y: \, \left. \frac{d V_y}{dx} \right|_x + p_{y}(x) = 0
```

for the vertical equilibrium along the $y$ axis, where

- $M_z$ is the moment rotating counter-clockwise around $+x$,
- $V_y$ is shear force pointing along $+y$,
- $p_{zz}$ is the distributed moment load,
- $p_{y}$ is the distributed vertical load,

And the stress resultants $M_z$ and $V_y$ are defied by

```{math}
:label: eq_def_Mz_beamXY
M_z(x) = \int_A -y \, \sigma_x(x) \, dA
```

and

```{math}
:label: eq_def_Vy_beamXY
V_y(x) = \int_A \tau_{xy}(x) \, dA.
```

If we rearrange {eq}`eq_mz_beamXY` for $V_y$, take the derivative wrt. $x$ and substitute it into {eq}`eq_fy_beamXY`, we arrive at the equation

```{math}
:label: eq_total_beamXY
\sum F_y: \, \left. -\frac{d^2 M_z}{dx^2} \right|_x - \left. \frac{d p_{zz}}{dx} \right|_{x} + p_{y}(x) = 0.
```

### Geometric equations

We are talking about a beam theory, which means that we use a 1D domain to describe the motion of something that is 3D in essence. This means that we have to provide the mapping between the two systems. If we define a rotation $\vartheta_z$ and define it such that a positive value of it means a positive, counter-clockwise rotation around $+x$, we obtain

```{math}
:label: eq_geom_bernoulli_beamXY
u(x, y, z) = -y \, \vartheta_z(x).
```

According to our assumptions of having small strains, we have

```{math}
:label: eq_geom_epsxx_bernoulli_beamXY
\varepsilon_x (x,y,z) = \left. \frac{d u}{dx} \right|_{(x,y,z)} = -y \, \left. \frac{d \vartheta_z}{dx} \right|_{x}
```

and

```{math}
:label: eq_geom_epsxy_bernoulli_beamXY
\gamma_{xy}(x,y,z) = \left. \frac{d u}{dy} \right|_{(x,y,z)} + \left. \frac{d v}{dx} \right|_{(x,y,z)} = \left. \frac{d v}{dx} \right|_{x} - \vartheta_z(x)
```

for the nonzero engineering-strain components of the small-strain tensor, where we used equation {eq}`eq_geom_bernoulli_beamXY` in the right side of {eq}`eq_geom_epsxy_bernoulli_beamXY`. According to the Bernoulli hypothesis of planar cross-sections remaining perpendicular to the $x$ axis, we can equate {eq}`eq_geom_epsxy_bernoulli_beamXY` with zero, obtaining the relationship

```{math}
:label: eq_geom_epsxy_bernoulli_beamXY_v2
\left. \frac{d v}{dx} \right|_{x} = \vartheta_z(x),
```

that we will use a couple of steps later to calculate the rotations after having the unknown function $v(x)$ readily computed.

### Material equations

Assuming a linear elastic material without initial stresses (the undeformed configuration is stress free), the material equation is really nothing more than

```{math}
:label: eq_mat_bernoulli_beamXY
\varepsilon_x (x,y,z) = E \, \sigma_x (x,y,z)
```

where $\sigma_x$ denotes normal stresses along $x$, and $E$ is Young's modulus of elasticity. If we combine {eq}`eq_def_Mz_beamXY` with the material equation {eq}`eq_mat_bernoulli_beamXY` and the geometric relationship {eq}`eq_geom_epsxx_bernoulli_beamXY`, we obtain the relationship between the bending moment $M_z$ and the curvature $\kappa_x$:

```{math}
:label: eq_bernoulli_beamXY_total_1
\begin{align*}
M_z(x)
&= - \int_A y \, \sigma_x (x) \, dA \\
&= - \int_A y \, E \varepsilon_x (x) \, dA \\
&=  \int_A y^2 \, E \, \left. \frac{d \vartheta_z}{dx} \right|_{x} \, dA \\
&= E I_z \kappa_x(x),
\end{align*}
```

where $I_z = \int_A y^2 \\, dA$ is the second moment of inertia about $z$ and $\kappa_x(x)=\left. \frac{d \vartheta_z}{dx}\right|_{x}$ is the curvature about $z$.

### Putting it all together

If we take second derivative of {eq}`eq_bernoulli_beamXY_total_1` wrt. $x$ and substitute into {eq}`eq_total_beamXY`, we arrive at a $4^{th}$ order DE, that fully describes the behaviour of an Euler-Bernoulli beam under the effect of static loads, within the bounds of the assumptions we've made.

```{math}
:label: eq_bernoulli_beamXY_final
\sum F_y: \, E I_z \left. \frac{d^4 v}{dx^4} \right|_{x} + \left. \frac{d p_{zz}}{dx} \right|_{x} - p_{y}(x) = 0
```

This DE alongside with the sufficient number of boundary conditions, completes the boundary-value problem (BVP) of an Euler-Bernoulli beam.

### BVP of simply-supported Euler-Bernoulli beams

Equation {eq}`eq_bernoulli_beamXY_final` governs the behaviour of all Euler-Bernoulli beams under the effects of static loads -within the bounds of the assumptions we've made-, regardless to the beam being a console, a continous multi-span beam, or something as simple as a simply-supported beam. The difference between these structures comes down to different boundary-conditions, which are constraints posed on the unknowns, at the boundaries. Because the DE is a $4^{th}$ order one, we need 4 conditions. For a simply-supported beam, in the simplest case, these conditions are

```{math}
:label: eq_bernoulli_beamXY_BC
v(0)=0, \quad v(L)=0, \quad \kappa_x(0)=0, \quad \kappa_x(L)=0 
```

specifying zero vertical displacements and curvatures at the boundaries. Equations {eq}`eq_bernoulli_beamXY_final` and {eq}`eq_bernoulli_beamXY_BC` together consitute the BVP of a simply-supported Euler-Bernoulli beam.

(Navier_solution_simply_supported_bernoulli)=
## Navier's solution of simply-supported Euler-Bernoulli beams

The starting point of Navier's solution is to approximate the unknown function $v(x)$ as

```{math}
:label: eq_bernoulli_beamXY_v_sine_series_expansion
v(x) \approx \overline{v}(x) = \sum_{k=1}^{n} v_k \, \sin \left( \frac{\pi k x}{L} \right)
```

where $v_k$ are the coefficients to be determined. The idea behind this move is the recognition, that this form satisfies the boundary conditions of {eq}`eq_bernoulli_beamXY_BC` exactly, regardless of the actual values of the unknown coefficients $v_k$. If we disregard the moment loads $p_{zz}$ for a while apply the same trick for the vertical load component $q$, we get

```{math}
:label: eq_bernoulli_beamXY_q_sine_series_expansion
q(x) \approx \overline{q}(x) = \sum_{k=1}^{n} q_k \, \sin \left( \frac{\pi k x}{L} \right),
```

where $q_k$ are coefficients we can calculate for any kind of load. If we substitute both {eq}`eq_bernoulli_beamXY_q_sine_series_expansion` and {eq}`eq_bernoulli_beamXY_v_sine_series_expansion` into {eq}`eq_bernoulli_beamXY_final`, we arrive at

```{math}
\sum F_y: \, E I_z \sum_{k=1}^{n} v_k \left( \frac{\pi k}{L} \right)^4 \sin \left( \frac{\pi k x}{L} \right) = \sum_{k=1}^{n} q_k \, \sin \left( \frac{\pi k x}{L} \right),
```

and this is where the genious of Navier truly reveals itself. If we now multiply this by $\sin \left( \frac{\pi j x}{L} \right)$ for an arbitrary $j \in \\{ 1, \ldots, n \\}$ and make use of some trigonometric identities, the equations decouple:

```{math}
v_j E I_z \left( \frac{\pi j}{L} \right)^4 \sin \left( \frac{\pi j x}{L} \right) = q_j \, \sin \left( \frac{\pi j x}{L} \right),
```

and we can calculate the coefficient of the $j^{th}$ displacement mode as

```{math}
v_j = q_j \frac{L^4}{\pi^4 j^4 E I_z}.
```

Repeating this simple calculation for all $j \in \\{ 1, \ldots, n \\}$, yields the approximation of the solution in the form of {eq}`eq_bernoulli_beamXY_v_sine_series_expansion`.

To extend this for distributed moment loads, all we have to do is to take a look at equation {eq}`eq_bernoulli_beamXY_final`, which suggests that the moment load $p_{zz}$ needs to be expressed as

```{math}
:label: eq_bernoulli_beamXY_q_sine_series_expansion
p_{zz}(x) \approx \overline{p}_{zz}(x) = \sum_{k=1}^{n} r_k \, \cos \left( \frac{\pi k x}{L} \right),
```

and with a similar process as before, we can conclude that the unknown coefficients $v_j$ can be calculated as

```{math}
:label: eq_bernoulli_beamXY_v_coefficient_formula
v_j = \left( q_j + \frac{r_j \pi j}{L} \right) \frac{L^4}{\pi^4 j^4 E I_z}.
```

If all $v_j$ are calculated, we can compute all quantities of interest:

```{math}
:label: eq_bernoulli_beamXY_summary
\begin{align*}
v(x) \approx \overline{v}(x) &= \sum_{k=1}^{n} v_k \, \sin \left( \frac{\pi k x}{L} \right) \\
\vartheta_z(x) \approx \overline{\vartheta}_z(x) &= \sum_{k=1}^{n} v_k \, \frac{\pi k}{L} \, \cos \left( \frac{\pi k x}{L} \right) \\
\kappa_z(x) \approx \overline{\kappa}_z(x) &= \sum_{k=1}^{n} -v_k \, \left( \frac{\pi k}{L} \right)^2 \sin \left( \frac{\pi k x}{L} \right) \\
M_z(x) \approx \overline{M}_z(x) &= \sum_{k=1}^{n} -v_k \, EI_z \, \left( \frac{\pi k}{L} \right)^2 \sin \left( \frac{\pi k x}{L} \right) \\
V_y(x) \approx \overline{V}_y(x) &= \sum_{k=1}^{n} \left[ v_k \, EI_z \, \left( \frac{\pi k}{L} \right)^3 - r_k \right] \cos \left( \frac{\pi k x}{L} \right)
\end{align*}
```

(Navier_series_representation_of_beam_loads)=
### Series representation of arbitrary loads

There are a few loads for which the calculation of the coefficients $q_j$ and $r_j$ can be done with closed-form expressions, while for others we have to carry out numerical integration. For distributed vertical loads, the poblem we have is that we have a function $q(x)$ and we want to find the approximation of it as $\overline{q}(x)$, expressed by {eq}`eq_bernoulli_beamXY_q_sine_series_expansion`:

```{math}
q(x) \approx \overline{q}(x) = \sum_{k=1}^{n} q_k \, \sin \left( \frac{\pi k x}{L} \right).
```

The question is how to choose $q_k$ such that $\overline{q}(x)$ is the closest to $q(x)$. This only opens the question of what *distance* means between two functions. The explanation will be a bit handwavy for mathematicians, but I wanted to avoid using advanced linear algebra and keep it simple enough for people who don't have that kind of expertise. After all, this library is for engineers.

It's quite obvious that the equality

```{math}
\int_0^L q(x) dx = \int_0^L \overline{q}(x) dx
```

doesn't really mean that $\overline{q}(x)$ and $q(x)$ are the same function. It's only one measurement after all. However, if we can write enough independent measurements like this, we have enough reason to think that they are close enough. What this practically means is that we have performed a number of experiments on both $\overline{q}(x)$ and $q(x)$, and they responded with the same answer each time. The more the experiments, the greater the confidence that they are equal. Unfortunatelly, this is how far I can go at the moment without advanced math that would probably freak most people out.

We have $n$ number of coefficients to determine, so we need $n$ independent experiments. One way to do that is to write

```{math}
\int_0^L q(x) \phi_{i}(x) dx = \int_0^L \overline{q}(x) \phi_{i}(x) dx
```

for all $i \in \\{ 1, \ldots, n \\}$, where $\phi_{i}(x)= \sin \left( \frac{\pi i x}{L} \right)$. Utilizing some trigonometric identities again, for the $i^{th}$ equation this gives

```{math}
:label: eq_bernoulli_beamXY_q_coefficient_formula
q_i = \frac{2}{L} \int_0^L q(x) \sin \left( \frac{\pi i}{L} x \right) dx.
```

For distributed moment loads, the procedure is very similar, except that we work with cosine functions and we utilize another set of trigonometric identities. The result of that line of thought is the formula

```{math}
:label: eq_bernoulli_beamXY_pzz_coefficient_formula
r_i = \frac{2}{L} \int_0^L p_{zz}(x) \cos \left( \frac{\pi i}{L} x \right) dx.
```

In some cases, evaluation of the integral expressions {eq}`eq_bernoulli_beamXY_q_coefficient_formula` and {eq}`eq_bernoulli_beamXY_pzz_coefficient_formula` is simple enough to do it a-priori and store the result. In the general case, one can use a good old trapezoidal rule, the Monte-Carlo method or any other numerical integration technique.

### Examples

#### Concentrated vertical load

Perhaps the most basic example is a single concentrated load with a value of $F$, located at $x_F$. In this case our load function is

```{math}
q(x) =
\begin{cases}
F, & x = x_F, \\
0, & \text{otherwise}.
\end{cases}
```

It's easy to see that expression {eq}`eq_bernoulli_beamXY_q_coefficient_formula` for the load coefficients yields

```{math}
q_i = \frac{2}{L} F \sin \left( \frac{\pi i}{L} x_F \right).
```

#### Distributed vertical load of constant intensity

Take a uniformly distributed vertical load now, with an intensity of $\hat{q}$, applied over the interval $\[a, b\]$:

```{math}
q(x) =
\begin{cases}
\hat{q}, & b \ge x \ge a , \\
0, & \text{otherwise}.
\end{cases}
```

Expression {eq}`eq_bernoulli_beamXY_q_coefficient_formula` for the load coefficients yields

```{math}
q_i = \frac{\hat{q} L}{\pi i} \left[ \cos \left( \frac{i \pi a}{L} \right) - \cos \left( \frac{i \pi b}{L} \right) \right].
```
