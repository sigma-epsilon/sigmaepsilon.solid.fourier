
# Navier's Solution of Simply-Supported Kirchhoff-Love Plates

After thoroughly discussing Euler-Bernoulli and Timoshenko beam theories, we now move forward, assuming the reader is familiar with the derivation process. Here, we focus on the essential steps required for understanding Kirchhoff-Love plate theory.

## Behavior of Kirchhoff-Love Plates Under Static Loads

Kirchhoff-Love plate theory {cite:p}`enwiki:1297807693` is based on the following assumptions:

1. Strains and displacements are small.
2. The material exhibits linear elastic behavior.
3. Surface normals remain perpendicular to the mid-surface after deformation.
4. The plate thickness is constant.

### Equilibrium Equations

The equilibrium equations for an infinitesimally small section of the plate are:

```{math}
:label: eq_equilibrium_equations_kirchhoff
\begin{align*}
\sum F_z:& \quad p_{z}(x,y) + \left. \frac{\partial v_x}{\partial x}\right|_{(x,y)} + \left.\frac{\partial v_y}{\partial y}\right|_{(x,y)} &= 0, \\
\sum M_x:& \quad p_{xx}(x,y) - \left.\frac{\partial m_y}{\partial y}\right|_{(x,y)} - \left.\frac{\partial m_{xy}}{\partial x}\right|_{(x,y)} + v_y(x,y) &= 0, \\
\sum M_y:& \quad p_{yy}(x,y) + \left.\frac{\partial m_x}{\partial x}\right|_{(x,y)} + \left.\frac{\partial m_{xy}}{\partial y}\right|_{(x,y)} - v_x(x,y) &= 0,
\end{align*}
```

where the internal forces for a plate of constant thickness $t$ are defined as:

```{math}
:label: eq_internal_forces_kirchhoff
\begin{align*}
m_x(x,y) &= \int_{-t/2}^{t/2} z \, \sigma_x(x,y,z) \, dz, \\
m_y(x,y) &= \int_{-t/2}^{t/2} z \, \sigma_y(x,y,z) \, dz, \\
m_{xy}(x,y) &= \int_{-t/2}^{t/2} z \, \tau_{xy}(x,y,z) \, dz, \\
v_x(x,y) &= \int_{-t/2}^{t/2} \tau_{xz}(x,y,z) \, dz, \\
v_y(x,y) &= \int_{-t/2}^{t/2} \tau_{yz}(x,y,z) \, dz.
\end{align*}
```

These equations can be summarized using the following matrix notation:

```{math}
:label: eq_internal_forces_matrix_notation_kirchhoff
\underline{m} = 
\begin{bmatrix}
m_x \\
m_y \\
m_z
\end{bmatrix}
=
\int_{-t/2}^{t/2} z \, \underline{\sigma} \, dz
```

where $\underline{\sigma}$ is the matrix of stress components of interest, as

```{math}
:label: eq_stress_vector_kirchhoff
\underline{\sigma} = 
\begin{bmatrix}
\sigma_x & \sigma_y & \tau_{xy}
\end{bmatrix} ^ T.
```

Following similar steps as in the Euler-Bernoulli beam theory, the equations in {eq}`eq_equilibrium_equations_kirchhoff` can be combined into a single equation:

```{math}
:label: eq_equilibrium_equations_combined_kirchhoff
\sum F_z: \quad p_{z} + \frac{\partial^2 m_x}{\partial x^2} + \frac{\partial^2 m_y}{\partial y^2} + 2 \frac{\partial^2 m_{xy}}{\partial x \partial y} + \frac{\partial p_{yy}}{\partial x} - \frac{\partial p_{xx}}{\partial y}  = 0.
```

### Geometric Equations

The displacement field is described by:

```{math}
:label: eq_displacement_field_kirchhoff
\begin{align*}
u(x,y,z) &= z \, \vartheta_y(x,y), \\
v(x,y,z) &= -z \, \vartheta_x(x,y), \\
w(x,y,z) &= w(x,y),
\end{align*}
```

where $w(x,y)$, $\vartheta_y(x,y)$, and $\vartheta_x(x,y)$ are the reduced displacement functions. The relationship between the components of the small-strain tensor and the reduced displacement field is:

```{math}
:label: eq_strain_displacement_equations_kirchhoff
\begin{align*}
\varepsilon_x(x,y,z) &= \left. \frac{\partial u}{\partial x} \right|_{(x,y)} = z \, \left. \frac{\partial \vartheta_y}{\partial x} \right|_{(x,y)} = z \, \kappa_{x}(x,y), \\
\varepsilon_y(x,y,z) &= \left. \frac{\partial v}{\partial y} \right|_{(x,y)} = -z \, \left. \frac{\partial \vartheta_x}{\partial y} \right|_{(x,y)} = z \, \kappa_{y}(x,y), \\
\gamma_{xy}(x,y,z) &= \left. \frac{\partial u}{\partial y} \right|_{(x,y)} + \left. \frac{\partial v}{\partial x} \right|_{(x,y)} = z \, \left. \left( \frac{\partial \vartheta_y}{\partial y} - \frac{\partial \vartheta_x}{\partial x} \right) \right|_{(x,y)} = z \, \kappa_{xy}(x,y), \\
\gamma_{xz}(x,y,z) &= \left. \frac{\partial u}{\partial z} \right|_{(x,y)} + \left. \frac{\partial w}{\partial x} \right|_{(x,y)} = \vartheta_y(x,y) + \left. \frac{\partial w}{\partial x} \right|_{(x,y)}, \\
\gamma_{yz}(x,y,z) &= \left. \frac{\partial v}{\partial z} \right|_{(x,y)} + \left. \frac{\partial w}{\partial x} \right|_{(x,y)} = -\vartheta_x(x,y) + \left. \frac{\partial w}{\partial y} \right|_{(x,y)}.
\end{align*}
```

Due to the hypothesis of planar sections, the last two equations above must be zero, which gives:

```{math}
:label: eq_gammayz_zero_kirchhoff
\vartheta_x(x,y) = \left. \frac{\partial w}{\partial y} \right|_{(x,y)}
```

and

```{math}
:label: eq_gammaxz_zero_kirchhoff
\vartheta_y(x,y) = - \left. \frac{\partial w}{\partial x} \right|_{(x,y)},
```

These relationships allow us to recover the rotations once the displacement function $w$ is known. The first three equations of {eq}`eq_strain_displacement_equations_kirchhoff` can be written in matrix form as:

```{math}
:label: eq_gammaxz_zero_kirchhoff
\underline{\varepsilon} = z \, \underline{\kappa},
```

with

```{math}
:label: eq_strain_vector_kirchhoff
\underline{\varepsilon} = 
\begin{bmatrix}
\varepsilon_x & \varepsilon_y & \gamma_{xy}
\end{bmatrix} ^ T
```

and

```{math}
:label: eq_generalized_strain_vector_kirchhoff
\underline{\kappa} = 
\begin{bmatrix}
\kappa_x & \kappa_y & \kappa_{xy}
\end{bmatrix} ^ T.
```

### Material Equations

The material relationship at the material level is:

```{math}
:label: eq_material_equations_kirchhoff
\underline{\sigma} = \underline{\underline C} \, \underline{\varepsilon}
= \underline{\underline C} \, z \, \underline{\kappa},
```

where $\underline{\underline C}$ is the material stiffness matrix. Substituting this into {eq}`eq_internal_forces_matrix_notation_kirchhoff` gives:

```{math}
:label: eq_reduced_material_equations_kirchhoff
\underline{m} = \underline{\underline D} \, \underline{\kappa}
```

where $\underline{\underline D}$ is the reduced stiffness matrix of the plate section, defined as:

```{math}
:label: eq_reduced_stiffness_matrix_kirchhoff
\underline{\underline D} = 
\begin{bmatrix}
D_{11} & D_{12} & 0 \\
D_{21} & D_{22} & 0 \\
0 & 0 & D_{66}
\end{bmatrix}
=
\int_{-t/2}^{t/2} z^2 \, \underline{\underline C} \, dz.
```

For a linear elastic, isotropic material and a plate of constant thickness $t$, this becomes:

```{math}
\underline{\underline D} = \frac{E t^3}{12 (1 - \nu^2)}
\begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \frac{1 - \nu}{2}
\end{bmatrix},
```

where $E$ is Young's modulus and $\nu$ is Poisson's ratio.

### Summary

Combining the equilibrium equation {eq}`eq_equilibrium_equations_combined_kirchhoff` with the geometric and material relationships {eq}`eq_strain_displacement_equations_kirchhoff` and {eq}`eq_reduced_material_equations_kirchhoff`, we obtain the following partial differential equation (PDE):

```{math}
:label: eq_PDE_kirchhoff
\begin{align*}
\sum F_z: \quad &D_{11} \, \frac{\partial^4 w}{\partial x^4} + D_{22} \, \frac{\partial^4 w}{\partial y^4} + (2 D_{12} + 4 D_{66}) \frac{\partial^4 w}{\partial x^2 \partial y^2} \\
& = p_{z} + \frac{\partial p_{yy}}{\partial x} - \frac{\partial p_{xx}}{\partial y}.
\end{align*}
```

This PDE, together with the appropriate boundary conditions, defines the boundary-value problem (BVP) for a Kirchhoff-Love plate.

### BVP of Simply-Supported Kirchhoff-Love Plates

The boundary conditions for a simply-supported Kirchhoff-Love plate are:

```{math}
:label: eq_BC_kirchhoff
\begin{align*}
w(x, 0) &= w(x, L_y) = 0 \qquad &\forall x \in (0, L_x), \\
w(0, y) &= w(L_x, y) = 0 \qquad &\forall y \in (0, L_y), \\
m_x(0, y) &= m_x(L_x, y) = 0 \qquad &\forall y \in (0, L_y), \\
m_y(x, 0) &= m_y(x, L_y) = 0 \qquad &\forall x \in (0, L_x).
\end{align*}
```

Equations {eq}`eq_PDE_kirchhoff` and {eq}`eq_BC_kirchhoff` together constitute the BVP for a simply-supported Kirchhoff-Love plate.

## Solution of the BVP Using Navier's Approach

To simplify further expressions, we introduce the following notation:

```{math}
:label: eq_Si_Sj_notation_kirchhoff
\begin{align*}
&S_i(x) = \sin \left(\frac{\pi \,i \,x}{Lx}\right), \qquad
&S_j(y) = \sin \left(\frac{\pi \,j \,y}{Ly}\right), \\
&C_i(x) = \cos \left(\frac{\pi \,i \,x}{Lx}\right), \qquad
&C_j(y) = \cos \left(\frac{\pi \,j \,y}{Ly}\right).
\end{align*}
```

The displacements and loads are approximated as:

```{math}
:label: eq_approx_kirchhoff
\begin{align*}
w(x, y) \approx \overline{w}(x, y) &= \sum_{i=1}^m \sum_{j=1}^n w^{(ij)} \, S_i(x) \, S_j(y), \\
p_z(x, y) \approx \overline{p}_z(x, y) &= \sum_{i=1}^m \sum_{j=1}^n p_z^{(ij)} \, S_i(x) \, S_j(y), \\
p_{xx}(x, y) \approx \overline{p}_{xx}(x, y) &= \sum_{i=1}^m \sum_{j=1}^n p_{xx}^{(ij)} \, S_i(x) \, C_j(y), \\
p_{yy}(x, y) \approx \overline{p}_{yy}(x, y) &= \sum_{i=1}^m \sum_{j=1}^n p_{yy}^{(ij)} \, C_i(x) \, S_j(y).
\end{align*}
```

Using the strain-displacement equations {eq}`eq_strain_displacement_equations_kirchhoff` and the material relationships {eq}`eq_reduced_material_equations_kirchhoff` and {eq}`eq_reduced_stiffness_matrix_kirchhoff`, one can verify that the following approximation:

```{math}
\overline{w}(x, y) = \sum_{i=1}^m \sum_{j=1}^n w^{(ij)} \, S_i \, S_j
```

satisfies the boundary conditions of {eq}`eq_BC_kirchhoff`. Substituting the approximations from {eq}`eq_approx_kirchhoff` into {eq}`eq_PDE_kirchhoff`, using trigonometric identities and rearranging, we find that the solution for $w^{(ij)}$ is:

```{math}
:label: eq_solution_wij_kirchhoff
w^{(ij)} = \frac
{ p_z^{(ij)} - \frac{\pi \, i}{L_x} p_{yy}^{(ij)} - \frac{\pi \, j}{L_y} p_{xx}^{(ij)} }
{ \frac{\pi^4}{L_x^4 \, L_y^4} \left[ D_{11} L_y^4 \, i^4 + D_{22} \, L_x^4 \, j^4 + 2 \, L_x^2 \, L_y^2 \, i^2 \, j^2 \left( D_{12} + 2 \, D_{66} \right) \right] }
```

where the load coefficients are obtained by evaluating the following expressions:

```{math}
:label: eq_load_coeff_expressions_kirchhoff
\begin{align*}
p_z^{(ij)} &= \frac{4}{L_x \, L_y} \int_0^{L_y} \int_0^{L_x} p_z(x,y) \, S_i(x) \, S_j(y) \, dxdy, \\
p_{xx}^{(ij)} &= \frac{4}{L_x \, L_y} \int_0^{L_y} \int_0^{L_x} p_{xx}(x,y) \, S_i(x) \, C_j(y) \, dxdy, \\
p_{yy}^{(ij)} &= \frac{4}{L_x \, L_y} \int_0^{L_y} \int_0^{L_x} p_{yy}(x,y) \, C_i(x) \, S_j(y) \, dxdy.
\end{align*}
```
