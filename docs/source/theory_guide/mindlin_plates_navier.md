# Navier's Solution of Simply-Supported Mindlin-Reissner Plates

## Behavior of Mindlin-Reissner Plates Under Static Loads

Mindlin-Reissner plate theory (aka. first-order shear deformation theory or FSDT) {cite:p}`enwiki:1322532912, 10.1115/1.4010217, Reissner1945TheEO` is based on the following core assumptions:

1. Strains and displacements are small compared to the thickness of the plate.
2. The material exhibits linear elastic behavior.
3. Normals to the mid-surface remain straight but not perpendicular after deformation.
   - This is the defining kinematic assumption.
   - Normals remaining straight has the consequence of no transverse normal warping.
   - Releasing the perpendicularity assumption of Kirchhoff-Love plates enables shear-deformations to be accounted for.
4. Transverse shear strains are constant through the thickness.
   - This assumption is not physically exact (true shear stresses are parabolic) and necessitates a shear correction factor to recover correct shear energy.

On top of these fundamental assumptions, we make the following simplifications:

- The plate thickness is constant.
- The midsurface is flat.

### Equilibrium Equations

The equilibrium equations for an infinitesimally small section of the plate are:

```{math}
:label: eq_equilibrium_equations_mindlin
\begin{align*}
\sum F_z:& \quad p_{z}(x,y) + \left. \frac{\partial v_x}{\partial x}\right|_{(x,y)} + \left.\frac{\partial v_y}{\partial y}\right|_{(x,y)} &= 0, \\
\sum M_x:& \quad p_{xx}(x,y) - \left.\frac{\partial m_y}{\partial y}\right|_{(x,y)} - \left.\frac{\partial m_{xy}}{\partial x}\right|_{(x,y)} + v_y(x,y) &= 0, \\
\sum M_y:& \quad p_{yy}(x,y) + \left.\frac{\partial m_x}{\partial x}\right|_{(x,y)} + \left.\frac{\partial m_{xy}}{\partial y}\right|_{(x,y)} - v_x(x,y) &= 0,
\end{align*}
```

where the internal forces for a plate of constant thickness $t$ are defined as:

```{math}
:label: eq_internal_forces_mindlin
\begin{align*}
m_x(x,y) &= \int_{-t/2}^{t/2} z \, \sigma_x(x,y,z) dz, \\
m_y(x,y) &= \int_{-t/2}^{t/2} z \, \sigma_y(x,y,z) dz, \\
m_{xy}(x,y) &= \int_{-t/2}^{t/2} z \, \tau_{xy}(x,y,z) dz, \\
v_x(x,y) &= \int_{-t/2}^{t/2} \tau_{xz}(x,y,z) dz, \\
v_y(x,y) &= \int_{-t/2}^{t/2} \tau_{yz}(x,y,z) dz.
\end{align*}
```

These equations can be summarized using the following matrix notation:

```{math}
:label: eq_internal_forces_matrix_notation_mindlin
\underline{m} = 
\begin{bmatrix}
m_x \\
m_y \\
m_z
\end{bmatrix}
=
\int_{-t/2}^{t/2} z \, \underline{\sigma} \, dz
\qquad \text{and} \qquad
\underline{v} = 
\begin{bmatrix}
v_x \\
v_y
\end{bmatrix}
=
\int_{-t/2}^{t/2} \underline{\tau} \, dz
```

with

```{math}
:label: eq_stress_vector_mindlin
\underline{\sigma} = 
\begin{bmatrix}
\sigma_x & \sigma_y & \tau_{xy}
\end{bmatrix} ^ T
\qquad \text{and} \qquad
\underline{\tau} = 
\begin{bmatrix}
\tau_{xz} & \tau_{yz}
\end{bmatrix} ^ T.
```

```{note}
In the Kirchhoff-Love plate theory, equations {eq}`eq_equilibrium_equations_mindlin` are typically combined into a single equation. Here, however, we retain them as separate equations.
```

### Geometric Equations

The displacement field is described by:

```{math}
:label: eq_displacement_field_mindlin
\begin{align*}
u(x,y,z) &= z \, \vartheta_y(x,y), \\
v(x,y,z) &= -z \, \vartheta_x(x,y), \\
w(x,y,z) &= w(x,y),
\end{align*}
```

where $w(x,y)$, $\vartheta_y(x,y)$, and $\vartheta_x(x,y)$ are the reduced displacement functions. The relationship between the components of the small-strain tensor and the reduced displacement field is:

```{math}
:label: eq_strain_displacement_equations_mindlin
\begin{align*}
\varepsilon_x(x,y,z) &= \left. \frac{\partial u}{\partial x} \right|_{(x,y)} = z \, \left. \frac{\partial \vartheta_y}{\partial x} \right|_{(x,y)} = z \, \kappa_{x}(x,y), \\
\varepsilon_y(x,y,z) &= \left. \frac{\partial v}{\partial y} \right|_{(x,y)} = -z \, \left. \frac{\partial \vartheta_x}{\partial y} \right|_{(x,y)} = z \, \kappa_{y}(x,y), \\
\gamma_{xy}(x,y,z) &= \left. \frac{\partial u}{\partial y} \right|_{(x,y)} + \left. \frac{\partial v}{\partial x} \right|_{(x,y)} = z \, \left. \left( \frac{\partial \vartheta_y}{\partial y} - \frac{\partial \vartheta_x}{\partial x} \right) \right|_{(x,y)} = z \, \kappa_{xy}(x,y), \\
\gamma_{xz}(x,y,z) &= \left. \frac{\partial u}{\partial z} \right|_{(x,y)} + \left. \frac{\partial w}{\partial x} \right|_{(x,y)} = \vartheta_y(x,y) + \left. \frac{\partial w}{\partial x} \right|_{(x,y)}, \\
\gamma_{yz}(x,y,z) &= \left. \frac{\partial v}{\partial z} \right|_{(x,y)} + \left. \frac{\partial w}{\partial x} \right|_{(x,y)} = -\vartheta_x(x,y) + \left. \frac{\partial w}{\partial y} \right|_{(x,y)}.
\end{align*}
```

```{note}
In the Kirchhoff-Love theory, the assumption that normals to the mid-surface remain perpendicular to the mid-surface leads to setting the transverse shear-strain components to zero. Since the Mindlin-Reissner theory relaxes this assumption, we do not make this simplification here.
```

The generalized strain components with matrix notation:

```{math}
:label: eq_strain_displacement_mindlin_1
\underline{\varepsilon}
=
\begin{bmatrix}
\varepsilon_x & \varepsilon_y & \gamma_{xy}
\end{bmatrix} ^ T
= 
z \, \underline{\kappa}
= z \, \begin{bmatrix}
\kappa_x & \kappa_y & \kappa_{xy}
\end{bmatrix} ^ T.
```

and

```{math}
:label: eq_strain_displacement_mindlin_2
\underline{\gamma} = 
\begin{bmatrix}
\gamma_{xy} & \gamma_{xz}
\end{bmatrix} ^ T.
```

### Material Equations

The constitutive relationship at the material level is:

```{math}
:label: eq_material_equations_mindlin
\underline{\sigma} = \underline{\underline Q} \, \underline{\varepsilon}
= \underline{\underline Q} \, z \, \underline{\kappa}
\qquad \text{and} \qquad
\underline{\tau} = \underline{\underline P} \, \underline{\gamma},
```

Substituting this into {eq}`eq_internal_forces_matrix_notation_mindlin` gives:

```{math}
:label: eq_reduced_material_equations_mindlin
\underline{m} = \underline{\underline D} \, \underline{\kappa}
\qquad \text{and} \qquad
\underline{v} = \underline{\underline S} \, \underline{\gamma}
```

with

```{math}
:label: eq_reduced_stiffness_matrix_D_mindlin
\underline{\underline D} = 
\begin{bmatrix}
D_{11} & D_{12} & 0 \\
D_{21} & D_{22} & 0 \\
0 & 0 & D_{66}
\end{bmatrix}
=
\int_{-t/2}^{t/2} z^2 \, \underline{\underline Q} \, dz
```

and

```{math}
:label: eq_reduced_stiffness_matrix_S_mindlin
\underline{\underline S} = 
\begin{bmatrix}
S_{55} & 0 \\
0 & S_{44}
\end{bmatrix}
=
\int_{-t/2}^{t/2} \underline{\underline P} \, dz.
```

For a linear elastic, isotropic material and a plate of constant thickness $t$, these become:

```{math}
\underline{\underline D} = \frac{E t^3}{12 (1 - \nu^2)}
\begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \frac{1 - \nu}{2}
\end{bmatrix},
```

and

```{math}
\underline{\underline S} =
\begin{bmatrix}
k_{x} G t & 0 \\
0 & k_{y} G t
\end{bmatrix},
```

where $E$ is Young's modulus, $\nu$ is Poisson's ratio, $G$ is the shear modulus, $k_x$ and $k_y$ are the shear correction factors for the $x$ and $y$ directions.

### Summary

Combining the equilibrium equations {eq}`eq_equilibrium_equations_mindlin` with the geometric and material relationships {eq}`eq_strain_displacement_equations_mindlin` and {eq}`eq_reduced_material_equations_mindlin`, we obtain the following partial differential equations:

```{math}
:nowrap:
:label: eq_PDE_mindlin
\begin{align*}
\sum F_z:& \qquad 
&S_{44}\, \left( \frac{\partial \vartheta_x}{\partial y}
- \frac{\partial^2 w}{\partial y^2} \right)
- S_{55}\, \left( \frac{\partial \vartheta_y}{\partial x}
+ \frac{\partial^2 w}{\partial x^2} \right) &= p_z, \\
\sum M_x:& \qquad 
&D_{12}\,\frac{\partial^2 \vartheta_y}{\partial x\,\partial y}
- D_{22}\,\frac{\partial^2 \vartheta_x}{\partial y^2}
+ D_{66}\, \left( \frac{\partial^2 \vartheta_y}{\partial x\,\partial y} - \frac{\partial^2 \vartheta_x}{\partial x^2} \right) & \\
&& + S_{44}\,\left( \vartheta_x - \frac{\partial w}{\partial y} \right) &= p_{xx}, \\
\sum M_y:& \qquad
&- D_{11}\,\frac{\partial^2 \vartheta_y}{\partial x^2}
+ D_{12}\,\frac{\partial^2 \vartheta_x}{\partial x\,\partial y}
+ D_{66}\,\left( \frac{\partial^2 \vartheta_x}{\partial x\,\partial y} - \frac{\partial^2 \vartheta_y}{\partial y^2} \right) & \\
&& + S_{55}\, \left( \vartheta_y + \frac{\partial w}{\partial x} \right) &= p_{yy}.
\end{align*}
```

These PDEs, together with the appropriate boundary conditions, defines the boundary-value problem (BVP) for a Mindlin-Reissner plate.

### BVP of Simply-Supported Mindlin-Reissner Plates

The boundary conditions for a simply-supported Mindlin-Reissner plate are:

```{math}
:label: eq_BC_mindlin
\begin{align*}
w(x, 0) &= w(x, L_y) = 0 \qquad &\forall x \in (0, L_x), \\
w(0, y) &= w(L_x, y) = 0 \qquad &\forall y \in (0, L_y), \\
m_x(0, y) &= m_x(L_x, y) = 0 \qquad &\forall y \in (0, L_y), \\
m_y(x, 0) &= m_y(x, L_y) = 0 \qquad &\forall x \in (0, L_x).
\end{align*}
```

Equations {eq}`eq_PDE_mindlin` and {eq}`eq_BC_mindlin` together constitute the BVP for a simply-supported Mindlin-Reissner plate.

## Solution of the BVP Using Navier's Approach

To simplify further expressions, we introduce the following notation:

```{math}
:label: eq_Si_Sj_notation_mindlin
\begin{align*}
&S_i(x) = \sin \left(\frac{\pi \,i \,x}{Lx}\right), \qquad
&S_j(y) = \sin \left(\frac{\pi \,j \,y}{Ly}\right), \\
&C_i(x) = \cos \left(\frac{\pi \,i \,x}{Lx}\right), \qquad
&C_j(y) = \cos \left(\frac{\pi \,j \,y}{Ly}\right).
\end{align*}
```

The displacements and loads are approximated as:

```{math}
:label: eq_approx_mindlin
\begin{align*}
w(x, y) \approx \overline{w}(x, y) &= \sum_{i=1}^m \sum_{j=1}^n w^{(ij)} \, S_i(x) \, S_j(y), \\
\vartheta_x(x, y) \approx \overline{\vartheta}_x(x, y) &= \sum_{i=1}^m \sum_{j=1}^n \vartheta_x^{(ij)} \, S_i(x) \, C_j(y), \\
\vartheta_y(x, y) \approx \overline{\vartheta}_y(x, y) &= \sum_{i=1}^m \sum_{j=1}^n \vartheta_y^{(ij)} \, C_i(x) \, S_j(y), \\
p_z(x, y) \approx \overline{p}_z(x, y) &= \sum_{i=1}^m \sum_{j=1}^n p_z^{(ij)} \, S_i(x) \, S_j(y), \\
p_{xx}(x, y) \approx \overline{p}_{xx}(x, y) &= \sum_{i=1}^m \sum_{j=1}^n p_{xx}^{(ij)} \, S_i(x) \, C_j(y), \\
p_{yy}(x, y) \approx \overline{p}_{yy}(x, y) &= \sum_{i=1}^m \sum_{j=1}^n p_{yy}^{(ij)} \, C_i(x) \, S_j(y).
\end{align*}
```

Substituting the approximations from {eq}`eq_approx_mindlin` into {eq}`eq_PDE_mindlin`, using trigonometric identities and rearranging, we find that the solution for the unknown coefficients $w^{(ij)}$, $\vartheta_x^{(ij)}$ and $\vartheta_y^{(ij)}$ can be obtained from

```{math}
:label: eq_solution_ij_mindlin
\begin{bmatrix}
w^{(ij)} & \vartheta_x^{(ij)} & \vartheta_y^{(ij)}
\end{bmatrix} ^ T
=
{\underline{\underline{A}}^{(ij)}}^{-1}
\begin{bmatrix}
p_z^{(ij)} & p_{xx}^{(ij)} & p_{yy}^{(ij)}
\end{bmatrix} ^ T
```

with coefficient matrix

```{math}
:label: eq_solution_ij_coeff_matrix_mindlin
\underline{\underline{A}}^{(ij)} = 
\left[\begin{matrix}\frac{\pi^{2} S_{44} j^{2}}{L_{y}^{2}} + \frac{\pi^{2} S_{55} i^{2}}{L_{x}^{2}} & - \frac{\pi S_{44} j}{L_{y}} & \frac{\pi S_{55} i}{L_{x}}\\- \frac{\pi S_{44} j}{L_{y}} & \frac{\pi^{2} D_{22} j^{2}}{L_{y}^{2}} + \frac{\pi^{2} D_{66} i^{2}}{L_{x}^{2}} + S_{44} & - \frac{\pi^{2} D_{12} \,i \, j}{L_{x} L_{y}} - \frac{\pi^{2} D_{66} \,i \, j}{L_{x} L_{y}}\\\frac{\pi S_{55} i}{L_{x}} & - \frac{\pi^{2} D_{12} \, i \, j}{L_{x} L_{y}} - \frac{\pi^{2} D_{66} \, i \, j}{L_{x} L_{y}} & \frac{\pi^{2} D_{11} i^{2}}{L_{x}^{2}} + \frac{\pi^{2} D_{66} j^{2}}{L_{y}^{2}} + S_{55}\end{matrix}\right]
```

and load coefficients

```{math}
:label: eq_load_coeff_expressions_mindlin
\begin{align*}
p_z^{(ij)} &= \frac{4}{L_x \, L_y} \int_0^{L_y} \int_0^{L_x} p_z(x,y) \, S_i(x) \, S_j(y) \, dxdy, \\
p_{xx}^{(ij)} &= \frac{4}{L_x \, L_y} \int_0^{L_y} \int_0^{L_x} p_{xx}(x,y) \, S_i(x) \, C_j(y) \, dxdy, \\
p_{yy}^{(ij)} &= \frac{4}{L_x \, L_y} \int_0^{L_y} \int_0^{L_x} p_{yy}(x,y) \, C_i(x) \, S_j(y) \, dxdy.
\end{align*}
```
