(sign_conventions)=
# Notions, Coordinate-Systems and Sign Conventions

## Beams

Beams are bent in the $x-y$ plane and the neutral axis in the unloaded configuration lies on the $x$ axis. **The origin of the coordinate system is at the left support.** When you specify the location of a concentrated force, you specify the distance of the point of application from the left support. When you are asking for the displacement at a point, you specify the distance of that point from the left support.

(beam_sign_conventions)=
### Sign conventions for loads of beams

```{figure} ../_static/sign_convention_beam_XY_load_light.png
:align: center
:class: only-light

Sign convention for beam loads in the x-y plane. The loads on the figure would have positive values.
```

```{figure} ../_static/sign_convention_beam_XY_load_dark.png
:align: center
:class: only-dark

Sign convention for beam loads in the x-y plane. The loads on the figure would have positive values.
```

### Sign conventions for internal forces and displacements of beams

The positive meaning of the bending moment and the shear force can be read from the definitions

```{math}
M_z(x) = \int_A -y \, \sigma_x(x) \, dA \qquad \text{and} \qquad V_y(x) = \int_A \tau_{xy}(x) \, dA.
```

```{figure} ../_static/sign_convention_beam_XY_light.png
:align: center
:class: only-light

Sign conventions for internal forces, couples, displacements and rotations for beams in the x-y plane. Quantities with a positive sign should be understood as they appear on the figure.
```

```{figure} ../_static/sign_convention_beam_XY_dark.png
:align: center
:class: only-dark

Sign conventions for internal forces, couples, displacements and rotations for beams in the x-y plane. Quantities with a positive sign should be understood as they appear on the figure.
```

## Plates

Plates are flat with a constant thickness and they lie in the x-y plane, with the reference surface ($z=0$) being the midsurface of the plate. **The origin of the orthonormal, right-handed coordinate system is in the bottom-left corner.** Whenever you specify the locations of loads or points for evaluating calculated quantities, you need to provide the coordinates wrt. this coordinate system.

(plate_sign_conventions)=
### Sign conventions for loads and displacements of plates

```{figure} ../_static/sign_convention_plate_1_light.png
:align: center
:class: only-light

Sign conventions for loads and displacement of plates.
```

```{figure} ../_static/sign_convention_plate_1_dark.png
:align: center
:class: only-dark

Sign conventions for loads and displacement of plates.
```

### Sign conventions for internal forces of plates

The internal forces for a plate of constant thickness $t$ are defined as:

```{math}
\begin{align*}
m_x(x,y) &= \int_{-t/2}^{t/2} z \, \sigma_x(x,y,z) \, dz, \\
m_y(x,y) &= \int_{-t/2}^{t/2} z \, \sigma_y(x,y,z) \, dz, \\
m_{xy}(x,y) &= \int_{-t/2}^{t/2} z \, \tau_{xy}(x,y,z) \, dz, \\
v_x(x,y) &= \int_{-t/2}^{t/2} \tau_{xz}(x,y,z) \, dz, \\
v_y(x,y) &= \int_{-t/2}^{t/2} \tau_{yz}(x,y,z) \, dz.
\end{align*}
```

```{figure} ../_static/sign_plate_bending_xy.png
:align: center
:height: 200px
:class: only-light

Bending in the XY plane.
```

```{figure} ../_static/sign_plate_bending_xy_dark.png
:align: center
:height: 200px
:class: only-dark

Bending in the XY plane.
```

```{figure} ../_static/sign_plate_bending_xz.png
:align: center
:height: 200px
:class: only-light

Bending in the XZ plane.
```

```{figure} ../_static/sign_plate_bending_xz_dark.png
:align: center
:height: 200px
:class: only-dark

Bending in the XZ plane.
```

```{figure} ../_static/sign_plate_twisting.png
:align: center
:height: 140px
:class: only-light

Twisting.
```

```{figure} ../_static/sign_plate_twisting_dark.png
:align: center
:height: 140px
:class: only-dark

Twisting.
```
