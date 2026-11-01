---
myst:
  html_meta:
    description: "User guide of sigmaepsilon.solid.fourier: the workflow of setting up domains, specifying loads and running a linear static analysis."
---

(user_guide)=

# User Guide

The user guide provides a detailed walkthrough of the library, touching
the key features with useful background information and explanation.
It teaches you how to use the tools provided by the library through simple
examples. If you want to know about the details of specific classes and functions,
please refer to the {doc}`API Reference <../api>`.

## The workflow

The typical workflow in any simulation in the field of computational solid mechanics
consists of the following steps, and the user guide follows the same structure:

1. {doc}`Setting up the domains <domains>` and defining material properties.

   The tools in the library are limited to simply supported beams and plates.

2. {doc}`Specifying boundary conditions <boundary_conditions>` (loads and supports).

   You only have to set up loads here, since the essential boundary conditions (displacement supports) are automatically satisfied by the approximation model.

3. Performing some calculations, like a {doc}`linear static analysis <linear_static_analysis>`.

   Currently, the library supports linear elastic calculations.

4. Visualization of the results.

   Check out the {ref}`examples gallery <examples_gallery>`.

Every page of the user guide is a Jupyter Notebook that you can download from the
source repository and run on its own.

```{toctree}
:maxdepth: 1
:caption: Getting Started

introduction
installation
```

```{toctree}
:maxdepth: 2
:caption: Contents

domains
boundary_conditions
linear_static_analysis
```

```{toctree}
:maxdepth: 1
:caption: Examples
:numbered:

/examples/plate_utils_pyvista
/examples/mindlin_vs_kirchhoff
/examples/beam_influence_diagrams
```
