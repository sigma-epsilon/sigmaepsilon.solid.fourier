---
myst:
  html_meta:
    description: "Gallery of visualization examples with sigmaepsilon.solid.fourier, plotting loads and analysis results of beams and plates with Matplotlib and PyVista."
---

(examples_gallery)=

# Gallery

The gallery contains a collection of examples that include some
kind of visualization. The examples here don't have much explanation,
for that you should refer to the {doc}`User Guide <user_guide/index>`.

## Beams

```{toctree}
:hidden:

gallery/bernoulli_beam_advanced_plot
gallery/plot_loads_1d
gallery/timoshenko_matplotlib
```

::::{grid} 1 2 3 3
:gutter: 3
:class-container: gallery

:::{grid-item-card} Advanced Plotting of Beam Results
:img-top: _static/gallery/bernoulli_beam_advanced_plot.png
:link: gallery/bernoulli_beam_advanced_plot
:link-type: doc
:::

:::{grid-item-card} Plotting Loads for Beams Using Matplotlib
:img-top: _static/gallery/plot_loads_1d.png
:link: gallery/plot_loads_1d
:link-type: doc
:::

:::{grid-item-card} Plotting Linear Static Analysis Results for Beams
:img-top: _static/gallery/timoshenko_matplotlib.png
:link: gallery/timoshenko_matplotlib
:link-type: doc
:::

::::

## Plates

```{toctree}
:hidden:

gallery/plate_utils
gallery/Uflyand_Mindlin_matplotlib
```

::::{grid} 1 2 3 3
:gutter: 3
:class-container: gallery

:::{grid-item-card} Plotting Plate Utilizations in 3D with PyVista
:img-top: _static/gallery/plate_utils.jpg
:link: gallery/plate_utils
:link-type: doc
:::

:::{grid-item-card} Contour Plots of Linear Static Analysis Results for Plates
:img-top: _static/gallery/Uflyand_Mindlin_matplotlib.png
:link: gallery/Uflyand_Mindlin_matplotlib
:link-type: doc
:::

::::
