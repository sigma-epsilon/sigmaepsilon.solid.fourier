---
myst:
  html_meta:
    description: "sigmaepsilon.solid.fourier provides fast semi-analytical Fourier series solutions for simply supported Euler-Bernoulli and Timoshenko beams and Kirchhoff-Love and Uflyand-Mindlin plates in Python."
---

# SigmaEpsilon.Solid.Fourier – Fourier Solutions for Plate and Beam Problems in Python

```{toctree}
:maxdepth: 1
:hidden:

User Guide <user_guide/index>
Theory Guide <theory_guide/index>
Gallery <examples_gallery>
API Reference <api>
Development <development>
```

```{image} _static/logo.png
:align: center
:class: only-light
```

```{image} _static/logo_dark.png
:align: center
:class: only-dark
```

**Version**: {{ version }}

**Useful links**:
{doc}`Installation <user_guide/installation>` |
{ref}`Getting Started <getting_started>` |
[Issue Tracker](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/issues) |
[Source Repository](https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier) |
{doc}`Bibliography <bibliography>` |
{doc}`Using with LLMs <LLMs>`

The [sigmaepsilon.solid.fourier](https://sigmaepsilonsolidfourier.readthedocs.io/en/latest/) library provides semi-analytical solutions for selected beam and plate bending problems, where boundary conditions are inherently satisfied through the careful choice of approximating functions. While the available solutions cover only a limited set of boundary conditions, they are significantly faster than, for example, finite element methods when applicable. This makes the library especially valuable for:

* Experimentation
* Verification
* Concept validation
* Education
* Publication

All implementations leverage the speed and efficiency of libraries such as [NumPy](https://numpy.org/doc/stable/index.html) and [SciPy](https://scipy.org/), with performance-critical code sections accelerated using [Numba](https://numba.pydata.org/) where needed.

(highlights)=

## Highlights

```{include} _snippets/highlights.md
```

## Contents

::::::{grid} 2

:::::{grid-item-card}
:img-top: _static/index-images/getting_started.svg

Getting Started
^^^

Start here to set up your development environment and take your first steps with the library.

+++

```{button-ref} user_guide/introduction
:ref-type: doc
:expand:
:color: secondary
:click-parent:

Get me started
```
:::::

:::::{grid-item-card}
:img-top: _static/index-images/user_guide.svg

User Guide
^^^

Explore the user guide for a comprehensive walkthrough of the library's main features, with helpful background and explanations.

+++

```{button-ref} user_guide/index
:ref-type: doc
:expand:
:color: secondary
:click-parent:

To the user guide
```
:::::

:::::{grid-item-card}
:img-top: _static/index-images/api.svg

API Reference
^^^

The reference guide details all functions, modules, and objects in the library, explaining their usage and parameters. Some familiarity with the core concepts is assumed.

+++

```{button-ref} api
:ref-type: doc
:expand:
:color: secondary
:click-parent:

To the reference guide
```
:::::

:::::{grid-item-card}
:img-top: _static/index-images/contributor.svg

Contributor's Guide
^^^

Interested in contributing? The guidelines will walk you through the process of improving the library.

+++

```{button-ref} development
:ref-type: doc
:expand:
:color: secondary
:click-parent:

To the contributor's guide
```
:::::

:::::{grid-item-card}
:columns: 12
:img-top: _static/index-images/llm.svg

Using the Library with LLMs
^^^

Working with an AI assistant? Feed it the LLM-friendly version of this documentation: download `llms.txt` or copy its URL for quick access.

+++

```{button-ref} LLMs
:ref-type: doc
:expand:
:color: secondary
:click-parent:

To the LLM guide
```
:::::

::::::

## Indices and tables

* {doc}`Bibliography <bibliography>`
* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
