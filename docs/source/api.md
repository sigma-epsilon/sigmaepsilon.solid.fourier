---
myst:
  html_meta:
    description: "API reference of sigmaepsilon.solid.fourier: the beam and plate models, the load classes, the result containers, the enumerations, the protocols and the configuration."
---

(api_reference)=

# API reference

The reference guide contains a detailed description of the functions,
modules, and objects included in the library. The reference describes how the
methods work and which parameters can be used. It assumes that you have an
understanding of the key concepts.

## Models

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   ~sigmaepsilon.solid.fourier.problem.NavierProblem
   ~sigmaepsilon.solid.fourier.beam.NavierBeam
   ~sigmaepsilon.solid.fourier.plate.NavierPlate
```

## Loads

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-data-template.rst

   ~sigmaepsilon.solid.fourier.loads.Float1d
   ~sigmaepsilon.solid.fourier.loads.Float2d
   ~sigmaepsilon.solid.fourier.loads.LoadDomainType
   ~sigmaepsilon.solid.fourier.loads.LoadValueType
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   ~sigmaepsilon.solid.fourier.loads.LoadGroup
   ~sigmaepsilon.solid.fourier.loads.LoadCase
   ~sigmaepsilon.solid.fourier.loads.PointLoad
   ~sigmaepsilon.solid.fourier.loads.LineLoad
   ~sigmaepsilon.solid.fourier.loads.RectangleLoad
   ~sigmaepsilon.solid.fourier.loads.DiskLoad
```

## Results

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   ~sigmaepsilon.solid.fourier.result.LoadCaseResultLinStat
   ~sigmaepsilon.solid.fourier.result.BeamLoadCaseResultLinStat
   ~sigmaepsilon.solid.fourier.result.PlateLoadCaseResultLinStat
```

## Enumerations

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   ~sigmaepsilon.solid.fourier.enums.MechanicalModelType
   ~sigmaepsilon.solid.fourier.enums.IntegrationMethod
   ~sigmaepsilon.solid.fourier.enums.PostprocMethod
```

## Protocols

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst

   ~sigmaepsilon.solid.fourier.protocols
```

## Configuration

```{eval-rst}
.. currentmodule:: sigmaepsilon.solid.fourier.config

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   FourierConfig
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-data-template.rst

   Config
```
