.. _api_reference:

=============
API reference
=============

The reference guide contains a detailed description of the functions,
modules, and objects included in the library. The reference describes how the
methods work and which parameters can be used. It assumes that you have an
understanding of the key concepts.

Models
======

.. autoclass:: sigmaepsilon.solid.fourier.problem.NavierProblem
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.beam.NavierBeam
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.plate.NavierPlate
   :members:

Loads
=====

.. autoclass:: sigmaepsilon.solid.fourier.loads.Float1d
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.loads.Float2d
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.loads.LoadDomainType
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.loads.LoadValueType
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.loads.LoadGroup
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.loads.LoadCase
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.loads.PointLoad
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.loads.LineLoad
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.loads.RectangleLoad
   :members:

Results
=======

.. autoclass:: sigmaepsilon.solid.fourier.result.LoadCaseResultLinStat
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.result.BeamLoadCaseResultLinStat
   :members:

.. autoclass:: sigmaepsilon.solid.fourier.result.PlateLoadCaseResultLinStat
   :members:
