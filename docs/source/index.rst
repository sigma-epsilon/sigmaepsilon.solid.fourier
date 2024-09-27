============================================================================================
**SigmaEpsilon.Solid.Fourier** - Fourier Solutions of Some Plate and Beam Problems in Python
============================================================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <user_guide>
   Gallery <examples_gallery>
   API Reference <api>
   Development <development>

.. image:: _static/logo.png
   :align: center

**Version**: |version|

**Useful links**:
:doc:`Installation <user_guide/installation>` |
:ref:`Getting Started <getting_started>` |
`Issue Tracker <https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier/issues>`_ | 
`Source Repository <https://github.com/sigma-epsilon/sigmaepsilon.solid.fourier>`_

.. include:: global_refs.rst

The `sigmaepsilon.solid.fourier`_ library offers semi-analytic solutions to some beam and plate 
bending problems, where the boundary conditions are a-priori satisfied by careful selection of the 
approximating functions. Although the calculations only cover a handful of boundary conditions, 
when they are applicable, they are significantly faster than let say a finite element solution. 
For this reason, it is very useful for a couple of things:

* experimentation
* verification
* concept validation
* education
* publication

The implementations in the library all rely on fast and efficient algorithms provided by the goodies of
`NumPy`_, `SciPy`_ and the likes. Where necessary, computationally intensive parts of the code are written
using `Numba`_.

.. _highlights:

Highlights
==========

.. include:: highlights.rst

Contents
========

.. grid:: 2
    
    .. grid-item-card::
        :img-top: ../source/_static/index-images/getting_started.svg

        Getting Started
        ^^^^^^^^^^^^^^^

        The getting started guide is your entry point. It helps you to set up
        a development environment and make the first steps with the library.

        +++

        .. button-ref:: user_guide/index
            :expand:
            :color: secondary
            :click-parent:

            Get me started

    .. grid-item-card::
        :img-top: ../source/_static/index-images/user_guide.svg

        User Guide
        ^^^^^^^^^^

        The user guide provides a detailed walkthrough of the library, touching 
        the key features with useful background information and explanation.

        +++

        .. button-ref:: user_guide
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in the library. It describes how the
        methods work and which parameters can be used. It assumes that you have an
        understanding of the key concepts.

        +++

        .. button-ref:: api
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/contributor.svg

        Contributor's Guide
        ^^^^^^^^^^^^^^^^^^^

        Want to add to the codebase? The contributing guidelines will guide you through
        the process of improving the library.

        +++

        .. button-ref:: development_guide
            :expand:
            :color: secondary
            :click-parent:

            To the contributor's guide
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



