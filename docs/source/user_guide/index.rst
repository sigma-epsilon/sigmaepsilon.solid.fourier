.. _getting_started:

============
Introduction
============

you need to use the library effectively. The documentation is divided into several parts, each of them
the information you are looking for. If you are new to the library, we suggest you start with the

Welcome to the library’s documentation! Here you’ll find everything you need to use the library effectively. The documentation is organized into several sections, each serving a specific purpose. It is designed for easy navigation, so you can quickly find the information you need. If you’re new to the library, we recommend starting with the :doc:`User Guide <user_guide>`.


Documentation Structure
=======================

The documentation is divided into five main sections:

* :ref:`Getting Started Guide <getting_started>` – This very guide. It walks you through the initial steps with the library, offering a brief course to help you get started.

* :ref:`User Guide <user_guide>` – Introduces the core concepts and demonstrates usage of the library’s classes and algorithms with concise examples. The user guide also references other documentation sections, including the API Reference. **If you’re new, this is the best place to begin.** All chapters are available as downloadable Jupyter Notebooks from the source repository (see the link at the top of each page).

* :ref:`Theory Guide <theory_guide>` – Explains the mathematical background and theoretical foundations behind the library. This section is ideal for those who want to understand the underlying principles, derivations, and assumptions, or who plan to extend the library.

* :ref:`API Reference <api_reference>` – Comprehensive documentation for all classes and algorithms. The API Reference is rich in examples and explanations, and is the primary resource for learning how to use the library. Experienced developers may want to jump straight to this section.

* :ref:`Gallery <examples_gallery>` – A collection of visualization examples. Code is typically provided as a single executable block, with occasional notes. These examples are light on explanation, so if you’re new, we suggest starting with the User Guide.

What is `sigmaepsilon.solid.fourier`?
=====================================

.. include:: ../global_refs.rst


The `sigmaepsilon.solid.fourier`_ library provides semi-analytical solutions for certain beam and plate bending problems, where boundary conditions are inherently satisfied by carefully chosen approximating functions. While only a limited set of boundary conditions is supported, the solutions are significantly faster than, for example, finite element methods when applicable. This makes the library especially useful for:

* Experimentation
* Verification
* Concept validation
* Education
* Publication

Highlights
----------

.. include:: ../highlights.rst


Prerequisites
=============

To use the library effectively, you should be familiar with:

* Python – any level for basic usage, advanced for contributing
* `NumPy`_, `SciPy`_ – intermediate familiarity recommended

It’s also helpful, though not essential, to have a basic understanding of:

* `Numba`_ – Numba is critical for performance, handling much of the heavy lifting alongside NumPy and SciPy. It uses JIT compilation to accelerate code. Understanding how Numba works is especially important if you plan to contribute to the sigmaepsilon libraries.

Recommended additional knowledge:

* `xarray`_ and `Pandas`_ – These are optional dependencies, but highly recommended for handling results. Otherwise, you’ll need to manage multidimensional arrays and keep track of axis meanings manually.
* `Matplotlib`_ – Most examples and the user guide include high-quality, publication-ready plots using Matplotlib. If you’re unfamiliar with it, learning the basics will be helpful. Using the plotting features also allows you to provide feedback or contribute improvements.


Performance
===========

The library is designed for speed, leveraging the vectorized math capabilities of `NumPy`_ and `SciPy`_. Performance-critical computations are JIT-compiled with `Numba`_, allowing the algorithms to bypass Python’s GIL and run in parallel across multiple cores, making full use of your hardware.

Installation
============


To install the library, follow the instructions in the :doc:`Installation Guide <installation>`.
