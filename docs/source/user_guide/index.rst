.. _getting_started:

============
Introduction
============

How to use the documentation
============================

The documentation is devoted to three main parts:

* **Getting Started Guide** - This very guide you are reading. It helps you to get started with the 
  library by laying down the steps of a miniature course, through wich you hopefully get a hang of 
  things.

* **User Guide** - Introduces the concepts of the library and illustrates the usage of the classes 
  and algorithms by small examples. All of the chapters can be downloaded as Jupyter Notebooks from 
  the source repository of the library using the link at the top of the pages.

* **API Reference** - The documentation of the classes and algorithms of the library. The API 
  Reference is rich in examples and explanations, and this should be the primary source 
  of information as to how to use the library. If you are a seasoned developer, you might consider 
  jumping into the API Reference directly.

* **Gallery** - This is a collection of examples that involve some kind of visualization. The code 
  is ususally provided as one big block of execution, with ocassional internal notes. If all this 
  is new to you, we suggest to go through the User Guide or the API Reference first.

What is `sigmaepsilon.solid.fourier`?
=====================================

.. include:: ..\global_refs.rst

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

Highlights
----------

.. include:: ../highlights.rst

Prerequisities
==============

In order to use the library effectively, you should be familiar with

* Python - any level for usage, advanced for contribution
* NumPy, SciPy - intermediate level familiarity is required
* xarray - basic level of familiaarity required

It is not crucial, but it`s better to have a basic understanding of

* Numba - Numba is critical for the performance of the library as it does most of the heavy lifting, 
  alongside NumPy and SciPy of course. In most cases it uses JIT-compilation to speed stuff up and 
  it helps to understand how this all works. This is crucial if you consider to contribute to any of 
  the libraries in the sigmaepsilon namespace.

It is recommended to be familiar with

* Matplotlib - Most of the examples and the user guide is full of plots, most of them using matplotlib. 
  The plots are high quality, publication ready illustrations. It might be useful to learn how to use 
  it if you have no prior experience. Also, by using the poltting facility, you can give some feedback 
  if you feel something is missing, becoming a contributor yourself.

What about performance?
=======================

Yes, it is. The library is designed to be fast, as it relies on the vector math capabilities of `NumPy`_ and `SciPy`_, 
while other computationally sensitive calculations are JIT-compiled using `Numba`_. Thanks to `Numba`_, the 
implemented algorithms are able to bypass the limitations of Python's GIL and are parallelized on multiple cores, 
utilizing the full potential of what the hardware has to offer.

Installation
============

To install the library, follow the instructions in the :doc:`Installation Guide <installation>`.
