=========================
How to document your code
=========================

Documenting your code is an essential part of the development process. It helps
you and other developers understand the codebase and its functionality. In this
section, we will discuss how to document your code using `Sphinx`.

Documenting your code consists of the following steps:

1. **Writing docstrings**: Docstrings are the first step in documenting your
   code. They are used to describe the purpose and functionality of a function,
   class, or module.

2. **Writing source files for Sphinx**: Sphinx is a documentation generator
   that supports various file formats. You need to write
   source files in either of these formats to generate documentation using Sphinx.

3. **Building the documentation**: Once you have written the source files, you
   can build the documentation using Sphinx. Sphinx will generate HTML, PDF, or
   other formats based on the source files.

In the remaining sections, we will discuss each of these steps in detail.

Writing docstrings
==================

Docstrings are used to describe the purpose and functionality of a function,
class, or module. They are written as a string literal that is the first
statement in a function, class, or module. Docstrings can be accessed using the
`__doc__` attribute of the object.

There are several formats for writing docstrings, such as `Google`, `Numpy`,
and `reStructuredText`. In the projects developed by SigmaEpsilon, we use the
`NumPy` format for writing docstrings.

A few resources to help you write good docstrings:

PEP 257: https://www.python.org/dev/peps/pep-0257/

NumPy Docstring Guide: https://numpydoc.readthedocs.io/en/latest/format.html

Writing source files for Sphinx
===============================

In projects belonging to SigmaEpsilon, we use a mix of `reStructuredText` and
`Jupyter Notebooks` for writing source files for Sphinx. `reStructuredText` is a
lightweight markup language that is used to write documentation for Python code.
It is easy to read and write, and Sphinx supports it out of the box. `Jupyter Notebooks`
are used to write tutorials and examples for the library.

You can edit `reStructuredText` files using any text editor, such as `VS Code`,
`Sublime Text`, or `Atom`. For editing `Jupyter Notebooks`, you can use editors like `VS Code`
as well, but we suggest `Jupyter Lab` as you will probably need to include raw 
`reStructuredText` in the notebooks, which requires a bit of tweaking `VS Code` doesn't
support at the moment of writing this document.

Building the documentation
==========================

Once you have written the source files, you can build the documentation using
Sphinx. For this, navigate to the `docs` directory and run the following command:

.. code-block:: shell

   poetry run make html

This command will generate the HTML documentation in the `docs/build` directory. To open
the documentation in a browser, navigate to the `docs/build/html` directory and open
`index.html` in a browser.