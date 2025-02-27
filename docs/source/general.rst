.. _general:


Introduction
============

variPEPS is a Python tensor network library developed for variational ground
state simulations in two spatial dimensions applying gradient optimization using
automatic differentation.

For a detailed report on the method, please see `our open-access publication on
SciPost (doi:10.21468/SciPostPhysLectNotes.86)
<https://doi.org/10.21468/SciPostPhysLectNotes.86>`_.

Installation using pip
======================

The current version of the variPEPS Python package is available on `PyPI
<https://pypi.org/project/variPEPS/>`_. It can be easily installed using the
Python package manager pip:

.. code-block:: console

   $ python3 -m pip install variPEPS

Usage
=====

The :obj:`varipeps` module is organized in several submodules corresponding to
the different features. For a variational optimization the most important parts
are (a full overview can be found in the :ref:`_api`):

* :obj:`varipeps.peps`: To define iPEPS unit cell and the tensors on each site,
  the library provides in this submodule the abstractions to define such a unit
  cell.
* :obj:`varipeps.expectation`: In this submodule the helper functions to define
  and calculate common expecation functions on the iPEPS unit
  cell. Particularly, the function can be used to define the Hamiltonian terms
  of the model of interest.
* :obj:`varipeps.mapping`: If not only interactions on the square lattice are of
  interest but also models on other 2d lattices, in this submodule one can find
  mappings of other lattices. Also the files there can be a good starting point
  to implement even more lattices.
* :obj:`varipeps.optimization`: The submodule providing the optimization
  algorithm and interface of the library. In almost all cases, one will interact
  with this part by the main function
  :obj:`varipeps.optimization.optimize_peps_network`.

All these different modules can be seen in action in the :ref:`_examples`
section of the documentation where exemplary code is discussed in detail.

Citation
========

We are happy if you want to use the framework for your research. To cite our
work we provide a list of our preferred references on the `GitHub page of the
project
<https://github.com/variPEPS/variPEPS_Python?tab=readme-ov-file#citation>`_. Please
check there for a current list.
