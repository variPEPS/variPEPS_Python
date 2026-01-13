.. _varipeps:

Variational iPEPS
=================

The text in this section is mainly copied (and only slightly modifed) from the
publication `SciPost Phys. Lect. Notes 86 (2024)
<https://doi.org/10.21468/SciPostPhysLectNotes.86>`_ by Jan Naumann, Erik
Lennart Weerda, Matteo Rizzi, Jens Eisert and Philipp Schmoll. This section is
licensed under the :ref:`license_cc_by` as the original work.

We seek to find the TN representation of the state vector
:math:`\ket{\psi}_\mathrm{TN}` that best approximates the true ground state
vector :math:`\ket{\psi_0}` of an Hamilton operator of the form

.. math::

   H = \sum_{j \in \Lambda} T_j (h) \, ,

where :math:`T_j` is the translation operator on the lattice :math:`\Lambda`,
and :math:`h` is a generic :math:`k`-local Hamiltonian, i.e., it includes an
arbitrary number of operators acting on lattice sites at most at a (lattice)
distance :math:`k` from a reference lattice point. Such a situation is very
common in condensed matter physics, to say the least. To this aim, we employ the
variational principle

.. math::

   \frac{\langle \psi \vert H \vert \psi \rangle}{\langle \psi \vert \psi
    \rangle} \ge E_0 \hspace{0.5cm} \forall \, \ket{\psi},

and use an energy gradient with respect to the tensor coefficients to search for
the minimum -- the precise optimization strategy being discussed later.  Such an
energy gradient is accessed by means of tools from *automatic differentiation*
(AD), a set of techniques to evaluate the derivative of a function specified by
a computer program. In the :obj:`varipeps` library we use the
:obj:`jax`-framework which already implements AS for common mathematical
functions.

Since we directly target systems in the thermodynamic limit, a *corner transfer
matrix renormalization group* (CTMRG) procedure constitutes the backbone of the
algorithm, and also will come in handy for AD purposes.  This is used to compute
the approximate contraction of the infinite lattice, which is crucial in order
to compute accurate expectation values in the first place.
