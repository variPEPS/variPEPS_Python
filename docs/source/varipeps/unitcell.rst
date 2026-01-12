.. _varipeps_unitcell:

iPEPS Unit Cells
================

The text in this section is mainly copied (and only slightly modifed) from the
publication `SciPost Phys. Lect. Notes 86 (2024)
<https://doi.org/10.21468/SciPostPhysLectNotes.86>`_ by Jan Naumann, Erik
Lennart Weerda, Matteo Rizzi, Jens Eisert and Philipp Schmoll. This section is
licensed under the :ref:`license_cc_by` as the original work.

General iPEPS unit cell structure
---------------------------------

We aim to simulate quantum many-body systems directly in the thermodynamic
limit. To this end, we consider a unit cell of lattice sites that is repeated
periodically over the infinite two-dimensional lattice. Reflecting this, the
general configurtions of the iPEPS Ansatz are defined with an arbitrary unit
cell of size :math:`(L_x, L_y)` on the square lattice. The lattice setup,
denoted by :math:`\mathcal{L}`, can be specified by a single matrix, which
uniquely determines the different lattice sites as well as their
arrangement. Let us consider a concrete example of an :math:`(L_x, L_y) = (2,
2)` state with only two and all four individual tensors, denoted by

.. math::

   \mathcal L_1 = \begin{pmatrix} A & B \\ B & A \end{pmatrix}, \hspace{0.5cm}
   \mathcal L_2 = \begin{pmatrix} A & C \\ B & D \end{pmatrix}.

.. subfigure:: AB
   :align: center
   :width: 75%
   :layout-sm: A|B

   .. image:: ../images/varipeps/ctmrgExample_1.*

   .. image:: ../images/varipeps/ctmrgExample_2.*              

   iPEPS ansätze with a unit cell of size :math:`(L_x, L_y) = (2, 2)` and only
   two (left) and four (right) different tensors.

The corresponding iPEPS ansätze are visualized in figure above. Here, the
rows/columns of :math:`\mathcal{L}` correspond to the :math:`x`/:math:`y`
lattice directions. The unit cell :math:`\mathcal{L}` is repeated periodically
to generate the full two-dimensional system. The bulk bond dimension of the
iPEPS tensors, denoted by :math:`\chi_B`, controls the accuracy of the
ansatz. An iPEPS state with :math:`N` different tensors in the unit cell
consists of :math:`N p \chi_B^4` variational parameters, which we aim to
optimize such that the iPEPS wave function represents an approximation of the
ground state of a specific Hamiltonian. The parameter :math:`p` denotes the
dimension of the physical Hilbert space, e.g., :math:`p = 2` for a system of
spin-:math:`1/2` particles.

The right choice of the unit cell is crucial in order to capture the structure
of the targeted state.  A mismatch of the ansatz could not only lead to a bad
estimate of the ground state, but also to no convergence in the CTMRG routine at
all.  Different lattice configurations have to be evaluated for specific
problems to find the correct pattern.


Spiral PEPS ansatz
------------------

To circumvent the problem of a fixed and a priori chosen unit cell structure,
recently an alternative description to the periodic structure has been proposed
(`Phys. Rev. Lett. 133, 176502 (2024)
<https://doi.org/10.1103/PhysRevLett.133.176502>`_). This approach is applicable
if the Hamiltonian has a certain global symmetry, where the additional degree of
freedom can be employed to reduce the description of the state to a subspace,
e.g. :math:`SU(2)` for spin-:math:`1/2` systems. Here the state is described by
the smallest possible unit cell, i.e. a single site for a square lattice, as
well as a product of local unitary operators parameterized by a wave vector
:math:`\mathbf{k} = (k_x, k_y)`. A fixed choice of the wave vector then
corresponds to the specification of a unit cell structure in the common iPEPS
setup. This approach allows for a variational optimization of the wave vector
along with the translationally invariant iPEPS tensor, removing the need to
choose a fixed unit cell structure altogether.
