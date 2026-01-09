.. _examples_heisenberg_afm_triangular:

Heisenberg antiferromagnet on the triangular lattice
----------------------------------------------------

.. figure:: /images/triangular_lattice.*
   :align: center
   :width: 60%
   :alt: Two dimensional triangular lattice with links indicating nearest neighbor
         interactions.

   Two dimensional triangular lattice

The Hamiltonian for the Heisenberg antiferromagnet with constant exchange
interaction strength :math:`J>0` is defined as:

.. math::

   H = J \sum_{\langle i j \rangle} \vec{S}_i \vec{S}_j ,

where :math:`\langle i j \rangle` denotes the sum over all nearest neighbors in
the lattice.

Our aim is now to find the ground state of the model using the variational iPEPS
code of the variPEPS library.

Loading of relevant Python modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import varipeps
   import jax
   import jax.numpy as jnp

First of all we have to load the relevant Python modules for our simulation. The
:obj:`varipeps` module includes the full library to perform the variational
optimization. Internally it is based on the :obj:`jax` framework and its
:obj:`numpy`-like interface to execute the calculations. Since we will need
arrays to define for example the Hamiltonian, we load this numpy interface as
well.

variPEPS config settings
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Config Setting
   
   ## Set maximal steps for the CTMRG routine
   varipeps.config.ctmrg_max_steps = 100
   ## Set convergence threshold for the CTMRG routine
   varipeps.config.ctmrg_convergence_eps = 1e-7
   ## Select the method used to calculate the (full) projectors in the CTMRG routine
   varipeps.config.ctmrg_full_projector_method = (
       varipeps.config.Projector_Method.FISHMAN
   )
   ## Enable dynamic increase of CTMRG environment bond dimension
   varipeps.config.ctmrg_heuristic_increase_chi = True
   ## Increase CTMRG enviroment bond dimension if truncation error exceeds this value
   varipeps.config.ctmrg_heuristic_increase_chi_threshold = 1e-4
   
   ## Set maximal steps for the fix point routine in the gradient calculation
   varipeps.config.ad_custom_max_steps = 100
   ## Set convergence threshold for the fix point routine in the gradient calculation
   varipeps.config.ad_custom_convergence_eps = 5e-8

   ## Enable/Disable printing of the convergence of the single CTMRG/gradient fix point steps.
   ## Useful to enable this during debugging, should be disabled for batch runs
   varipeps.config.ctmrg_print_steps = True
   varipeps.config.ad_custom_print_steps = False

   ## Select the method used to calculate the descent direction during optimization
   varipeps.config.optimizer_method = varipeps.config.Optimizing_Methods.CG
   ## Set maximal number of steps for the optimization routine
   varipeps.config.optimizer_max_steps = 2000

The :obj:`varipeps` library allows to configure a large number of numerical
parameters to fine-tune the simulation. In this example we include several
options commonly used in an optimization run. For a detailed description of the
configurable options we refer to the API description of the config class:
:obj:`varipeps.config.VariPEPS_Config`.

Model parameters
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Set constants for the simulation
   modelName = "HeisenbergModel"
   # Interaction strength
   J = 1
   # iPEPS bond dimension
   chiB = 2
   # Physical dimension
   p = 2
   # Maximal enviroment bond dimension
   maxChi = 64
   # Start value for enviroment bond dimension
   startChi = maxChi

In this block we define imporant parameters for the model we want to simulate,
such as as the interaction strength, the physical dimension of our tensor
network and the iPEPS bond dimension. In the last two lines the initial and the
maximal enviroment bond dimension is defined. A feature of the variPEPS library
is that it not only supports simulation at a fixed enviroment bond dimension,
but also a heurisitic increase/decrease of the dimension up to a maximal
value. The dynamic change is controlled by the truncation error in the CTMRG
projector calculation (increase if the truncation error becomes too large,
decrease if it becomes insignificant). For example, in the config block above
the parameter ``ctmrg_heuristic_increase_chi_threshold`` is set to the threshold
at which to increase the refinement parameter. The maximal bond dimension
``maxChi`` ensures that the parameter does now grow unbounded, to the point
where the memory and computational resources are exhausted.

For the triangular lattice Heisenberg AFM it is known that a quite large
environment bond dimension is needed such that we directly start the simulation
with the maximal allowed dimension to avoid unnecessary calculations.

Constructing the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # define spin-1/2 matrices
   Id = jnp.eye(2)
   Sx = jnp.array([[0, 1], [1, 0]]) / 2
   Sy = jnp.array([[0, -1j], [1j, 0]]) / 2
   Sz = jnp.array([[1, 0], [0, -1]]) / 2

   # construct Hamiltonian terms
   hamiltonianGates = J * (jnp.kron(Sx, Sx) + jnp.kron(Sy, Sy) + jnp.kron(Sz, Sz))

   # create function to compute expectation values for the square Heisenberg AFM
   exp_func = (
       varipeps.expectation.triangular_two_sites.Triangular_Two_Sites_Expectation_Value(
           horizontal_gates=(hamiltonianGates,),
           vertical_gates=(hamiltonianGates,),
           diagonal_gates=(hamiltonianGates,),
           real_d=p,
           is_spiral_peps=True,
           spiral_unitary_operator=Sy,
       )
   )

Here the Hamiltonian is constructed for our model. The Heisenberg AFM on the
triangular lattice can be described by the sum of the spin-spin interactions on
the horizontal, vertical and diagonal bonds. Since we assume a constant
interaction strength for all bonds in our example, the expectation value can be
calculated by the same two-site interaction gate applied in all nearest neighbor
directions. The expectation function ``exp_func`` is later used in the
optimization to calculate the energy expectation value, which in turn is used as
cost function to obtain the ground state.

We use in this example the description of the model by the spiral-PEPS ansatz
(`Phys. Rev. Lett. 133, 176502 (2024)
<https://doi.org/10.1103/PhysRevLett.133.176502>`_). Here the model is described
by a single real iPEPS tensor and a relative rotation along the :math:`S_y` axis
for interactions with its neighbors. The rotation is set by a spiral vector
which is supplied later in this example. This reduces the computational effort
required for the optimization as only one tensor and not multiple ones have to be
optimized.

As discussed in the following section, we use the triangular-CTMRG method for
this example, therefore we use the provided expectation class for this case
(:obj:`~varipeps.expectation.triangular_two_sites.Triangular_Two_Sites_Expectation_Value`).

Initial unit cell construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Unit cell structure
   structure = [[0]]

Here we define the unit cell structure which is used to simulate our model. As
noted in the section above, due to the spiral ansatz we only need a single iPEPS
site.

.. code-block:: python

   # Create random initialization for the iPEPS unit cell
   unitcell = varipeps.peps.PEPS_Unit_Cell.random(
       structure,  # Unit cell structure
       p,  # Physical dimension
       chiB,  # iPEPS bond dimension
       startChi,  # Start value for enviroment bond dimension
       float,  # Data type for the tensors: `float` (real) or `complex` tensors
       max_chi=maxChi,  # Maximal enviroment bond dimension
       peps_type=varipeps.peps.PEPS_Type.TRIANGULAR,  # Select triangular PEPS
   )

Using the unit cell structure and the model parameter defined above, we can
generate an initial unit cell. Here we initialize the iPEPS tensors with random
numbers. Other ways to initialize the tensors are provided, for example loading results 
from a simple update calculation.

As we simulate a triangular lattice, we use the triangular-CTMRG method
described in `Phys. Rev. B 113, 045117 (2026)
<https://doi.org/10.1103/g5gm-tzf8>`_. This is selected at the time of creation
of the unit cell by the ``peps_type`` parameter.

Run the optimization
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Run optimization
   result = varipeps.optimization.optimize_unitcell_fixed_spiral_vector(
       unitcell,
       jnp.array((2 / 3, 2 / 3), dtype=jnp.float64),  # Spiral vector
       exp_func,
       autosave_filename=f"data/autosave_triangular_chiB_{chiB:d}_chiMax_{maxChi:d}.hdf5",
   )

This function call executes the main function of the library, the variational
energy optimization to obtain a good ground state candidate. We use one of the
wrapper around the main optimization function which is predefined for the case
of a spiral PEPS ansatz with a fixed value for the spiral vector. There are
other variants for example for the variational optimization of the full spiral
vector or for the optimization of just the :math:`x`- or :math:`y`-component.
The other arguments are the function for calculating the energy expectation
value, and a file path for autosaving the optimization process, enabling the
restoration of interrupted simulations.

Evaluate the results
^^^^^^^^^^^^^^^^^^^^

In this section we show some exemplary evaluation of the result of the optimization.

.. code-block:: python

   # Calculate magnetic expectation values
   Mag_Gates = [Sx, Sy, Sz]


   def calc_magnetic(unitcell):
       mag_result = []
       for ti, t in enumerate(unitcell.get_unique_tensors()):
           r = varipeps.expectation.one_site.calc_one_site_multi_gates(
               t.tensor, t, Mag_Gates
           )
           mag_result += r
       return mag_result


   magnetic_exp_values = calc_magnetic(result.unitcell)

We assume for our example that we are interested in the single-site spin
expectation values. These could be used to analyse the :math:`z`-magnetization
or the staggered magnetization of our model at/near the ground state.

.. code-block:: python

   # Define some auxiliary data which should be stored along the final iPEPS unit cell
   auxiliary_data = {
       "best_energy": result.fun,
       "best_run": result.best_run,
       "magnetic_exp_values": magnetic_exp_values,
   }
   for k in sorted(result.max_trunc_error_list.keys()):
       auxiliary_data[f"max_trunc_error_list_{k:d}"] = result.max_trunc_error_list[k]
       auxiliary_data[f"step_energies_{k:d}"] = result.step_energies[k]
       auxiliary_data[f"step_chi_{k:d}"] = result.step_chi[k]
       auxiliary_data[f"step_conv_{k:d}"] = result.step_conv[k]
       auxiliary_data[f"step_runtime_{k:d}"] = result.step_runtime[k]

   # save full iPEPS state
   result.unitcell.save_to_file(
       f"data/heisenberg_triangular_J_{J:d}_chiB_{chiB:d}_chiMax_{maxChi:d}.hdf5",
       auxiliary_data=auxiliary_data,
   )

Finally, we want to save the unit cell with the optimized tensors to a file for
further analysis. The library allows to store the data directly into a
HDF5 file along with user-supplied auxiliary data. Here, for example, we not only
want to store the plain tensors but also the calculated energy, meta information
from the optimization run (e.g. energy per step or the runtime per step) and the
calculated magnetic expectation values. At a later examination of the results,
these data can be easily loaded along with the tensors of the tensor network.
