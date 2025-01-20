.. _examples:


Examples
========

We provide several examples in the `examples/ folder of the variPEPS Git repository <https://github.com/variPEPS/variPEPS_Python/tree/main/examples>`_ that demonstrate how to use the code for variational energy optimization in typical 2D many-body problems.

.. In this section we want to elaborately walk through the example for the Heisenberg AFM on the 2d square lattice to explain a typical usage of the library.

Heisenberg antiferromagnet on the square lattice
------------------------------------------------

.. figure:: /images/square_lattice.*
   :align: center
   :width: 60%
   :alt: Two dimensional square lattice with red links indicating horizontal and
         blue links indicating vertical interactions.

   Two dimensional square lattice

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
arrays to define for example the Hamiltonian, we load this numpy
interface as well.

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
options commonly used in an optimization run. For a detailed
description of the configurable options we refer to the API description of the
config class: :obj:`varipeps.config.VariPEPS_Config`.

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
   maxChi = 36
   # Start value for enviroment bond dimension
   startChi = chiB**2 if chiB**2 < maxChi else maxChi

In this block we define imporant parameters for the model we want to simulate, such as as the interaction strength, the physical dimension of our tensor network and the iPEPS bond dimension. In the last two lines the initial and the maximal enviroment bond dimension is defined. A feature of the variPEPS library is that it not only supports simulation at a fixed enviroment bond dimension, but also a heurisitic increase/decrease of the dimension up to a maximal value. The dynamic change is controlled  by the truncation error in the CTMRG projector calculation (increase if the truncation errror becomes too large, decrease if it becomes insignificant). For example, in the config block above the parameter ``ctmrg_heuristic_increase_chi_threshold`` is set to the threshold at which to increase the refinement parameter. The maximal bond dimension ``maxChi`` ensures that the parameter does now grow unbounded, to the point where the memory and computational resources are exhausted.

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
   exp_func = varipeps.expectation.Two_Sites_Expectation_Value(
       horizontal_gates=(hamiltonianGates,),
       vertical_gates=(hamiltonianGates,),
   )

Here the Hamiltonian is constructed for our model. The Heisenberg AFM on the
square lattice can be described by the sum of the spin-spin interactions on
the horizontal and vertical bonds. Since we assume a constant
interaction strength for all bonds in our example, the expectation value can be calculated by
the same two-site interaction gate applied in all nearest neighbor
directions. The expectation function ``exp_func`` is later used in the
optimization to calculate the energy expectation value, which in turn is used as cost function to obtain
the ground state.

Initial unit cell construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Unit cell structure
   structure = [[0, 1], [1, 0]]

Here we define the unit cell structure which is used to simulate our model. In
this example we assume a
:math:`\scriptsize{\begin{matrix}A&B\\B&A\end{matrix}}`-structure, i.e. a two-site antiferromagnetic state.

.. code-block:: python

   # Create random initialization for the iPEPS unit cell
   unitcell = varipeps.peps.PEPS_Unit_Cell.random(
       structure,  # Unit cell structure
       p,  # Physical dimension
       chiB,  # iPEPS bond dimension
       startChi,  # Start value for enviroment bond dimension
       float,  # Data type for the tensors: `float` (real) or `complex` tensors
       max_chi=maxChi,  # Maximal enviroment bond dimension
   )

Using the unit cell structure and the model parameter defined above, we can
generate an initial unit cell. Here we initialize the iPEPS tensors with random
numbers. Other ways to initialize the tensors are provided, for example loading results 
from a simple update calculation.

Run the optimization
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Run optimization
   result = varipeps.optimization.optimize_peps_network(
       unitcell,
       exp_func,
       autosave_filename=f"data/autosave_square_chiB_{chiB:d}.hdf5",
   )

This function call executes the main function of the library, the variational energy 
optimization to obtain a good ground state candidate. The function offers several options 
to customize the optimization for different iPEPS ansätze, such as the spiral iPEPS 
approach. In our example, we only need to provide the initial unit cell, the function 
for calculating the energy expectation value, and a file path for autosaving the 
optimization process, enabling the restoration of interrupted simulations.

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
       f"data/heisenberg_square_J_{J:d}_chiB_{chiB:d}_chiMax_{chiM:d}.hdf5",
       auxiliary_data=auxiliary_data,
   )

Finally, we want to save the unit cell with the optimized tensors to a file for
further analysis. The library allows to store the data directly into a
HDF5 file along with user-supplied auxiliary data. Here, for example, we not only
want to store the plain tensors but also the calculated energy, meta information
from the optimization run (e.g. energy per step or the runtime per step) and the
calculated magnetic expectation values. At a later examination of the results,
these data can be easily loaded along with the tensors of the tensor network.
