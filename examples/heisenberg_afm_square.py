import varipeps
import jax
import jax.numpy as jnp

# Config Setting
## Set maximal steps for the CTMRG routine
varipeps.config.ad_custom_max_steps = 100
## Set maximal steps for the fix point routine in the gradient calculation
varipeps.config.ctmrg_max_steps = 100
## Set convergence threshold for the CTMRG routine
varipeps.config.ctmrg_convergence_eps = 1e-7
## Set convergence threshold for the fix point routine in the gradient calculation
varipeps.config.ad_custom_convergence_eps = 5e-8
## Enable/Disable printing of the convergence of the single CTMRG/gradient fix point steps.
## Useful to enable this during debugging, should be disabled for batch runs
varipeps.config.ctmrg_print_steps = True
varipeps.config.ad_custom_print_steps = False
## Select the method used to calculate the descent direction during optimization
varipeps.config.optimizer_method = varipeps.config.Optimizing_Methods.CG
## Select the method used to calculate the (full) projectors in the CTMRG routine
varipeps.config.ctmrg_full_projector_method = (
    varipeps.config.Projector_Method.FISHMAN
)
## Set maximal steps for the optimization routine
varipeps.config.optimizer_max_steps = 2000
## Increase enviroment bond dimension if truncation error is below this value
varipeps.config.ctmrg_heuristic_increase_chi_threshold = 1e-4

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

# Unit cell structure
structure = [[0, 1], [1, 0]]

# Create random initialization for the iPEPS unit cell
unitcell = varipeps.peps.PEPS_Unit_Cell.random(
    structure,  # Unit cell structure
    p,  # Physical dimension
    chiB,  # iPEPS bond dimension
    startChi,  # Start value for enviroment bond dimension
    float,  # Data type for the tensors: float (real) or complex tensors
    max_chi=maxChi,  # Maximal enviroment bond dimension
)

# Run optimization
result = varipeps.optimization.optimize_peps_network(
    unitcell,
    exp_func,
    autosave_filename=f"data/autosave_square_chiB_{chiB:d}.hdf5",
)

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
