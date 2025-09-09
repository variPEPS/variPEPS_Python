import jax.numpy as jnp

from varipeps import varipeps_config
from varipeps.ctmrg import calc_ctmrg_env, CTMRGNotConvergedError
from varipeps.peps import PEPS_Unit_Cell

from . import overlap_single_site
from . import overlap_four_sites

overlap_mapping = {
    (1, 1): overlap_single_site.Overlap_Single_Site,
    (2, 2): overlap_four_sites.Overlap_Four_Sites_Square,
}


def calculate_overlap(
    unitcell_A, unitcell_B, chi, max_chi, *, test_automatic_overlap_conv=True
):
    structure_A = tuple(tuple(i) for i in unitcell_A.data.structure)
    structure_B = tuple(tuple(i) for i in unitcell_B.data.structure)

    if structure_A != structure_B:
        raise ValueError("Structure of both unit cells have to be the same.")

    size = unitcell_A.get_size()
    num_tensors = size[0] * size[1]
    overlap_func = overlap_mapping[size].calc_overlap

    unitcell_A = unitcell_A.convert_to_full_transfer()
    unitcell_B = unitcell_B.convert_to_full_transfer()

    norm_A = overlap_func(unitcell_A)
    norm_B = overlap_func(unitcell_B)

    overlap_tensors = [
        type(t).from_tensor(
            t.tensor / norm_A ** (1 / (2 * num_tensors)), t.d, t.D, chi, max_chi=max_chi
        )
        for t in unitcell_A.get_unique_tensors()
    ]

    for i, e in enumerate(overlap_tensors):
        e.tensor_conj = unitcell_B.get_unique_tensors()[i].tensor.conj() / norm_B ** (
            1 / (2 * num_tensors)
        )

    overlap_unitcell = PEPS_Unit_Cell.from_tensor_list(overlap_tensors, structure_A)

    if test_automatic_overlap_conv:
        tmp_max_steps = varipeps_config.ctmrg_max_steps
        tmp_fail = varipeps_config.ctmrg_fail_if_not_converged

        varipeps_config.ctmrg_fail_if_not_converged = False
        varipeps_config.ctmrg_max_steps = 10

        overlap_unitcell, _ = calc_ctmrg_env(
            [i.tensor for i in overlap_tensors], overlap_unitcell
        )
        overlap_AB = overlap_func(overlap_unitcell)

        varipeps_config.ctmrg_max_steps = 1
        for count in range(tmp_max_steps):
            old_overlap_AB = overlap_AB

            overlap_unitcell, _ = calc_ctmrg_env(
                [i.tensor for i in overlap_tensors], overlap_unitcell
            )
            overlap_AB = overlap_func(overlap_unitcell)

            if (
                jnp.abs(overlap_AB - old_overlap_AB)
                <= varipeps_config.ctmrg_convergence_eps
            ):
                varipeps_config.ctmrg_fail_if_not_converged = tmp_fail
                varipeps_config.ctmrg_max_steps = tmp_max_steps
                break
        if count == (tmp_max_steps - 1):
            raise CTMRGNotConvergedError
    else:
        overlap_unitcell, _ = calc_ctmrg_env(
            [i.tensor for i in overlap_tensors], overlap_unitcell
        )

        overlap_AB = overlap_func(overlap_unitcell)

    return overlap_AB
