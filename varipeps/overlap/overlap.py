import jax.numpy as jnp

from varipeps.ctmrg import calc_ctmrg_env
from varipeps.peps import PEPS_Unit_Cell

from . import overlap_single_site
from . import overlap_four_sites

overlap_mapping = {
    (1, 1): overlap_single_site.Overlap_Single_Site,
    (2, 2): overlap_four_sites.Overlap_Four_Sites_Square,
}


def calculate_overlap(unitcell_A, unitcell_B, chi, max_chi):
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

    overlap_unitcell, _ = calc_ctmrg_env(
        [i.tensor for i in overlap_tensors], overlap_unitcell
    )

    overlap_AB = overlap_func(overlap_unitcell)

    return overlap_AB
