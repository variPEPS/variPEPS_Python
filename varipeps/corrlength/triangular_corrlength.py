import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

import jax.numpy as jnp

from varipeps.contractions import apply_contraction_jitted
from varipeps.peps import PEPS_Unit_Cell


def calculate_triangular_correlation_length(unitcell: PEPS_Unit_Cell):
    initial_vector_60 = apply_contraction_jitted(
        "triangular_ctmrg_corrlen_vec_60",
        (unitcell[-1, 0][0][0].tensor,),
        (unitcell[-1, 0][0][0],),
        (),
    )
    initial_vector_60 = initial_vector_60.reshape(-1)

    initial_vector_120 = apply_contraction_jitted(
        "triangular_ctmrg_corrlen_vec_120",
        (unitcell[-1, -1][0][0].tensor,),
        (unitcell[-1, -1][0][0],),
        (),
    )
    initial_vector_120 = initial_vector_120.reshape(-1)

    initial_vector_180 = apply_contraction_jitted(
        "triangular_ctmrg_corrlen_vec_180",
        (unitcell[0, -1][0][0].tensor,),
        (unitcell[0, -1][0][0],),
        (),
    )
    initial_vector_180 = initial_vector_180.reshape(-1)

    def matvec_60(vec):
        vec = jnp.asarray(vec)
        for _, view in unitcell.iter_one_column(0):
            if vec.ndim != 4:
                vec = vec.reshape(
                    view[0, 0][0][0].T6b.shape[3],
                    view[0, 0][0][0].tensor.shape[1],
                    view[0, 0][0][0].tensor.shape[1],
                    view[0, 0][0][0].T3a.shape[0],
                )
            vec = apply_contraction_jitted(
                "triangular_ctmrg_corrlen_absorb_60",
                (view[0, 0][0][0].tensor,),
                (view[0, 0][0][0],),
                (vec,),
            )
        return vec.reshape(-1)

    def matvec_120(vec):
        vec = jnp.asarray(vec)
        for _, view in unitcell.iter_main_diagonal():
            if vec.ndim != 4:
                vec = vec.reshape(
                    view[0, 0][0][0].T2a.shape[0],
                    view[0, 0][0][0].tensor.shape[0],
                    view[0, 0][0][0].tensor.shape[0],
                    view[0, 0][0][0].T5b.shape[3],
                )
            vec = apply_contraction_jitted(
                "triangular_ctmrg_corrlen_absorb_120",
                (view[0, 0][0][0].tensor,),
                (view[0, 0][0][0],),
                (vec,),
            )
        return vec.reshape(-1)

    def matvec_180(vec):
        vec = jnp.asarray(vec)
        for _, view in unitcell.iter_one_row(0):
            if vec.ndim != 4:
                vec = vec.reshape(
                    view[0, 0][0][0].T1a.shape[0],
                    view[0, 0][0][0].tensor.shape[5],
                    view[0, 0][0][0].tensor.shape[5],
                    view[0, 0][0][0].T4b.shape[3],
                )
            vec = apply_contraction_jitted(
                "triangular_ctmrg_corrlen_absorb_180",
                (view[0, 0][0][0].tensor,),
                (view[0, 0][0][0],),
                (vec,),
            )
        return vec.reshape(-1)

    lin_op_60 = LinearOperator(
        (initial_vector_60.shape[0], initial_vector_60.shape[0]),
        matvec=matvec_60,
    )

    lin_op_120 = LinearOperator(
        (initial_vector_120.shape[0], initial_vector_120.shape[0]),
        matvec=matvec_120,
    )

    lin_op_180 = LinearOperator(
        (initial_vector_180.shape[0], initial_vector_180.shape[0]),
        matvec=matvec_180,
    )

    eig_60, eigvec_60 = eigs(lin_op_60, k=5, v0=initial_vector_60, which="LM")

    eig_60 = eig_60[np.argsort(np.abs(eig_60))[::-1]]
    eig_60 /= np.abs(eig_60[0])

    corr_len_60 = -1 / np.log(np.abs(eig_60[1]))

    eig_120, eigvec_120 = eigs(lin_op_120, k=5, v0=initial_vector_120, which="LM")

    eig_120 = eig_120[np.argsort(np.abs(eig_120))[::-1]]
    eig_120 /= np.abs(eig_120[0])

    corr_len_120 = -1 / np.log(np.abs(eig_120[1]))

    eig_180, eigvec_180 = eigs(lin_op_180, k=5, v0=initial_vector_180, which="LM")

    eig_180 = eig_180[np.argsort(np.abs(eig_180))[::-1]]
    eig_180 /= np.abs(eig_180[0])

    corr_len_180 = -1 / np.log(np.abs(eig_180[1]))

    return (corr_len_60, eig_60), (corr_len_120, eig_120), (corr_len_180, eig_180)
