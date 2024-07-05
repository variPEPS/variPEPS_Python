import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

import jax.numpy as jnp
from jaxlib.xla_extension import XlaRuntimeError

from varipeps.contractions import apply_contraction_jitted
from varipeps.peps import PEPS_Unit_Cell


def calculate_correlation_length(unitcell: PEPS_Unit_Cell):
    use_full = True

    full_transfer_right = None
    full_transfer_bottom = None

    try:
        for _, view in unitcell.iter_one_row(0):
            transfer_right = apply_contraction_jitted(
                "corrlength_transfer_right",
                (view[0, 0][0][0].tensor,),
                (view[0, 0][0][0],),
                (),
            )
            transfer_right = transfer_right.reshape(
                np.prod(transfer_right.shape[:4]), np.prod(transfer_right.shape[4:])
            )

            if full_transfer_right is None:
                full_transfer_right = transfer_right
            else:
                full_transfer_right = full_transfer_right @ transfer_right

        initial_vector_right = apply_contraction_jitted(
            "corrlength_vector_right",
            (unitcell[0, -1][0][0].tensor,),
            (view[0, -1][0][0],),
            (),
        )
        initial_vector_right = initial_vector_right.reshape(-1)

        for _, view in unitcell.iter_one_column(0):
            transfer_bottom = apply_contraction_jitted(
                "corrlength_transfer_bottom",
                (view[0, 0][0][0].tensor,),
                (view[0, 0][0][0],),
                (),
            )
            transfer_bottom = transfer_bottom.reshape(
                np.prod(transfer_bottom.shape[:4]), np.prod(transfer_bottom.shape[4:])
            )

            if full_transfer_bottom is None:
                full_transfer_bottom = transfer_bottom
            else:
                full_transfer_bottom = full_transfer_bottom @ transfer_bottom

        initial_vector_bottom = apply_contraction_jitted(
            "corrlength_vector_bottom",
            (unitcell[-1, 0][0][0].tensor,),
            (view[-1, 0][0][0],),
            (),
        )
        initial_vector_bottom = initial_vector_bottom.reshape(-1)

        eig_right, eigvec_right = eigs(
            full_transfer_right, k=5, v0=initial_vector_right, which="LM"
        )

        eig_right = np.abs(eig_right)
        eig_right /= eig_right[0]

        corr_len_right = -1 / np.log(eig_right[1])

        eig_bottom, eigvec_bottom = eigs(
            full_transfer_bottom, k=5, v0=initial_vector_bottom, which="LM"
        )

        eig_bottom = np.abs(eig_bottom)
        eig_bottom /= eig_bottom[0]

        corr_len_bottom = -1 / np.log(eig_bottom[1])

        return (corr_len_right, eig_right), (corr_len_bottom, eig_bottom)
    except XlaRuntimeError:
        initial_vector_left = apply_contraction_jitted(
            "corrlength_vector_left",
            (unitcell[0, 0][0][0].tensor,),
            (view[0, 0][0][0],),
            (),
        )
        initial_vector_left = initial_vector_left.reshape(-1)

        initial_vector_top = apply_contraction_jitted(
            "corrlength_vector_top",
            (unitcell[0, 0][0][0].tensor,),
            (view[0, 0][0][0],),
            (),
        )
        initial_vector_top = initial_vector_top.reshape(-1)

        def left_matvec(vec):
            vec = jnp.asarray(vec)
            for _, view in unitcell.iter_one_row(0):
                if vec.ndim != 4:
                    vec = vec.reshape(
                        view[0, 0][0][0].T1.shape[0],
                        view[0, 0][0][0].tensor.shape[0],
                        view[0, 0][0][0].tensor.shape[0],
                        view[0, 0][0][0].T3.shape[0],
                    )
                vec = apply_contraction_jitted(
                    "corrlength_absorb_one_column",
                    (view[0, 0][0][0].tensor,),
                    (view[0, 0][0][0],),
                    (vec,),
                )
            return vec.reshape(-1)

        left_lin_op = LinearOperator(
            (initial_vector_left.shape[0], initial_vector_left.shape[0]),
            matvec=left_matvec,
        )

        eig_left, eigvec_left = eigs(
            left_lin_op, k=5, v0=initial_vector_left, which="LM"
        )

        eig_left = np.abs(eig_left)
        eig_left /= eig_left[0]

        corr_len_left = -1 / np.log(eig_left[1])

        def top_matvec(vec):
            vec = jnp.asarray(vec)
            for _, view in unitcell.iter_one_row(0):
                if vec.ndim != 4:
                    vec = vec.reshape(
                        view[0, 0][0][0].T4.shape[3],
                        view[0, 0][0][0].tensor.shape[4],
                        view[0, 0][0][0].tensor.shape[4],
                        view[0, 0][0][0].T2.shape[3],
                    )
                vec = apply_contraction_jitted(
                    "corrlength_absorb_one_row",
                    (view[0, 0][0][0].tensor,),
                    (view[0, 0][0][0],),
                    (vec,),
                )
            return vec.reshape(-1)

        top_lin_op = LinearOperator(
            (initial_vector_top.shape[0], initial_vector_top.shape[0]),
            matvec=top_matvec,
        )

        eig_top, eigvec_top = eigs(top_lin_op, k=5, v0=initial_vector_top, which="LM")

        eig_top = np.abs(eig_top)
        eig_top /= eig_top[0]

        corr_len_top = -1 / np.log(eig_top[1])

        return (corr_len_left, eig_left), (corr_len_top, eig_top)
