"""
Definitions for contractions in this module.
"""

import tensornetwork as tn

from typing import Dict, Optional, Union, List, Tuple, Sequence

Definition = Dict[
    str,
    List[Union[List[Union[str, List[str]]], List[Union[Tuple[int], List[Tuple[int]]]]]],
]


class Definitions:
    """
    Class to define the contractions used along in this package.

    The convention used for the ordering of the tensors is for a single PEPS
    tensor::

      single = [PEPS tensor, relevant CTM tensors starting at C1 and then clockwise]

    which is repeated for each PEPS tensor and its environment in the ordering
    fixed x and all corresponding y values in increasing order and then the next
    x value in increasing order. For example for a quadratic four site the
    structure would look like::

       (x, y)  -  (x, y+1)
         |           |
      (x+1, y) - (x+1, y+1)

      tensors = [single(x, y), single(x, y+1), single(x+1, y), single(x+1, y+1)]

    Using this convention each contraction is specified in the format::

      contraction_name = {
        "tensors": [
          [Relevant elements for first single PEPS tensors incl. env],
          [Relevant elements for second single PEPS tensors incl. env],
          ...,
          "Optional string elements indicating additional non-PEPS tensors (e.g. gates)"
        ],
        "network": [
          [
            (Axes for first element of the tensors for first PEPS Tensor),
            (Axes for second element of the tensors for first PEPS Tensor)
          ],
          [
            (Axes for first element of the tensors for second PEPS Tensor),
            (Axes for second element of the tensors for second PEPS Tensor)
          ],
          ...,
          (Axes for additional non-PEPS tensors)
        ]
      }

    The axes are hereby specified in the ncon format where positive numbers
    describes the axes to be contracted and negative number the open axes
    after the contraction.
    The elements are strings indicating the elements of the PEPS tensor object
    elements. Hereby the special element "tensor_conj" describes the
    conjugated and transposed PEPS tensor.
    """

    @staticmethod
    def _create_filter_and_network(
        contraction: Definition,
    ) -> Tuple[
        List[Sequence[str]],
        List[str],
        List[List[Tuple[int, ...]]],
        List[Tuple[int, ...]],
    ]:
        filter_peps_tensors: List[Sequence[str]] = []
        filter_additional_tensors: List[str] = []

        network_peps_tensors: List[List[Tuple[int, ...]]] = []
        network_additional_tensors: List[Tuple[int, ...]] = []

        for t in contraction["tensors"]:
            if isinstance(t, (list, tuple)):
                if len(filter_additional_tensors) != 0:
                    raise ValueError("Invalid specification for contraction.")

                filter_peps_tensors.append(t)
            else:
                filter_additional_tensors.append(t)

        for n in contraction["network"]:
            if isinstance(n, (list, tuple)) and all(
                isinstance(ni, (list, tuple)) for ni in n
            ):
                if len(network_additional_tensors) != 0:
                    raise ValueError("Invalid specification for contraction.")

                network_peps_tensors.append(n)  # type: ignore
            elif isinstance(n, (list, tuple)) and all(isinstance(ni, int) for ni in n):
                network_additional_tensors.append(n)  # type: ignore
            else:
                raise ValueError("Invalid specification for contraction.")

        if len(network_peps_tensors) != len(filter_peps_tensors) or not all(
            len(network_peps_tensors[i]) == len(filter_peps_tensors[i])
            for i in range(len(filter_peps_tensors))
        ):
            raise ValueError("Invalid specification for contraction.")

        return (
            filter_peps_tensors,
            filter_additional_tensors,
            network_peps_tensors,
            network_additional_tensors,
        )

    @classmethod
    def _process_def(cls, e):
        (
            filter_peps_tensors,
            filter_additional_tensors,
            network_peps_tensors,
            network_additional_tensors,
        ) = cls._create_filter_and_network(e)

        ncon_network = [
            j for i in network_peps_tensors for j in i
        ] + network_additional_tensors
        (
            mapped_ncon_network,
            mapping,
        ) = tn.ncon_interface._canonicalize_network_structure(ncon_network)
        flat_network = tuple(l for sublist in mapped_ncon_network for l in sublist)
        unique_flat_network = list(set(flat_network))

        out_order = tuple(
            sorted([l for l in unique_flat_network if l < 0], reverse=True)
        )
        con_order = tuple(sorted([l for l in unique_flat_network if l > 0]))
        sizes = tuple(len(l) for l in ncon_network)

        e["filter_peps_tensors"] = filter_peps_tensors
        e["filter_additional_tensors"] = filter_additional_tensors
        e["network_peps_tensors"] = network_peps_tensors
        e["network_additional_tensors"] = network_additional_tensors
        e["ncon_network"] = ncon_network
        e["ncon_flat_network"] = flat_network
        e["ncon_sizes"] = sizes
        e["ncon_con_order"] = con_order
        e["ncon_out_order"] = out_order

    @classmethod
    def _prepare_defs(cls):
        for e in dir(cls):
            if e.startswith("_"):
                continue

            e = getattr(cls, e)

            cls._process_def(e)

    density_matrix_one_site: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "C1", "T1", "C2", "T2", "C3", "T3", "C4", "T4"]
        ],
        "network": [
            [
                (6, 7, -1, 10, 9),  # tensor
                (13, 14, -2, 15, 16),  # tensor_conj
                (2, 12),  # C1
                (12, 9, 16, 3),  # T1
                (3, 8),  # C2
                (10, 15, 4, 8),  # T2
                (11, 4),  # C3
                (1, 11, 14, 7),  # T3
                (1, 5),  # C4
                (5, 13, 6, 2),  # T4
            ]
        ],
    }

    density_matrix_two_sites_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "C1", "T1", "T3", "C4", "T4"]],
        "network": [
            [
                (5, 9, -1, -4, 4),  # tensor
                (7, 10, -2, -5, 6),  # tensor_conj
                (1, 3),  # C1
                (3, 4, 6, -3),  # T1
                (2, -6, 10, 9),  # T3
                (2, 8),  # C4
                (8, 7, 5, 1),  # T4
            ]
        ],
    }

    density_matrix_two_sites_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2", "T3", "C3"]],
        "network": [
            [
                (-2, 8, -5, 5, 4),  # tensor
                (-3, 9, -6, 7, 6),  # tensor_conj
                (-1, 4, 6, 3),  # T1
                (3, 1),  # C2
                (5, 7, 10, 1),  # T2
                (-4, 2, 9, 8),  # T3
                (2, 10),  # C3
            ]
        ],
    }

    density_matrix_two_sites_top: Definition = {
        "tensors": [["tensor", "tensor_conj", "C1", "T1", "C2", "T2", "T4"]],
        "network": [
            [
                (8, -4, -1, 4, 5),  # tensor
                (9, -5, -2, 6, 7),  # tensor_conj
                (2, 10),  # C1
                (10, 5, 7, 1),  # T1
                (1, 3),  # C2
                (4, 6, -6, 3),  # T2
                (-3, 9, 8, 2),  # T4
            ]
        ],
    }

    density_matrix_two_sites_bottom: Definition = {
        "tensors": [["tensor", "tensor_conj", "T2", "C3", "T3", "C4", "T4"]],
        "network": [
            [
                (4, 5, -5, 8, -2),  # tensor
                (6, 7, -6, 9, -3),  # tensor_conj
                (8, 9, 2, -4),  # T2
                (10, 2),  # C3
                (1, 10, 7, 5),  # T3
                (1, 3),  # C4
                (3, 6, 4, -1),  # T4
            ]
        ],
    }

    density_matrix_four_sites_top_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "T4", "C1", "T1"]],
        "network": [
            [
                (3, -4, -1, -7, 4),  # tensor
                (5, -5, -2, -8, 6),  # tensor_conj
                (-3, 5, 3, 1),  # T4
                (1, 2),  # C1
                (2, 4, 6, -6),  # T1
            ]
        ],
    }

    density_matrix_four_sites_top_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2"]],
        "network": [
            [
                (-4, -8, -1, 4, 3),  # tensor
                (-5, -7, -2, 6, 5),  # tensor_conj
                (-3, 3, 5, 1),  # T1
                (1, 2),  # C2
                (4, 6, -6, 2),  # T2
            ]
        ],
    }

    density_matrix_four_sites_bottom_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "T3", "C4", "T4"]],
        "network": [
            [
                (3, 4, -1, -5, -7),  # tensor
                (5, 6, -2, -4, -8),  # tensor_conj
                (2, -3, 6, 4),  # T3
                (2, 1),  # C4
                (1, 5, 3, -6),  # T4
            ]
        ],
    }

    density_matrix_four_sites_bottom_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T2", "T3", "C3"]],
        "network": [
            [
                (-8, 3, -1, 4, -5),  # tensor
                (-7, 5, -2, 6, -4),  # tensor_conj
                (4, 6, 2, -3),  # T2
                (-6, 1, 5, 3),  # T3
                (1, 2),  # C3
            ]
        ],
    }

    density_matrix_three_sites_triangle_without_top_left: Definition = {
        "tensors": ["top-left", "top-right", "bottom-left", "bottom-right"],
        "network": [
            (10, 11, 12, 1, 2, 3),  # top-left
            (-1, -4, 1, 2, 3, 4, 5, 6),  # top-right
            (-2, -5, 7, 8, 9, 10, 11, 12),  # bottom-left
            (-3, -6, 4, 5, 6, 7, 8, 9),  # bottom-right
        ],
    }

    ctmrg_top_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "T4", "C1", "T1"]],
        "network": [
            [
                (3, -2, 7, -5, 4),  # tensor
                (5, -3, 7, -6, 6),  # tensor_conj
                (-1, 5, 3, 1),  # T4
                (1, 2),  # C1
                (2, 4, 6, -4),  # T1
            ]
        ],
    }

    ctmrg_top_left_large_d = {
        "tensors": [["tensor", "tensor_conj", "T4", "C1", "T1"]],
        "network": [
            [
                (4, -2, 3, -5, 5),  # tensor
                (6, -3, 3, -6, 7),  # tensor_conj
                (-1, 6, 4, 1),  # T4
                (1, 2),  # C1
                (2, 5, 7, -4),  # T1
            ]
        ],
    }

    ctmrg_top_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2"]],
        "network": [
            [
                (-2, -6, 7, 4, 3),  # tensor
                (-3, -5, 7, 6, 5),  # tensor_conj
                (-1, 3, 5, 1),  # T1
                (1, 2),  # C2
                (4, 6, -4, 2),  # T2
            ]
        ],
    }

    ctmrg_top_right_large_d: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2"]],
        "network": [
            [
                (-2, -6, 3, 5, 4),  # tensor
                (-3, -5, 3, 7, 6),  # tensor_conj
                (-1, 4, 6, 1),  # T1
                (1, 2),  # C2
                (5, 7, -4, 2),  # T2
            ]
        ],
    }

    ctmrg_bottom_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "T3", "C4", "T4"]],
        "network": [
            [
                (3, 4, 7, -3, -5),  # tensor
                (5, 6, 7, -2, -6),  # tensor_conj
                (2, -1, 6, 4),  # T3
                (2, 1),  # C4
                (1, 5, 3, -4),  # T4
            ]
        ],
    }

    ctmrg_bottom_left_large_d: Definition = {
        "tensors": [["tensor", "tensor_conj", "T3", "C4", "T4"]],
        "network": [
            [
                (4, 5, 3, -3, -5),  # tensor
                (6, 7, 3, -2, -6),  # tensor_conj
                (2, -1, 7, 5),  # T3
                (2, 1),  # C4
                (1, 6, 4, -4),  # T4
            ]
        ],
    }

    ctmrg_bottom_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T2", "T3", "C3"]],
        "network": [
            [
                (-6, 3, 7, 4, -3),  # tensor
                (-5, 5, 7, 6, -2),  # tensor_conj
                (4, 6, 2, -1),  # T2
                (-4, 1, 5, 3),  # T3
                (1, 2),  # C3
            ]
        ],
    }

    ctmrg_bottom_right_large_d: Definition = {
        "tensors": [["tensor", "tensor_conj", "T2", "T3", "C3"]],
        "network": [
            [
                (-6, 4, 3, 5, -3),  # tensor
                (-5, 6, 3, 7, -2),  # tensor_conj
                (5, 7, 2, -1),  # T2
                (-4, 1, 6, 4),  # T3
                (1, 2),  # C3
            ]
        ],
    }

    ctmrg_absorption_left_C1: Definition = {
        "tensors": [["C1", "T1"], "projector_left_bottom"],
        "network": [
            [
                (2, 1),  # C1
                (1, 3, 4, -2),  # T1
            ],
            (-1, 2, 3, 4),  # projector_left_bottom
        ],
    }

    ctmrg_absorption_left_T4: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T4"],
            "projector_left_top",
            "projector_left_bottom",
        ],
        "network": [
            [
                (3, 8, 6, -3, 2),  # tensor
                (4, 9, 6, -2, 5),  # tensor_conj
                (7, 4, 3, 1),  # T4
            ],
            (1, 2, 5, -4),  # projector_left_top
            (-1, 7, 8, 9),  # projector_left_bottom
        ],
    }

    ctmrg_absorption_left_T4_large_d: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T4"],
            "projector_left_top",
            "projector_left_bottom",
        ],
        "network": [
            [
                (4, 8, 2, -3, 3),  # tensor
                (5, 9, 2, -2, 6),  # tensor_conj
                (7, 5, 4, 1),  # T4
            ],
            (1, 3, 6, -4),  # projector_left_top
            (-1, 7, 8, 9),  # projector_left_bottom
        ],
    }

    ctmrg_absorption_left_C4: Definition = {
        "tensors": [["T3", "C4"], "projector_left_top"],
        "network": [
            [
                (1, -1, 4, 3),  # T3
                (1, 2),  # C4
            ],
            (2, 3, 4, -2),  # projector_left_top
        ],
    }

    ctmrg_absorption_right_C2: Definition = {
        "tensors": [["T1", "C2"], "projector_right_bottom"],
        "network": [
            [
                (-1, 3, 4, 1),  # T1
                (1, 2),  # C2
            ],
            (2, 4, 3, -2),  # projector_right_bottom
        ],
    }

    ctmrg_absorption_right_T2: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T2"],
            "projector_right_top",
            "projector_right_bottom",
        ],
        "network": [
            [
                (-1, 8, 6, 3, 2),  # tensor
                (-2, 9, 6, 5, 4),  # tensor_conj
                (3, 5, 7, 1),  # T2
            ],
            (-4, 1, 4, 2),  # projector_right_top
            (7, 9, 8, -3),  # projector_right_bottom
        ],
    }

    ctmrg_absorption_right_T2_large_d: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T2"],
            "projector_right_top",
            "projector_right_bottom",
        ],
        "network": [
            [
                (-1, 8, 2, 4, 3),  # tensor
                (-2, 9, 2, 6, 5),  # tensor_conj
                (4, 6, 7, 1),  # T2
            ],
            (-4, 1, 5, 3),  # projector_right_top
            (7, 9, 8, -3),  # projector_right_bottom
        ],
    }

    ctmrg_absorption_right_C3: Definition = {
        "tensors": [["C3", "T3"], "projector_right_top"],
        "network": [
            [
                (1, 2),  # C3
                (-1, 1, 4, 3),  # T3
            ],
            (-2, 2, 4, 3),  # projector_right_top
        ],
    }

    ctmrg_absorption_top_C1: Definition = {
        "tensors": [["T4", "C1"], "projector_top_right"],
        "network": [
            [
                (-1, 4, 3, 1),  # T4
                (1, 2),  # C1
            ],
            (2, 3, 4, -2),  # projector_top_right
        ],
    }

    ctmrg_absorption_top_T1: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T1"],
            "projector_top_left",
            "projector_top_right",
        ],
        "network": [
            [
                (3, -2, 6, 8, 2),  # tensor
                (5, -3, 6, 9, 4),  # tensor_conj
                (1, 2, 4, 7),  # T1
            ],
            (-1, 1, 3, 5),  # projector_top_left
            (7, 8, 9, -4),  # projector_top_right
        ],
    }

    ctmrg_absorption_top_T1_large_d: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T1"],
            "projector_top_left",
            "projector_top_right",
        ],
        "network": [
            [
                (4, -2, 2, 8, 3),  # tensor
                (6, -3, 2, 9, 5),  # tensor_conj
                (1, 3, 5, 7),  # T1
            ],
            (-1, 1, 4, 6),  # projector_top_left
            (7, 8, 9, -4),  # projector_top_right
        ],
    }

    ctmrg_absorption_top_C2: Definition = {
        "tensors": [["C2", "T2"], "projector_top_left"],
        "network": [
            [
                (2, 1),  # C2
                (3, 4, -2, 1),  # T2
            ],
            (-1, 2, 3, 4),  # projector_top_left
        ],
    }

    ctmrg_absorption_bottom_C4: Definition = {
        "tensors": [["C4", "T4"], "projector_bottom_right"],
        "network": [
            [
                (2, 1),  # C4
                (1, 4, 3, -2),  # T4
            ],
            (-1, 2, 4, 3),  # projector_bottom_right
        ],
    }

    ctmrg_absorption_bottom_T3: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T3"],
            "projector_bottom_left",
            "projector_bottom_right",
        ],
        "network": [
            [
                (3, 2, 6, 8, -4),  # tensor
                (5, 4, 6, 9, -3),  # tensor_conj
                (1, 7, 4, 2),  # T3
            ],
            (1, 5, 3, -1),  # projector_bottom_left
            (-2, 7, 9, 8),  # projector_bottom_right
        ],
    }

    ctmrg_absorption_bottom_T3_large_d: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T3"],
            "projector_bottom_left",
            "projector_bottom_right",
        ],
        "network": [
            [
                (4, 3, 2, 8, -4),  # tensor
                (6, 5, 2, 9, -3),  # tensor_conj
                (1, 7, 5, 3),  # T3
            ],
            (1, 6, 4, -1),  # projector_bottom_left
            (-2, 7, 9, 8),  # projector_bottom_right
        ],
    }

    ctmrg_absorption_bottom_C3: Definition = {
        "tensors": [["T2", "C3"], "projector_bottom_left"],
        "network": [
            [
                (3, 4, 1, -2),  # T2
                (2, 1),  # C3
            ],
            (2, 4, 3, -1),  # projector_bottom_left
        ],
    }

    ctmrg_gauge_fix_T1: Definition = {
        "tensors": [["T1"], "left_unitary", "right_unitary"],
        "network": [
            [(1, -2, -3, 2)],  # T1
            (-1, 1),  # left_unitary
            (-4, 2),  # right_unitary
        ],
    }

    ctmrg_gauge_fix_C2: Definition = {
        "tensors": [["C2"], "left_unitary", "bottom_unitary"],
        "network": [[(1, 2)], (-1, 1), (-2, 2)],  # C2  # left_unitary  # bottom_unitary
    }

    ctmrg_gauge_fix_T2: Definition = {
        "tensors": [["T2"], "top_unitary", "bottom_unitary"],
        "network": [
            [(-1, -2, 1, 2)],  # T2
            (-4, 2),  # top_unitary
            (-3, 1),  # bottom_unitary
        ],
    }

    ctmrg_gauge_fix_T3: Definition = {
        "tensors": [["T3"], "left_unitary", "right_unitary"],
        "network": [
            [(1, 2, -3, -4)],  # T3
            (1, -1),  # left_unitary
            (2, -2),  # right_unitary
        ],
    }

    ctmrg_gauge_fix_C4: Definition = {
        "tensors": [["C4"], "top_unitary", "right_unitary"],
        "network": [[(1, 2)], (2, -2), (1, -1)],  # C4  # top_unitary  # right_unitary
    }

    ctmrg_gauge_fix_T4: Definition = {
        "tensors": [["T4"], "top_unitary", "bottom_unitary"],
        "network": [
            [(1, -2, -3, 2)],  # T4
            (2, -4),  # top_unitary
            (1, -1),  # bottom_unitary
        ],
    }

    kagome_pess3_single_site_mapping: Definition = {
        "tensors": ["up_simplex", "down_simplex", "site1", "site2", "site3"],
        "network": [
            (1, 2, 3),  # up_simplex
            (-1, -7, 4),  # down_simplex
            (4, -3, 1),  # site1
            (-2, -4, 2),  # site2
            (-6, -5, 3),  # site3
        ],
    }

    kagome_downward_triangle_top_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "T4", "C1", "T1"]],
        "network": [
            [
                (3, -4, 7, 8, -1, -7, 4),  # tensor
                (5, -5, 7, 8, -2, -8, 6),  # tensor_conj
                (-3, 5, 3, 1),  # T4
                (1, 2),  # C1
                (2, 4, 6, -6),  # T1
            ]
        ],
    }

    kagome_downward_triangle_top_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2"]],
        "network": [
            [
                (-4, -8, 7, -1, 8, 4, 3),  # tensor
                (-5, -7, 7, -2, 8, 6, 5),  # tensor_conj
                (-3, 3, 5, 1),  # T1
                (1, 2),  # C2
                (4, 6, -6, 2),  # T2
            ]
        ],
    }

    kagome_downward_triangle_bottom_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "T3", "C4", "T4"]],
        "network": [
            [
                (3, 4, 7, 8, -1, -5, -7),  # tensor
                (5, 6, 7, 8, -2, -4, -8),  # tensor_conj
                (2, -3, 6, 4),  # T3
                (2, 1),  # C4
                (1, 5, 3, -6),  # T4
            ]
        ],
    }

    kagome_downward_triangle_bottom_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T2", "T3", "C3"]],
        "network": [
            [
                (-8, 3, -1, 7, 8, 4, -5),  # tensor
                (-7, 5, -2, 7, 8, 6, -4),  # tensor_conj
                (4, 6, 2, -3),  # T2
                (-6, 1, 5, 3),  # T3
                (1, 2),  # C3
            ]
        ],
    }

    honeycomb_density_matrix_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "C1", "T1", "T3", "C4", "T4"]],
        "network": [
            [
                (5, 9, 11, -1, -4, 4),  # tensor
                (7, 10, 11, -2, -5, 6),  # tensor_conj
                (1, 3),  # C1
                (3, 4, 6, -3),  # T1
                (2, -6, 10, 9),  # T3
                (2, 8),  # C4
                (8, 7, 5, 1),  # T4
            ]
        ],
    }

    honeycomb_density_matrix_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2", "T3", "C3"]],
        "network": [
            [
                (-2, 8, -5, 11, 5, 4),  # tensor
                (-3, 9, -6, 11, 7, 6),  # tensor_conj
                (-1, 4, 6, 3),  # T1
                (3, 1),  # C2
                (5, 7, 10, 1),  # T2
                (-4, 2, 9, 8),  # T3
                (2, 10),  # C3
            ]
        ],
    }

    honeycomb_density_matrix_top: Definition = {
        "tensors": [["tensor", "tensor_conj", "C1", "T1", "C2", "T2", "T4"]],
        "network": [
            [
                (8, -4, -1, 11, 4, 5),  # tensor
                (9, -5, -2, 11, 6, 7),  # tensor_conj
                (2, 10),  # C1
                (10, 5, 7, 1),  # T1
                (1, 3),  # C2
                (4, 6, -6, 3),  # T2
                (-3, 9, 8, 2),  # T4
            ]
        ],
    }

    honeycomb_density_matrix_bottom: Definition = {
        "tensors": [["tensor", "tensor_conj", "T2", "C3", "T3", "C4", "T4"]],
        "network": [
            [
                (4, 5, 11, -5, 8, -2),  # tensor
                (6, 7, 11, -6, 9, -3),  # tensor_conj
                (8, 9, 2, -4),  # T2
                (10, 2),  # C3
                (1, 10, 7, 5),  # T3
                (1, 3),  # C4
                (3, 6, 4, -1),  # T4
            ]
        ],
    }

    square_kagome_pess_mapping: Definition = {
        "tensors": [
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "simplex_left",
            "simplex_top",
            "simplex_right",
            "simplex_bottom",
        ],
        "network": [
            (-2, -3, 9),  # t1
            (7, -4, 1),  # t2
            (5, -5, 2),  # t3
            (6, -6, 3),  # t4
            (8, -7, 4),  # t5
            (-9, -8, 10),  # t6
            (-1, 5, 1),  # simplex_left
            (-10, 8, 2),  # simplex_top
            (10, 6, 4),  # simplex_right
            (9, 7, 3),  # simplex_bottom
        ],
    }

    triangular_pess_mapping: Definition = {
        "tensors": [
            "site",
            "simplex",
        ],
        "network": [
            (-1, 1, -2, -3),  # site
            (-5, 1, -4),  # simplex
        ],
    }

    maple_leaf_pess_mapping: Definition = {
        "tensors": ["t1", "t2", "t3", "up", "down"],
        "network": [
            (-3, -1, 1),  # t1
            (-4, -2, 2),  # t2
            (-5, 4, 3),  # t3
            (1, 2, 4),  # up
            (3, -7, -6),  # down
        ],
    }


Definitions._prepare_defs()
