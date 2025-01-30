"""
Definitions for contractions in this module.
"""

from collections import Counter

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
        name: str,
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
                    raise ValueError(f'Invalid specification for contraction "{name}".')

                filter_peps_tensors.append(t)
            else:
                filter_additional_tensors.append(t)

        for n in contraction["network"]:
            if isinstance(n, (list, tuple)) and all(
                isinstance(ni, (list, tuple)) for ni in n
            ):
                if len(network_additional_tensors) != 0:
                    raise ValueError(f'Invalid specification for contraction "{name}".')

                network_peps_tensors.append(n)  # type: ignore
            elif isinstance(n, (list, tuple)) and all(isinstance(ni, int) for ni in n):
                network_additional_tensors.append(n)  # type: ignore
            else:
                raise ValueError(f'Invalid specification for contraction "{name}".')

        if len(network_peps_tensors) != len(filter_peps_tensors) or not all(
            len(network_peps_tensors[i]) == len(filter_peps_tensors[i])
            for i in range(len(filter_peps_tensors))
        ):
            raise ValueError(f'Invalid specification for contraction "{name}".')

        return (
            filter_peps_tensors,
            filter_additional_tensors,
            network_peps_tensors,
            network_additional_tensors,
        )

    @classmethod
    def _process_def(cls, e, name):
        (
            filter_peps_tensors,
            filter_additional_tensors,
            network_peps_tensors,
            network_additional_tensors,
        ) = cls._create_filter_and_network(e, name)

        ncon_network = [
            j for i in network_peps_tensors for j in i
        ] + network_additional_tensors

        flatted_ncon_list = [j for i in ncon_network for j in i]
        counter_ncon_list = Counter(flatted_ncon_list)
        for ind, c in counter_ncon_list.items():
            if (ind > 0 and c != 2) or (ind < 0 and c != 1) or ind == 0:
                raise ValueError(
                    f'Invalid definition found for "{name}": Element {ind:d} has counter {c:d}.'
                )
        sorted_ncon_list = sorted(c for c in counter_ncon_list if c > 0)
        if len(sorted_ncon_list) != sorted_ncon_list[-1]:
            raise ValueError(
                f'Non-monotonous indices in definition "{name}". Please check!'
            )

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
        for name in dir(cls):
            if name.startswith("_"):
                continue

            e = getattr(cls, name)

            cls._process_def(e, name)

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

    density_matrix_one_site_C1_phase: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "C1_phase",
                "T1",
                "C2",
                "T2",
                "C3",
                "T3",
                "C4",
                "T4",
            ]
        ],
        "network": [
            [
                (6, 7, -1, 10, 9),  # tensor
                (13, 14, -2, 15, 16),  # tensor_conj
                (2, 12),  # C1_phase
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

    density_matrix_one_site_C2_phase: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "C1",
                "T1",
                "C2_phase",
                "T2",
                "C3",
                "T3",
                "C4",
                "T4",
            ]
        ],
        "network": [
            [
                (6, 7, -1, 10, 9),  # tensor
                (13, 14, -2, 15, 16),  # tensor_conj
                (2, 12),  # C1
                (12, 9, 16, 3),  # T1
                (3, 8),  # C2_phase
                (10, 15, 4, 8),  # T2
                (11, 4),  # C3
                (1, 11, 14, 7),  # T3
                (1, 5),  # C4
                (5, 13, 6, 2),  # T4
            ]
        ],
    }

    density_matrix_one_site_C3_phase: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "C1",
                "T1",
                "C2",
                "T2",
                "C3_phase",
                "T3",
                "C4",
                "T4",
            ]
        ],
        "network": [
            [
                (6, 7, -1, 10, 9),  # tensor
                (13, 14, -2, 15, 16),  # tensor_conj
                (2, 12),  # C1
                (12, 9, 16, 3),  # T1
                (3, 8),  # C2
                (10, 15, 4, 8),  # T2
                (11, 4),  # C3_phase
                (1, 11, 14, 7),  # T3
                (1, 5),  # C4
                (5, 13, 6, 2),  # T4
            ]
        ],
    }

    density_matrix_one_site_C4_phase: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "C1",
                "T1",
                "C2",
                "T2",
                "C3",
                "T3",
                "C4_phase",
                "T4",
            ]
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
                (1, 5),  # C4_phase
                (5, 13, 6, 2),  # T4
            ]
        ],
    }

    density_matrix_one_site_T1_phase: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "C1",
                "T1_phase",
                "C2",
                "T2",
                "C3",
                "T3",
                "C4",
                "T4",
            ]
        ],
        "network": [
            [
                (6, 7, -1, 10, 9),  # tensor
                (13, 14, -2, 15, 16),  # tensor_conj
                (2, 12),  # C1
                (12, 9, 16, 3),  # T1_phase
                (3, 8),  # C2
                (10, 15, 4, 8),  # T2
                (11, 4),  # C3
                (1, 11, 14, 7),  # T3
                (1, 5),  # C4
                (5, 13, 6, 2),  # T4
            ]
        ],
    }

    density_matrix_one_site_T2_phase: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "C1",
                "T1",
                "C2",
                "T2_phase",
                "C3",
                "T3",
                "C4",
                "T4",
            ]
        ],
        "network": [
            [
                (6, 7, -1, 10, 9),  # tensor
                (13, 14, -2, 15, 16),  # tensor_conj
                (2, 12),  # C1
                (12, 9, 16, 3),  # T1
                (3, 8),  # C2
                (10, 15, 4, 8),  # T2_phase
                (11, 4),  # C3
                (1, 11, 14, 7),  # T3
                (1, 5),  # C4
                (5, 13, 6, 2),  # T4
            ]
        ],
    }

    density_matrix_one_site_T3_phase: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "C1",
                "T1",
                "C2",
                "T2",
                "C3",
                "T3_phase",
                "C4",
                "T4",
            ]
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
                (1, 11, 14, 7),  # T3_phase
                (1, 5),  # C4
                (5, 13, 6, 2),  # T4
            ]
        ],
    }

    density_matrix_one_site_T4_phase: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "C1",
                "T1",
                "C2",
                "T2",
                "C3",
                "T3",
                "C4",
                "T4_phase",
            ]
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
                (5, 13, 6, 2),  # T4_phase
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

    density_matrix_two_sites_horizontal_rectangle: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T1"],  # top_middle
            ["tensor", "tensor_conj", "T3"],  # bottom_middle
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ],
        "network": [
            [
                (5, 11, 9, 19, 6),  # tensor
                (7, 13, 9, 20, 8),  # tensor_conj
                (4, 6, 8, 18),  # T1
            ],
            [
                (10, 16, 14, 25, 11),  # tensor
                (12, 17, 14, 26, 13),  # tensor_conj
                (15, 24, 17, 16),  # T3
            ],
            (-1, -3, 1, 2, 3, 4, 5, 7),  # top_left
            (18, 19, 20, 21, 22, 23),  # top_right
            (
                15,
                12,
                10,
                1,
                2,
                3,
            ),  # bottom_left
            (-2, -4, 21, 22, 23, 24, 26, 25),  # bottom_right
        ],
    }

    density_matrix_two_sites_vertical_rectangle: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T4"],
            ["tensor", "tensor_conj", "T2"],
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ],
        "network": [
            [
                (6, 19, 9, 11, 5),  # tensor
                (8, 20, 9, 13, 7),  # tensor_conj
                (18, 8, 6, 4),  # T4
            ],
            [
                (11, 22, 14, 16, 10),  # tensor
                (13, 23, 14, 17, 12),  # tensor_conj
                (16, 17, 21, 15),  # T2
            ],
            (-1, -3, 4, 5, 7, 1, 2, 3),  # top_left
            (1, 2, 3, 15, 12, 10),  # top_right
            (24, 25, 26, 18, 19, 20),  # bottom_left
            (-2, -4, 21, 23, 22, 24, 25, 26),  # bottom_right
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
    """
    .. figure:: /images/ctmrg_top_left.*
       :align: center
       :width: 60%
       :alt: Contraction for top left CTM corner with contraction order shown.

       Contraction for top left CTM corner.

    \\
    """

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
    """
    .. figure:: /images/ctmrg_top_left_large_d.*
       :align: center
       :width: 60%
       :alt: Contraction for top left CTM corner for high dimensions with contraction order shown.

       Contraction for top left CTM corner for high physical dimensions.

    \\
    """

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
    """
    .. figure:: /images/ctmrg_top_right.*
       :align: center
       :width: 60%
       :alt: Contraction for top right CTM corner with contraction order shown.

       Contraction for top right CTM corner.

    \\
    """

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
    """
    .. figure:: /images/ctmrg_top_right_large_d.*
       :align: center
       :width: 60%
       :alt: Contraction for top right CTM corner for high dimensions with contraction order shown.

       Contraction for top right CTM corner for high physical dimensions.

    \\
    """

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
    """
    .. figure:: /images/ctmrg_bottom_left.*
       :align: center
       :width: 60%
       :alt: Contraction for bottom left CTM corner with contraction order shown.

       Contraction for bottom left CTM corner.

    \\
    """

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
    """
    .. figure:: /images/ctmrg_bottom_left_large_d.*
       :align: center
       :width: 60%
       :alt: Contraction for bottom left CTM corner for high dimensions with contraction order shown.

       Contraction for bottom left CTM corner for high physical dimensions.

    \\
    """

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
    """
    .. figure:: /images/ctmrg_bottom_right.*
       :align: center
       :width: 60%
       :alt: Contraction for bottom right CTM corner with contraction order shown.

       Contraction for bottom right CTM corner.

    \\
    """

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
    """
    .. figure:: /images/ctmrg_bottom_right_large_d.*
       :align: center
       :width: 60%
       :alt: Contraction for bottom right CTM corner for high dimensions with contraction order shown.

       Contraction for bottom right CTM corner for high physical dimensions.

    \\
    """

    ctmrg_split_transfer_top: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T4_bra",
                "T4_ket",
                "C1",
                "T1_ket",
                "T1_bra",
                "C2",
                "T2_ket",
                "T2_bra",
            ]
        ],
        "network": [
            [
                (6, -2, 12, 8, 5),  # tensor
                (10, -4, 12, 14, 11),  # tensor_conj
                (-1, 10, 9),  # T4_bra
                (9, 6, 1),  # T4_ket
                (1, 2),  # C1
                (2, 5, 7),  # T1_ket
                (7, 11, 3),  # T1_bra
                (3, 4),  # C2
                (13, 8, 4),  # T2_ket
                (-3, 14, 13),  # T2_bra
            ]
        ],
    }

    ctmrg_split_transfer_bottom: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T2_ket",
                "T2_bra",
                "C3",
                "T3_bra",
                "T3_ket",
                "C4",
                "T4_bra",
                "T4_ket",
            ]
        ],
        "network": [
            [
                (14, 11, 12, 10, -4),  # tensor
                (8, 5, 12, 6, -2),  # tensor_conj
                (9, 10, -1),  # T2_ket
                (4, 6, 9),  # T2_bra
                (3, 4),  # C3
                (7, 5, 3),  # T3_bra
                (2, 11, 7),  # T3_ket
                (2, 1),  # C4
                (1, 8, 13),  # T4_bra
                (13, 14, -3),  # T4_ket
            ]
        ],
    }

    ctmrg_split_transfer_left: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T3_bra",
                "T3_ket",
                "C4",
                "T4_bra",
                "T4_ket",
                "C1",
                "T1_ket",
                "T1_bra",
            ]
        ],
        "network": [
            [
                (5, 8, 12, -4, 6),  # tensor
                (11, 14, 12, -2, 10),  # tensor_conj
                (13, 14, -1),  # T3_bra
                (1, 8, 13),  # T3_ket
                (1, 2),  # C4
                (2, 11, 7),  # T4_bra
                (7, 5, 3),  # T4_ket
                (3, 4),  # C1
                (4, 6, 9),  # T1_ket
                (9, 10, -3),  # T1_bra
            ]
        ],
    }

    ctmrg_split_transfer_right: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T1_ket",
                "T1_bra",
                "C2",
                "T2_ket",
                "T2_bra",
                "C3",
                "T3_bra",
                "T3_ket",
            ]
        ],
        "network": [
            [
                (-2, 10, 12, 11, 14),  # tensor
                (-4, 5, 12, 6, 8),  # tensor_conj
                (-1, 14, 13),  # T1_ket
                (13, 8, 1),  # T1_bra
                (1, 2),  # C2
                (7, 11, 2),  # T2_ket
                (3, 6, 7),  # T2_bra
                (4, 3),  # C3
                (9, 5, 4),  # T3_bra
                (-3, 10, 9),  # T3_ket
            ]
        ],
    }

    ctmrg_split_transfer_top_full: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T4_bra",
                "T4_ket",
                "C1",
                "T1_ket",
                "T1_bra",
            ],
            [
                "tensor",
                "tensor_conj",
                "T1_ket",
                "T1_bra",
                "C2",
                "T2_ket",
                "T2_bra",
            ],
            "projector_left_bottom_ket",
            "projector_right_bottom_bra",
        ],
        "network": [
            [
                (4, 7, 10, 23, 5),  # tensor
                (9, -2, 10, 24, 11),  # tensor_conj
                (1, 9, 6),  # T4_bra
                (6, 4, 3),  # T4_ket
                (3, 2),  # C1
                (2, 5, 8),  # T1_ket
                (8, 11, 25),  # T1_bra
            ],
            [
                (23, -4, 20, 21, 22),  # tensor
                (24, 15, 20, 16, 18),  # tensor_conj
                (25, 22, 19),  # T1_ket
                (19, 18, 13),  # T1_bra
                (13, 12),  # C2
                (17, 21, 12),  # T2_ket
                (14, 16, 17),  # T2_bra
            ],
            (-1, 1, 7),  # projector_left_bottom_ket
            (14, 15, -3),  # projector_right_bottom_bra
        ],
    }

    ctmrg_split_transfer_left_top_half: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T4_bra",
                "T4_ket",
                "C1",
                "T1_ket",
                "T1_bra",
            ],
            "projector_left_bottom_ket",
            "projector_top_right_ket",
        ],
        "network": [
            [
                (5, 8, 13, 10, 6),  # tensor
                (11, -2, 13, -4, 12),  # tensor_conj
                (1, 11, 7),  # T4_bra
                (7, 5, 2),  # T4_ket
                (2, 3),  # C1
                (3, 6, 9),  # T1_ket
                (9, 12, 4),  # T1_bra
            ],
            (-1, 1, 8),  # projector_left_bottom_ket
            (4, 10, -3),  # projector_top_right_ket
        ],
    }

    ctmrg_split_transfer_right_top_half: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T1_ket",
                "T1_bra",
                "C2",
                "T2_ket",
                "T2_bra",
            ],
            "projector_top_left_ket",
            "projector_right_bottom_bra",
        ],
        "network": [
            [
                (5, -4, 13, 8, 6),  # tensor
                (-2, 10, 13, 9, 12),  # tensor_conj
                (1, 6, 7),  # T1_ket
                (7, 12, 2),  # T1_bra
                (2, 3),  # C2
                (11, 8, 3),  # T2_ket
                (4, 9, 11),  # T2_bra
            ],
            (-1, 1, 5),  # projector_top_left_ket
            (4, 10, -3),  # projector_right_bottom_bra
        ],
    }

    ctmrg_split_transfer_bottom_full: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T4_ket",
                "T4_bra",
                "C4",
                "T3_ket",
                "T3_bra",
            ],
            [
                "tensor",
                "tensor_conj",
                "T3_ket",
                "T3_bra",
                "C3",
                "T2_bra",
                "T2_ket",
            ],
            "projector_left_top_ket",
            "projector_right_top_bra",
        ],
        "network": [
            [
                (16, 18, 20, 25, 15),  # tensor
                (21, 22, 20, 24, -4),  # tensor_conj
                (17, 16, 14),  # T4_ket
                (12, 21, 17),  # T4_bra
                (13, 12),  # C4
                (13, 18, 19),  # T3_ket
                (19, 22, 23),  # T3_bra
            ],
            [
                (25, 11, 10, 9, -2),  # tensor
                (24, 5, 10, 4, 7),  # tensor_conj
                (23, 11, 8),  # T3_ket
                (8, 5, 2),  # T3_bra
                (2, 3),  # C3
                (3, 4, 6),  # T2_bra
                (6, 9, 1),  # T2_ket
            ],
            (14, 15, -3),  # projector_left_top_ket
            (-1, 1, 7),  # projector_right_top_bra
        ],
    }

    ctmrg_split_transfer_left_bottom_half: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T4_ket",
                "T4_bra",
                "C4",
                "T3_ket",
                "T3_bra",
            ],
            "projector_left_top_ket",
            "projector_bottom_right_bra",
        ],
        "network": [
            [
                (5, 8, 13, -2, 6),  # tensor
                (12, 10, 13, 9, -4),  # tensor_conj
                (7, 5, 1),  # T4_ket
                (2, 12, 7),  # T4_bra
                (3, 2),  # C4
                (3, 8, 11),  # T3_ket
                (11, 10, 4),  # T3_bra
            ],
            (1, 6, -3),  # projector_left_top_ket
            (-1, 4, 9),  # projector_bottom_right_bra
        ],
    }

    ctmrg_split_transfer_right_bottom_half: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T3_ket",
                "T3_bra",
                "C3",
                "T2_bra",
                "T2_ket",
            ],
            "projector_bottom_right_bra",
            "projector_right_top_bra",
        ],
        "network": [
            [
                (-4, 11, 13, 12, -2),  # tensor
                (8, 5, 13, 6, 10),  # tensor_conj
                (1, 11, 7),  # T3_ket
                (7, 5, 2),  # T3_bra
                (2, 3),  # C3
                (3, 6, 9),  # T2_bra
                (9, 12, 4),  # T2_ket
            ],
            (1, 8, -3),  # projector_bottom_right_bra
            (-1, 4, 10),  # projector_right_top_bra
        ],
    }

    ctmrg_split_transfer_left_full: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T1_bra",
                "T1_ket",
                "C1",
                "T4_ket",
                "T4_bra",
            ],
            [
                "tensor",
                "tensor_conj",
                "T4_ket",
                "T4_bra",
                "C4",
                "T3_ket",
                "T3_bra",
            ],
            "projector_top_right_ket",
            "projector_bottom_right_bra",
        ],
        "network": [
            [
                (4, 24, 10, 7, 5),  # tensor
                (11, 25, 10, -4, 9),  # tensor_conj
                (6, 9, 1),  # T1_bra
                (3, 5, 6),  # T1_ket
                (2, 3),  # C1
                (8, 4, 2),  # T4_ket
                (23, 11, 8),  # T4_bra
            ],
            [
                (22, 21, 20, -2, 24),  # tensor
                (18, 16, 20, 15, 25),  # tensor_conj
                (19, 22, 23),  # T4_ket
                (13, 18, 19),  # T4_bra
                (12, 13),  # C4
                (12, 21, 17),  # T3_ket
                (17, 16, 14),  # T3_bra
            ],
            (1, 7, -3),  # projector_top_right_ket
            (-1, 14, 15),  # projector_bottom_right_bra
        ],
    }

    ctmrg_split_transfer_right_full: Definition = {
        "tensors": [
            [
                "tensor",
                "tensor_conj",
                "T1_ket",
                "T1_bra",
                "C2",
                "T2_ket",
                "T2_bra",
            ],
            [
                "tensor",
                "tensor_conj",
                "T2_ket",
                "T2_bra",
                "C3",
                "T3_bra",
                "T3_ket",
            ],
            "projector_top_left_ket",
            "projector_bottom_left_bra",
        ],
        "network": [
            [
                (15, 25, 20, 18, 16),  # tensor
                (-2, 24, 20, 22, 21),  # tensor_conj
                (14, 16, 17),  # T1_ket
                (17, 21, 12),  # T1_bra
                (12, 13),  # C2
                (19, 18, 13),  # T2_ket
                (23, 22, 19),  # T2_bra
            ],
            [
                (-4, 9, 10, 11, 25),  # tensor
                (7, 5, 10, 4, 24),  # tensor_conj
                (8, 11, 23),  # T2_ket
                (2, 4, 8),  # T2_bra
                (3, 2),  # C3
                (6, 5, 3),  # T3_bra
                (1, 9, 6),  # T3_ket
            ],
            (-1, 14, 15),  # projector_top_left_ket
            (1, 7, -3),  # projector_bottom_left_bra
        ],
    }

    ctmrg_split_transfer_phys_left_top: Definition = {
        "tensors": [
            [
                "tensor",
                "T4_ket",
                "C1",
                "T1_ket",
                "T1_bra",
            ]
        ],
        "network": [
            [
                (3, -2, -4, -6, 4),  # tensor
                (-1, 3, 1),  # T4_ket
                (1, 2),  # C1
                (2, 4, 5),  # T1_ket
                (5, -3, -5),  # T1_bra
            ]
        ],
    }

    ctmrg_split_transfer_phys_left_bottom: Definition = {
        "tensors": [
            [
                "tensor_conj",
                "T4_bra",
                "C4",
                "T3_ket",
                "T3_bra",
            ]
        ],
        "network": [
            [
                (4, 5, -6, -2, -5),  # tensor
                (1, 4, -3),  # T4_bra
                (2, 1),  # C4
                (2, -4, 3),  # T3_ket
                (3, 5, -1),  # T3_bra
            ]
        ],
    }

    ctmrg_split_transfer_phys_right_top: Definition = {
        "tensors": [
            [
                "tensor",
                "T1_ket",
                "T1_bra",
                "C2",
                "T2_ket",
            ]
        ],
        "network": [
            [
                (-2, -4, -6, 4, 5),  # tensor
                (-1, 5, 3),  # T1_ket
                (3, -5, 2),  # T1_bra
                (2, 1),  # C2
                (-3, 4, 1),  # T2_ket
            ]
        ],
    }

    ctmrg_split_transfer_phys_right_bottom: Definition = {
        "tensors": [
            [
                "tensor_conj",
                "T2_bra",
                "C3",
                "T3_bra",
                "T3_ket",
            ]
        ],
        "network": [
            [
                (-6, 4, -4, 3, -3),  # tensor_conj
                (1, 3, -1),  # T2_bra
                (2, 1),  # C3
                (5, 4, 2),  # T3_bra
                (-5, -2, 5),  # T3_ket
            ]
        ],
    }

    ctmrg_split_transfer_phys_top_left: Definition = {
        "tensors": [
            [
                "tensor",
                "T4_bra",
                "T4_ket",
                "C1",
                "T1_ket",
            ]
        ],
        "network": [
            [
                (4, -2, -6, -4, 3),  # tensor
                (-1, -5, 5),  # T4_bra
                (5, 4, 2),  # T4_ket
                (2, 1),  # C1
                (1, 3, -3),  # T1_ket
            ]
        ],
    }

    ctmrg_split_transfer_phys_top_right: Definition = {
        "tensors": [
            [
                "tensor_conj",
                "T1_bra",
                "C2",
                "T2_ket",
                "T2_bra",
            ]
        ],
        "network": [
            [
                (-3, -6, -4, 5, 4),  # tensor_conj
                (-1, 4, 1),  # T1_bra
                (1, 2),  # C2
                (3, -2, 2),  # T2_ket
                (-5, 5, 3),  # T2_bra
            ]
        ],
    }

    ctmrg_split_transfer_phys_bottom_left: Definition = {
        "tensors": [
            [
                "tensor",
                "T4_ket",
                "T4_bra",
                "C4",
                "T3_ket",
            ]
        ],
        "network": [
            [
                (5, 4, -4, -2, -6),  # tensor
                (3, 5, -5),  # T4_ket
                (2, -3, 3),  # T4_bra
                (1, 2),  # C4
                (1, 4, -1),  # T3_ket
            ]
        ],
    }

    ctmrg_split_transfer_phys_bottom_right: Definition = {
        "tensors": [
            [
                "tensor_conj",
                "T3_bra",
                "C3",
                "T2_bra",
                "T2_ket",
            ]
        ],
        "network": [
            [
                (-5, 4, -6, 3, -2),  # tensor_conj
                (-3, 4, 2),  # T3_bra
                (2, 1),  # C3
                (1, 3, 5),  # T2_bra
                (5, -4, -1),  # T2_ket
            ]
        ],
    }

    ctmrg_split_transfer_phys_top: Definition = {
        "tensors": [
            [
                "tensor",
                "T4_ket",
                "C1",
                "T1_ket",
                "T1_bra",
                "C2",
                "T2_ket",
            ]
        ],
        "network": [
            [
                (5, -2, -5, 8, 6),  # tensor
                (-1, 5, 1),  # T4_ket
                (1, 2),  # C1
                (2, 6, 7),  # T1_ket
                (7, -4, 3),  # T1_bra
                (3, 4),  # C2
                (-3, 8, 4),  # T2_ket
            ]
        ],
    }

    ctmrg_split_transfer_phys_bottom: Definition = {
        "tensors": [
            [
                "tensor_conj",
                "T2_bra",
                "C3",
                "T3_bra",
                "T3_ket",
                "C4",
                "T4_bra",
            ]
        ],
        "network": [
            [
                (8, 6, -3, 5, -2),  # tensor_conj
                (1, 5, -1),  # T2_bra
                (2, 1),  # C3
                (7, 6, 2),  # T3_bra
                (3, -5, 7),  # T3_ket
                (3, 4),  # C4
                (4, 8, -4),  # T4_bra
            ]
        ],
    }

    ctmrg_split_transfer_phys_left: Definition = {
        "tensors": [
            [
                "tensor",
                "T3_ket",
                "C4",
                "T4_bra",
                "T4_ket",
                "C1",
                "T1_ket",
            ]
        ],
        "network": [
            [
                (6, 8, -3, -5, 5),  # tensor
                (4, 8, -1),  # T3_ket
                (4, 3),  # C4
                (3, -2, 7),  # T4_bra
                (7, 6, 2),  # T4_ket
                (2, 1),  # C1
                (1, 5, -4),  # T1_ket
            ]
        ],
    }

    ctmrg_split_transfer_phys_right: Definition = {
        "tensors": [
            [
                "tensor_conj",
                "T1_bra",
                "C2",
                "T2_ket",
                "T2_bra",
                "C3",
                "T3_bra",
            ]
        ],
        "network": [
            [
                (-4, 5, -5, 6, 8),  # tensor_conj
                (-1, 8, 4),  # T1_bra
                (4, 3),  # C2
                (7, -2, 3),  # T2_ket
                (2, 6, 7),  # T2_bra
                (1, 2),  # C3
                (-3, 5, 1),  # T3_bra
            ]
        ],
    }

    ctmrg_split_transfer_phys_top_full: Definition = {
        "tensors": [
            [
                "tensor",
                "T4_ket",
                "C1",
                "T1_ket",
                "T1_bra",
            ],
            [
                "tensor",
                "T1_ket",
                "T1_bra",
                "C2",
                "T2_ket",
            ],
            "projector_left_bottom_ket",
            "projector_right_bottom_bra",
        ],
        "network": [
            [
                (3, 6, -3, 16, 4),  # tensor
                (5, 3, 1),  # T4_ket
                (1, 2),  # C1
                (2, 4, 7),  # T1_ket
                (7, -2, 15),  # T1_bra
            ],
            [
                (16, -5, -6, 13, 14),  # tensor
                (15, 14, 12),  # T1_ket
                (12, 11, 8),  # T1_bra
                (8, 9),  # C2
                (10, 13, 9),  # T2_ket
            ],
            (-1, 5, 6),  # projector_left_bottom_phys
            (10, 11, -4),  # projector_right_bottom_phys
        ],
    }

    ctmrg_split_transfer_phys_bottom_full: Definition = {
        "tensors": [
            [
                "tensor_conj",
                "T4_bra",
                "C4",
                "T3_ket",
                "T3_bra",
            ],
            [
                "tensor_conj",
                "T3_ket",
                "T3_bra",
                "C3",
                "T2_bra",
            ],
            "projector_left_top_phys",
            "projector_right_top_phys",
        ],
        "network": [
            [
                (13, 14, -6, 16, -5),  # tensor_conj
                (9, 13, 10),  # T4_bra
                (8, 9),  # C4
                (8, 11, 12),  # T3_ket
                (12, 14, 15),  # T3_bra
            ],
            [
                (16, 4, -3, 3, 6),  # tensor_conj
                (15, -2, 7),  # T3_ket
                (7, 4, 2),  # T3_bra
                (2, 1),  # C3
                (1, 3, 5),  # T2_bra
            ],
            (10, 11, -4),  # projector_left_top_phys
            (-1, 5, 6),  # projector_right_top_phys
        ],
    }

    ctmrg_split_transfer_phys_left_full: Definition = {
        "tensors": [
            [
                "tensor",
                "T1_ket",
                "C1",
                "T4_ket",
                "T4_bra",
            ],
            [
                "tensor",
                "T4_ket",
                "T4_bra",
                "C4",
                "T3_ket",
            ],
            "projector_top_right_phys",
            "projector_bottom_right_phys",
        ],
        "network": [
            [
                (4, 16, -6, 6, 3),  # tensor
                (1, 3, 5),  # T1_ket
                (2, 1),  # C1
                (7, 4, 2),  # T4_ket
                (15, -5, 7),  # T4_bra
            ],
            [
                (14, 13, -3, -2, 16),  # tensor
                (12, 14, 15),  # T4_ket
                (8, 11, 12),  # T4_bra
                (9, 8),  # C4
                (9, 13, 10),  # T3_ket
            ],
            (5, 6, -4),  # projector_top_right_phys
            (-1, 10, 11),  # projector_bottom_right_phys
        ],
    }

    ctmrg_split_transfer_phys_right_full: Definition = {
        "tensors": [
            [
                "tensor_conj",
                "T1_bra",
                "C2",
                "T2_ket",
                "T2_bra",
            ],
            [
                "tensor_conj",
                "T2_ket",
                "T2_bra",
                "C3",
                "T3_bra",
            ],
            "projector_top_left_phys",
            "projector_bottom_left_phys",
        ],
        "network": [
            [
                (-2, 16, -3, 14, 13),  # tensor_conj
                (10, 13, 9),  # T1_bra
                (9, 8),  # C2
                (12, 11, 8),  # T2_ket
                (15, 14, 12),  # T2_bra
            ],
            [
                (6, 3, -6, 4, 16),  # tensor_conj
                (7, -5, 15),  # T2_ket
                (2, 4, 7),  # T2_bra
                (1, 2),  # C3
                (5, 3, 1),  # T3_bra
            ],
            (-1, 10, 11),  # projector_top_left_phys
            (5, 6, -4),  # projector_bottom_left_phys
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
    """
    .. figure:: /images/CTMRG_Absorption_L_C1.*
       :align: center
       :width: 60%
       :alt: Contraction for C1 during left absorption with contraction order shown.

       Contraction for C1 during left absorption.

    \\
    """

    ctmrg_absorption_left_C1_phase_1: Definition = {
        "tensors": [["C1_phase", "T1"], "projector_left_bottom"],
        "network": [
            [
                (2, 1),  # C1_phase
                (1, 3, 4, -2),  # T1
            ],
            (-1, 2, 3, 4),  # projector_left_bottom
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_L_C1.*
       :align: center
       :width: 60%
       :alt: Contraction for C1 during left absorption with contraction order shown.

       Contraction for C1 during left absorption. Here C1 is the tensor with
       structure factor phase included which is grown by a normal T1 tensor.

    \\
    """

    ctmrg_absorption_left_C1_phase_2: Definition = {
        "tensors": [["C1", "T1_phase"], "projector_left_bottom"],
        "network": [
            [
                (2, 1),  # C1
                (1, 3, 4, -2),  # T1_phase
            ],
            (-1, 2, 3, 4),  # projector_left_bottom
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_L_C1.*
       :align: center
       :width: 60%
       :alt: Contraction for C1 during left absorption with contraction order shown.

       Contraction for C1 during left absorption. Here C1 is the tensor with
       structure factor phase not included yet which is grown by a T1 tensor
       which includes a phase already.

    \\
    """

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
    """
    .. figure:: /images/CTMRG_Absorption_L_T4.*
       :align: center
       :width: 60%
       :alt: Contraction for T4 during left absorption with contraction order shown.

       Contraction for T4 during left absorption.

    \\
    """

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

    ctmrg_absorption_left_T4_large_d_phase_1: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T4_phase"],
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

    ctmrg_absorption_left_T4_large_d_phase_2: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T4"],
            "projector_left_top",
            "projector_left_bottom",
            "structure_factor_gate",
        ],
        "network": [
            [
                (5, 9, 2, -3, 4),  # tensor
                (6, 10, 3, -2, 7),  # tensor_conj
                (8, 6, 5, 1),  # T4
            ],
            (1, 4, 7, -4),  # projector_left_top
            (-1, 8, 9, 10),  # projector_left_bottom
            (2, 3),  # structure_factor_gate
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
    """
    .. figure:: /images/CTMRG_Absorption_L_C4.*
       :align: center
       :width: 60%
       :alt: Contraction for C4 during left absorption with contraction order shown.

       Contraction for C4 during left absorption.

    \\
    """

    ctmrg_absorption_left_C4_phase_1: Definition = {
        "tensors": [["T3", "C4_phase"], "projector_left_top"],
        "network": [
            [
                (1, -1, 4, 3),  # T3
                (1, 2),  # C4_phase
            ],
            (2, 3, 4, -2),  # projector_left_top
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_L_C4.*
       :align: center
       :width: 60%
       :alt: Contraction for C4 during left absorption with contraction order shown.

       Contraction for C4 during left absorption. Here C4 is the tensor with
       structure factor phase included which is grown by a normal T3 tensor.

    \\
    """

    ctmrg_absorption_left_C4_phase_2: Definition = {
        "tensors": [["T3_phase", "C4"], "projector_left_top"],
        "network": [
            [
                (1, -1, 4, 3),  # T3_phase
                (1, 2),  # C4
            ],
            (2, 3, 4, -2),  # projector_left_top
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_L_C4.*
       :align: center
       :width: 60%
       :alt: Contraction for C4 during left absorption with contraction order shown.

       Contraction for C4 during left absorption. Here C3 is the tensor with
       structure factor phase not included yet which is grown by a T3 tensor
       which includes a phase already.

    \\
    """

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
    """
    .. figure:: /images/CTMRG_Absorption_R_C2.*
       :align: center
       :width: 60%
       :alt: Contraction for C2 during right absorption with contraction order shown.

       Contraction for C2 during right absorption.

    \\
    """

    ctmrg_absorption_right_C2_phase_1: Definition = {
        "tensors": [["T1", "C2_phase"], "projector_right_bottom"],
        "network": [
            [
                (-1, 3, 4, 1),  # T1
                (1, 2),  # C2_phase
            ],
            (2, 4, 3, -2),  # projector_right_bottom
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_R_C2.*
       :align: center
       :width: 60%
       :alt: Contraction for C2 during right absorption with contraction order shown.

       Contraction for C2 during right absorption. Here C2 is the tensor with
       structure factor phase included which is grown by a normal T1 tensor.

    \\
    """

    ctmrg_absorption_right_C2_phase_2: Definition = {
        "tensors": [["T1_phase", "C2"], "projector_right_bottom"],
        "network": [
            [
                (-1, 3, 4, 1),  # T1_phase
                (1, 2),  # C2
            ],
            (2, 4, 3, -2),  # projector_right_bottom
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_R_C2.*
       :align: center
       :width: 60%
       :alt: Contraction for C2 during right absorption with contraction order shown.

       Contraction for C2 during right absorption. Here C2 is the tensor with
       structure factor phase not included yet which is grown by a T1 tensor
       which includes a phase already.

    \\
    """

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
    """
    .. figure:: /images/CTMRG_Absorption_R_T2.*
       :align: center
       :width: 60%
       :alt: Contraction for T2 during right absorption with contraction order shown.

       Contraction for T2 during right absorption.

    \\
    """

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

    ctmrg_absorption_right_T2_large_d_phase_1: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T2_phase"],
            "projector_right_top",
            "projector_right_bottom",
        ],
        "network": [
            [
                (-1, 8, 2, 4, 3),  # tensor
                (-2, 9, 2, 6, 5),  # tensor_conj
                (4, 6, 7, 1),  # T2_phase
            ],
            (-4, 1, 5, 3),  # projector_right_top
            (7, 9, 8, -3),  # projector_right_bottom
        ],
    }

    ctmrg_absorption_right_T2_large_d_phase_2: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T2"],
            "projector_right_top",
            "projector_right_bottom",
            "structure_factor_gate",
        ],
        "network": [
            [
                (-1, 9, 2, 5, 4),  # tensor
                (-2, 10, 3, 7, 6),  # tensor_conj
                (5, 7, 8, 1),  # T2_phase
            ],
            (-4, 1, 6, 4),  # projector_right_top
            (8, 10, 9, -3),  # projector_right_bottom
            (2, 3),  # structure_factor_gate
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
    """
    .. figure:: /images/CTMRG_Absorption_R_C3.*
       :align: center
       :width: 60%
       :alt: Contraction for C3 during right absorption with contraction order shown.

       Contraction for C3 during right absorption.

    \\
    """

    ctmrg_absorption_right_C3_phase_1: Definition = {
        "tensors": [["C3_phase", "T3"], "projector_right_top"],
        "network": [
            [
                (1, 2),  # C3_phase
                (-1, 1, 4, 3),  # T3
            ],
            (-2, 2, 4, 3),  # projector_right_top
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_R_C3.*
       :align: center
       :width: 60%
       :alt: Contraction for C3 during right absorption with contraction order shown.

       Contraction for C3 during right absorption. Here C3 is the tensor with
       structure factor phase included which is grown by a normal T3 tensor.

    \\
    """

    ctmrg_absorption_right_C3_phase_2: Definition = {
        "tensors": [["C3", "T3_phase"], "projector_right_top"],
        "network": [
            [
                (1, 2),  # C3
                (-1, 1, 4, 3),  # T3_phase
            ],
            (-2, 2, 4, 3),  # projector_right_top
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_R_C3.*
       :align: center
       :width: 60%
       :alt: Contraction for C3 during right absorption with contraction order shown.

       Contraction for C3 during right absorption. Here C3 is the tensor with
       structure factor phase not included yet which is grown by a T3 tensor
       which includes a phase already.

    \\
    """

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
    """
    .. figure:: /images/CTMRG_Absorption_U_C1.*
       :align: center
       :width: 60%
       :alt: Contraction for C1 during top absorption with contraction order shown.

       Contraction for C1 during top absorption.

    \\
    """

    ctmrg_absorption_top_C1_phase_1: Definition = {
        "tensors": [["T4", "C1_phase"], "projector_top_right"],
        "network": [
            [
                (-1, 4, 3, 1),  # T4
                (1, 2),  # C1_phase
            ],
            (2, 3, 4, -2),  # projector_top_right
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_U_C1.*
       :align: center
       :width: 60%
       :alt: Contraction for C1 during top absorption with contraction order shown.

       Contraction for C1 during top absorption. Here C1 is the tensor with
       structure factor phase included which is grown by a normal T4 tensor.

    \\
    """

    ctmrg_absorption_top_C1_phase_2: Definition = {
        "tensors": [["T4_phase", "C1"], "projector_top_right"],
        "network": [
            [
                (-1, 4, 3, 1),  # T4_phase
                (1, 2),  # C1
            ],
            (2, 3, 4, -2),  # projector_top_right
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_U_C1.*
       :align: center
       :width: 60%
       :alt: Contraction for C1 during top absorption with contraction order shown.

       Contraction for C1 during top absorption. Here C1 is the tensor with
       structure factor phase not included yet which is grown by a T4 tensor
       which includes a phase already.

    \\
    """

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
    """
    .. figure:: /images/CTMRG_Absorption_U_T1.*
       :align: center
       :width: 60%
       :alt: Contraction for T1 during top absorption with contraction order shown.

       Contraction for T1 during top absorption.

    \\
    """

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

    ctmrg_absorption_top_T1_large_d_phase_1: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T1_phase"],
            "projector_top_left",
            "projector_top_right",
        ],
        "network": [
            [
                (4, -2, 2, 8, 3),  # tensor
                (6, -3, 2, 9, 5),  # tensor_conj
                (1, 3, 5, 7),  # T1_phase
            ],
            (-1, 1, 4, 6),  # projector_top_left
            (7, 8, 9, -4),  # projector_top_right
        ],
    }

    ctmrg_absorption_top_T1_large_d_phase_2: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T1"],
            "projector_top_left",
            "projector_top_right",
            "structure_factor_gate",
        ],
        "network": [
            [
                (5, -2, 2, 9, 4),  # tensor
                (7, -3, 3, 10, 6),  # tensor_conj
                (1, 4, 6, 8),  # T1
            ],
            (-1, 1, 5, 7),  # projector_top_left
            (8, 9, 10, -4),  # projector_top_right
            (2, 3),  # structure_factor_gate
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
    """
    .. figure:: /images/CTMRG_Absorption_U_C2.*
       :align: center
       :width: 60%
       :alt: Contraction for C2 during top absorption with contraction order shown.

       Contraction for C2 during top absorption.

    \\
    """

    ctmrg_absorption_top_C2_phase_1: Definition = {
        "tensors": [["C2_phase", "T2"], "projector_top_left"],
        "network": [
            [
                (2, 1),  # C2_phase
                (3, 4, -2, 1),  # T2
            ],
            (-1, 2, 3, 4),  # projector_top_left
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_U_C2.*
       :align: center
       :width: 60%
       :alt: Contraction for C2 during top absorption with contraction order shown.

       Contraction for C2 during top absorption. Here C2 is the tensor with
       structure factor phase included which is grown by a normal T2 tensor.

    \\
    """

    ctmrg_absorption_top_C2_phase_2: Definition = {
        "tensors": [["C2", "T2_phase"], "projector_top_left"],
        "network": [
            [
                (2, 1),  # C2
                (3, 4, -2, 1),  # T2_phase
            ],
            (-1, 2, 3, 4),  # projector_top_left
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_U_C2.*
       :align: center
       :width: 60%
       :alt: Contraction for C2 during top absorption with contraction order shown.

       Contraction for C2 during top absorption. Here C2 is the tensor with
       structure factor phase not included yet which is grown by a T2 tensor
       which includes a phase already.

    \\
    """

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
    """
    .. figure:: /images/CTMRG_Absorption_D_C4.*
       :align: center
       :width: 60%
       :alt: Contraction for C4 during bottom absorption with contraction order shown.

       Contraction for C4 during bottom absorption.

    \\
    """

    ctmrg_absorption_bottom_C4_phase_1: Definition = {
        "tensors": [["C4_phase", "T4"], "projector_bottom_right"],
        "network": [
            [
                (2, 1),  # C4_phase
                (1, 4, 3, -2),  # T4
            ],
            (-1, 2, 4, 3),  # projector_bottom_right
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_D_C4.*
       :align: center
       :width: 60%
       :alt: Contraction for C4 during bottom absorption with contraction order shown.

       Contraction for C4 during bottom absorption. Here C4 is the tensor with
       structure factor phase included which is grown by a normal T4 tensor.

    \\
    """

    ctmrg_absorption_bottom_C4_phase_2: Definition = {
        "tensors": [["C4", "T4_phase"], "projector_bottom_right"],
        "network": [
            [
                (2, 1),  # C4
                (1, 4, 3, -2),  # T4_phase
            ],
            (-1, 2, 4, 3),  # projector_bottom_right
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_D_C4.*
       :align: center
       :width: 60%
       :alt: Contraction for C4 during bottom absorption with contraction order shown.

       Contraction for C4 during bottom absorption. Here C4 is the tensor with
       structure factor phase not included yet which is grown by a T4 tensor
       which includes a phase already.

    \\
    """

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
    """
    .. figure:: /images/CTMRG_Absorption_D_T3.*
       :align: center
       :width: 60%
       :alt: Contraction for T3 during bottom absorption with contraction order shown.

       Contraction for T3 during bottom absorption.

    \\
    """

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

    ctmrg_absorption_bottom_T3_large_d_phase_1: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T3_phase"],
            "projector_bottom_left",
            "projector_bottom_right",
        ],
        "network": [
            [
                (4, 3, 2, 8, -4),  # tensor
                (6, 5, 2, 9, -3),  # tensor_conj
                (1, 7, 5, 3),  # T3_phase
            ],
            (1, 6, 4, -1),  # projector_bottom_left
            (-2, 7, 9, 8),  # projector_bottom_right
        ],
    }

    ctmrg_absorption_bottom_T3_large_d_phase_2: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T3"],
            "projector_bottom_left",
            "projector_bottom_right",
            "structure_factor_gate",
        ],
        "network": [
            [
                (5, 4, 2, 9, -4),  # tensor
                (7, 6, 3, 10, -3),  # tensor_conj
                (1, 8, 6, 4),  # T3
            ],
            (1, 7, 5, -1),  # projector_bottom_left
            (-2, 8, 10, 9),  # projector_bottom_right
            (2, 3),  # structure_factor_gate
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
    """
    .. figure:: /images/CTMRG_Absorption_D_C3.*
       :align: center
       :width: 60%
       :alt: Contraction for C3 during bottom absorption with contraction order shown.

       Contraction for C3 during bottom absorption.

    \\
    """

    ctmrg_absorption_bottom_C3_phase_1: Definition = {
        "tensors": [["T2", "C3_phase"], "projector_bottom_left"],
        "network": [
            [
                (3, 4, 1, -2),  # T2
                (2, 1),  # C3_phase
            ],
            (2, 4, 3, -1),  # projector_bottom_left
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_D_C3.*
       :align: center
       :width: 60%
       :alt: Contraction for C3 during bottom absorption with contraction order shown.

       Contraction for C3 during bottom absorption. Here C3 is the tensor with
       structure factor phase included which is grown by a normal T2 tensor.

    \\
    """

    ctmrg_absorption_bottom_C3_phase_2: Definition = {
        "tensors": [["T2_phase", "C3"], "projector_bottom_left"],
        "network": [
            [
                (3, 4, 1, -2),  # T2_phase
                (2, 1),  # C3
            ],
            (2, 4, 3, -1),  # projector_bottom_left
        ],
    }
    """
    .. figure:: /images/CTMRG_Absorption_D_C3.*
       :align: center
       :width: 60%
       :alt: Contraction for C3 during bottom absorption with contraction order shown.

       Contraction for C3 during bottom absorption. Here C3 is the tensor with
       structure factor phase not included yet which is grown by a T2 tensor
       which includes a phase already.

    \\
    """

    ctmrg_split_transfer_absorption_left_C1: Definition = {
        "tensors": [
            ["C1", "T1_ket", "T1_bra"],
            "projector_left_bottom_ket",
            "projector_left_bottom_bra",
        ],
        "network": [
            [
                (2, 1),  # C1
                (1, 3, 4),  # T1_ket
                (4, 6, -2),  # T1_bra
            ],
            (5, 2, 3),  # projector_left_bottom_ket
            (-1, 5, 6),  # projector_left_bottom_bra
        ],
    }

    ctmrg_split_transfer_absorption_left_T4: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T4_ket", "T4_bra"],
            "projector_left_top_ket",
            "projector_left_top_bra",
            "projector_left_bottom_ket",
            "projector_left_bottom_bra",
        ],
        "network": [
            [
                (4, 6, 10, -3, 3),  # tensor
                (9, 12, 10, -2, 8),  # tensor_conj
                (5, 4, 1),  # T4_ket
                (2, 9, 5),  # T4_bra
            ],
            (1, 3, 7),  # projector_left_top_ket
            (7, 8, -4),  # projector_left_top_bra
            (11, 2, 6),  # projector_left_bottom_ket
            (-1, 11, 12),  # projector_left_bottom_bra
        ],
    }

    ctmrg_split_transfer_absorption_left_T4_ket: Definition = {
        "tensors": [
            ["tensor", "T4_ket"],
            "projector_left_top_ket",
            "projector_left_top_bra",
            "projector_left_bottom_phys_ket",
            "projector_left_bottom_phys_bra",
        ],
        "network": [
            [
                (3, 5, 7, -2, 2),  # tensor
                (4, 3, 1),  # T4_ket
            ],
            (1, 2, 8),  # projector_left_top_ket
            (8, 9, -3),  # projector_left_top_bra
            (6, 4, 5),  # projector_left_bottom_phys_ket
            (-1, 6, 9, 7),  # projector_left_bottom_phys_bra
        ],
    }

    ctmrg_split_transfer_absorption_left_T4_bra: Definition = {
        "tensors": [
            ["tensor_conj", "T4_bra"],
            "projector_left_top_phys_ket",
            "projector_left_top_phys_bra",
            "projector_left_bottom_ket",
            "projector_left_bottom_bra",
        ],
        "network": [
            [
                (6, 9, 7, -2, 5),  # tensor_conj
                (2, 6, 1),  # T4_bra
            ],
            (1, 3, 4),  # projector_left_top_phys_ket
            (4, 5, 7, -3),  # projector_left_top_phys_bra
            (8, 2, 3),  # projector_left_bottom_ket
            (-1, 8, 9),  # projector_left_bottom_bra
        ],
    }

    ctmrg_split_transfer_absorption_left_C4: Definition = {
        "tensors": [
            ["T3_ket", "T3_bra", "C4"],
            "projector_left_top_ket",
            "projector_left_top_bra",
        ],
        "network": [
            [
                (1, 3, 4),  # T3_ket
                (4, 6, -1),  # T3_bra
                (1, 2),  # C4
            ],
            (2, 3, 5),  # projector_left_top_ket
            (5, 6, -2),  # projector_left_top_bra
        ],
    }

    ctmrg_split_transfer_absorption_right_C2: Definition = {
        "tensors": [
            ["T1_ket", "T1_bra", "C2"],
            "projector_right_bottom_ket",
            "projector_right_bottom_bra",
        ],
        "network": [
            [
                (-1, 6, 4),  # T1_ket
                (4, 3, 1),  # T1_bra
                (1, 2),  # C2
            ],
            (5, 6, -2),  # projector_right_bottom_ket
            (2, 3, 5),  # projector_right_bottom_bra
        ],
    }

    ctmrg_split_transfer_absorption_right_T2: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T2_ket", "T2_bra"],
            "projector_right_top_ket",
            "projector_right_top_bra",
            "projector_right_bottom_ket",
            "projector_right_bottom_bra",
        ],
        "network": [
            [
                (-3, 8, 10, 9, 12),  # tensor
                (-2, 3, 10, 4, 6),  # tensor_conj
                (5, 9, 2),  # T2_ket
                (1, 4, 5),  # T2_bra
            ],
            (-4, 11, 12),  # projector_right_top_ket
            (11, 2, 6),  # projector_right_top_bra
            (7, 8, -1),  # projector_right_bottom_ket
            (1, 3, 7),  # projector_right_bottom_bra
        ],
    }

    ctmrg_split_transfer_absorption_right_T2_ket: Definition = {
        "tensors": [
            ["tensor", "T2_ket"],
            "projector_right_top_ket",
            "projector_right_top_bra",
            "projector_right_bottom_phys_ket",
            "projector_right_bottom_phys_bra",
        ],
        "network": [
            [
                (-2, 5, 7, 6, 9),  # tensor
                (1, 6, 2),  # T2_ket
            ],
            (-3, 8, 9),  # projector_right_top_ket
            (8, 2, 3),  # projector_right_top_bra
            (4, 5, 7, -1),  # projector_right_bottom_phys_ket
            (1, 3, 4),  # projector_right_bottom_phys_bra
        ],
    }

    ctmrg_split_transfer_absorption_right_T2_bra: Definition = {
        "tensors": [
            ["tensor_conj", "T2_bra"],
            "projector_right_top_phys_ket",
            "projector_right_top_phys_bra",
            "projector_right_bottom_ket",
            "projector_right_bottom_bra",
        ],
        "network": [
            [
                (-2, 2, 7, 3, 5),  # tensor_conj
                (1, 3, 4),  # T2_bra
            ],
            (-3, 6, 9, 7),  # projector_right_top_phys_ket
            (6, 4, 5),  # projector_right_top_phys_bra
            (8, 9, -1),  # projector_right_bottom_ket
            (1, 2, 8),  # projector_right_bottom_bra
        ],
    }

    ctmrg_split_transfer_absorption_right_C3: Definition = {
        "tensors": [
            ["C3", "T3_ket", "T3_bra"],
            "projector_right_top_ket",
            "projector_right_top_bra",
        ],
        "network": [
            [
                (1, 2),  # C3
                (-1, 6, 4),  # T3_ket
                (4, 3, 1),  # T3_bra
            ],
            (-2, 5, 6),  # projector_right_top_ket
            (5, 2, 3),  # projector_right_top_bra
        ],
    }

    ctmrg_split_transfer_absorption_top_C1: Definition = {
        "tensors": [
            ["T4_ket", "T4_bra", "C1"],
            "projector_top_right_ket",
            "projector_top_right_bra",
        ],
        "network": [
            [
                (4, 3, 1),  # T4_ket
                (-1, 6, 4),  # T4_bra
                (1, 2),  # C1
            ],
            (2, 3, 5),  # projector_top_right_ket
            (5, 6, -2),  # projector_top_right_bra
        ],
    }

    ctmrg_split_transfer_absorption_top_T1: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T1_ket", "T1_bra"],
            "projector_top_left_ket",
            "projector_top_left_bra",
            "projector_top_right_ket",
            "projector_top_right_bra",
        ],
        "network": [
            [
                (3, -2, 10, 6, 4),  # tensor
                (8, -3, 10, 12, 9),  # tensor_conj
                (1, 4, 5),  # T1_ket
                (5, 9, 2),  # T1_bra
            ],
            (7, 1, 3),  # projector_top_left_ket
            (-1, 7, 8),  # projector_top_left_bra
            (2, 6, 11),  # projector_top_right_ket
            (11, 12, -4),  # projector_top_right_bra
        ],
    }

    ctmrg_split_transfer_absorption_top_T1_ket: Definition = {
        "tensors": [
            ["tensor", "T1_ket"],
            "projector_top_left_ket",
            "projector_top_left_bra",
            "projector_top_right_phys_ket",
            "projector_top_right_phys_bra",
        ],
        "network": [
            [
                (2, -2, 7, 5, 3),  # tensor
                (1, 3, 4),  # T1_ket
            ],
            (8, 1, 2),  # projector_top_left_ket
            (-1, 8, 9),  # projector_top_left_bra
            (4, 5, 6),  # projector_top_right_phys_ket
            (6, 9, 7, -3),  # projector_top_right_phys_bra
        ],
    }

    ctmrg_split_transfer_absorption_top_T1_bra: Definition = {
        "tensors": [
            ["tensor_conj", "T1_bra"],
            "projector_top_left_phys_ket",
            "projector_top_left_phys_bra",
            "projector_top_right_ket",
            "projector_top_right_bra",
        ],
        "network": [
            [
                (5, -2, 7, 9, 6),  # tensor_conj
                (1, 6, 2),  # T1_bra
            ],
            (4, 1, 3),  # projector_top_left_phys_ket
            (-1, 4, 5, 7),  # projector_top_left_phys_bra
            (2, 3, 8),  # projector_top_right_ket
            (8, 9, -3),  # projector_top_right_bra
        ],
    }

    ctmrg_split_transfer_absorption_top_C2: Definition = {
        "tensors": [
            ["C2", "T2_ket", "T2_bra"],
            "projector_top_left_ket",
            "projector_top_left_bra",
        ],
        "network": [
            [
                (2, 1),  # C2
                (4, 3, 1),  # T2_ket
                (-2, 6, 4),  # T2_bra
            ],
            (5, 2, 3),  # projector_top_left_ket
            (-1, 5, 6),  # projector_top_left_bra
        ],
    }

    ctmrg_split_transfer_absorption_bottom_C4: Definition = {
        "tensors": [
            ["C4", "T4_ket", "T4_bra"],
            "projector_bottom_right_ket",
            "projector_bottom_right_bra",
        ],
        "network": [
            [
                (2, 1),  # C4
                (4, 6, -2),  # T4_ket
                (1, 3, 4),  # T4_bra
            ],
            (-1, 5, 6),  # projector_bottom_right_ket
            (5, 2, 3),  # projector_bottom_right_bra
        ],
    }

    ctmrg_split_transfer_absorption_bottom_T3: Definition = {
        "tensors": [
            ["tensor", "tensor_conj", "T3_ket", "T3_bra"],
            "projector_bottom_left_ket",
            "projector_bottom_left_bra",
            "projector_bottom_right_ket",
            "projector_bottom_right_bra",
        ],
        "network": [
            [
                (12, 9, 10, 8, -2),  # tensor
                (6, 4, 10, 3, -3),  # tensor_conj
                (2, 9, 5),  # T3_ket
                (5, 4, 1),  # T3_bra
            ],
            (11, 12, -1),  # projector_bottom_left_ket
            (2, 6, 11),  # projector_bottom_left_bra
            (-4, 7, 8),  # projector_bottom_right_ket
            (7, 1, 3),  # projector_bottom_right_bra
        ],
    }

    ctmrg_split_transfer_absorption_bottom_T3_ket: Definition = {
        "tensors": [
            ["tensor", "T3_ket"],
            "projector_bottom_left_ket",
            "projector_bottom_left_bra",
            "projector_bottom_right_phys_ket",
            "projector_bottom_right_phys_bra",
        ],
        "network": [
            [
                (9, 6, 7, 5, -2),  # tensor
                (2, 6, 1),  # T3_ket
            ],
            (8, 9, -1),  # projector_bottom_left_ket
            (2, 3, 8),  # projector_bottom_left_bra
            (-3, 4, 5, 7),  # projector_bottom_right_phys_ket
            (4, 1, 3),  # projector_bottom_right_phys_bra
        ],
    }

    ctmrg_split_transfer_absorption_bottom_T3_bra: Definition = {
        "tensors": [
            ["tensor_conj", "T3_bra"],
            "projector_bottom_left_phys_ket",
            "projector_bottom_left_phys_bra",
            "projector_bottom_right_ket",
            "projector_bottom_right_bra",
        ],
        "network": [
            [
                (5, 3, 7, 2, -2),  # tensor_conj
                (4, 3, 1),  # T3_bra
            ],
            (6, 9, 7, -1),  # projector_bottom_left_phys_ket
            (4, 5, 6),  # projector_bottom_left_phys_bra
            (-3, 8, 9),  # projector_bottom_right_ket
            (8, 1, 2),  # projector_bottom_right_bra
        ],
    }

    ctmrg_split_transfer_absorption_bottom_C3: Definition = {
        "tensors": [
            ["T2_ket", "T2_bra", "C3"],
            "projector_bottom_left_ket",
            "projector_bottom_left_bra",
        ],
        "network": [
            [
                (4, 6, -2),  # T2_ket
                (1, 3, 4),  # T2_bra
                (2, 1),  # C3
            ],
            (5, 6, -1),  # projector_bottom_left_ket
            (2, 3, 5),  # projector_bottom_left_bra
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

    unitcell_bond_dim_change_left: Definition = {
        "tensors": [["tensor", "tensor_conj", "C1", "T1", "T3", "C4", "T4"]],
        "network": [
            [
                (5, 9, 11, -4, 4),  # tensor
                (7, 10, 11, -3, 6),  # tensor_conj
                (1, 3),  # C1
                (3, 4, 6, -1),  # T1
                (2, -2, 10, 9),  # T3
                (2, 8),  # C4
                (8, 7, 5, 1),  # T4
            ]
        ],
    }

    unitcell_bond_dim_change_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "C2", "T2", "T3", "C3"]],
        "network": [
            [
                (-1, 8, 11, 5, 4),  # tensor
                (-4, 9, 11, 7, 6),  # tensor_conj
                (-2, 4, 6, 3),  # T1
                (3, 1),  # C2
                (5, 7, 10, 1),  # T2
                (-3, 2, 9, 8),  # T3
                (2, 10),  # C3
            ]
        ],
    }

    unitcell_bond_dim_change_top: Definition = {
        "tensors": [["tensor", "tensor_conj", "C1", "T1", "C2", "T2", "T4"]],
        "network": [
            [
                (8, -4, 11, 4, 5),  # tensor
                (9, -3, 11, 6, 7),  # tensor_conj
                (2, 10),  # C1
                (10, 5, 7, 1),  # T1
                (1, 3),  # C2
                (4, 6, -2, 3),  # T2
                (-1, 9, 8, 2),  # T4
            ]
        ],
    }

    unitcell_bond_dim_change_bottom: Definition = {
        "tensors": [["tensor", "tensor_conj", "T2", "C3", "T3", "C4", "T4"]],
        "network": [
            [
                (4, 5, 11, 8, -1),  # tensor
                (6, 7, 11, 9, -4),  # tensor_conj
                (8, 9, 2, -3),  # T2
                (10, 2),  # C3
                (1, 10, 7, 5),  # T3
                (1, 3),  # C4
                (3, 6, 4, -2),  # T4
            ]
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

    kagome_pess_mapping_upper_triangle: Definition = {
        "tensors": ["up", "down", "site_1", "site_2", "site_3"],
        "network": [
            [1, 2, 3],  # up
            [-1, 4, -2],  # down
            [-7, -3, 1],  # site_1
            [4, -4, 2],  # site_2
            [-6, -5, 3],  # site_3
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

    corrlength_transfer_right: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "T3"]],
        "network": [
            [
                (-2, 4, 1, -6, 2),  # tensor
                (-3, 5, 1, -7, 3),  # tensor_conj
                (-1, 2, 3, -5),  # T1
                (-4, -8, 5, 4),  # T3
            ]
        ],
    }

    corrlength_vector_left: Definition = {
        "tensors": [["C1", "T4", "C4"]],
        "network": [
            [
                (1, -1),  # C1
                (2, -3, -2, 1),  # T4
                (-4, 2),  # C4
            ]
        ],
    }

    corrlength_vector_right: Definition = {
        "tensors": [["C2", "T2", "C3"]],
        "network": [
            [
                (-1, 1),  # C2
                (-2, -3, 2, 1),  # T2
                (-4, 2),  # C3
            ]
        ],
    }

    corrlength_transfer_bottom: Definition = {
        "tensors": [["tensor", "tensor_conj", "T4", "T2"]],
        "network": [
            [
                (2, -6, 1, 4, -2),  # tensor
                (3, -7, 1, 5, -3),  # tensor_conj
                (-5, 3, 2, -1),  # T4
                (4, 5, -8, -4),  # T2
            ]
        ],
    }

    corrlength_vector_top: Definition = {
        "tensors": [["C1", "T1", "C2"]],
        "network": [
            [
                (-1, 1),  # C1
                (1, -2, -3, 2),  # T1
                (2, -4),  # C2
            ]
        ],
    }

    corrlength_vector_bottom: Definition = {
        "tensors": [["C4", "T3", "C3"]],
        "network": [
            [
                (1, -1),  # C4
                (1, 2, -3, -2),  # T3
                (2, -4),  # C3
            ]
        ],
    }

    corrlength_absorb_one_column: Definition = {
        "tensors": [["tensor", "tensor_conj", "T1", "T3"], "vec"],
        "network": [
            [
                (3, 8, 2, -2, 4),  # tensor
                (5, 9, 2, -3, 6),  # tensor_conj
                (1, 4, 6, -1),  # T1
                (7, -4, 9, 8),  # T3
            ],
            [1, 3, 5, 7],  # vec
        ],
    }

    corrlength_absorb_one_row: Definition = {
        "tensors": [["tensor", "tensor_conj", "T4", "T2"], "vec"],
        "network": [
            [
                (4, -2, 2, 8, 3),  # tensor
                (6, -3, 2, 9, 5),  # tensor_conj
                (-1, 6, 4, 1),  # T4
                (8, 9, -4, 7),  # T3
            ],
            [1, 3, 5, 7],  # vec
        ],
    }


Definitions._prepare_defs()
