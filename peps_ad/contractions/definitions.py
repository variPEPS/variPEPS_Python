"""
Definitions for contractions in this module.
"""


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

    density_matrix_one_site = {
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

    density_matrix_two_sites_left = {
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

    density_matrix_two_sites_right = {
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

    density_matrix_two_sites_top = {
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

    density_matrix_two_sites_bottom = {
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

    density_matrix_four_sites_top_left = {
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

    density_matrix_four_sites_top_right = {
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

    density_matrix_four_sites_bottom_left = {
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

    density_matrix_four_sites_bottom_right = {
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

    density_matrix_three_sites_triangle_without_top_left = {
        "tensors": ["top-left", "top-right", "bottom-left", "bottom-right"],
        "network": [
            (10, 11, 12, 1, 2, 3),  # top-left
            (-1, -4, 1, 2, 3, 4, 5, 6),  # top-right
            (-2, -5, 7, 8, 9, 10, 11, 12),  # bottom-left
            (-3, -6, 4, 5, 6, 7, 8, 9),  # bottom-right
        ],
    }

    ctmrg_top_left = {
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

    ctmrg_top_right = {
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

    ctmrg_top_right_large_d = {
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

    ctmrg_bottom_left = {
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

    ctmrg_bottom_left_large_d = {
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

    ctmrg_bottom_right = {
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

    ctmrg_bottom_right_large_d = {
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

    ctmrg_absorption_left_C1 = {
        "tensors": [["C1", "T1"], "projector_left_bottom"],
        "network": [
            [
                (2, 1),  # C1
                (1, 3, 4, -2),  # T1
            ],
            (-1, 2, 3, 4),  # projector_left_bottom
        ],
    }

    ctmrg_absorption_left_T4 = {
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

    ctmrg_absorption_left_C4 = {
        "tensors": [["T3", "C4"], "projector_left_top"],
        "network": [
            [
                (1, -1, 4, 3),  # T3
                (1, 2),  # C4
            ],
            (2, 3, 4, -2),  # projector_left_top
        ],
    }

    ctmrg_absorption_right_C2 = {
        "tensors": [["T1", "C2"], "projector_right_bottom"],
        "network": [
            [
                (-1, 3, 4, 1),  # T1
                (1, 2),  # C2
            ],
            (2, 4, 3, -2),  # projector_right_bottom
        ],
    }

    ctmrg_absorption_right_T2 = {
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

    ctmrg_absorption_right_C3 = {
        "tensors": [["C3", "T3"], "projector_right_top"],
        "network": [
            [
                (1, 2),  # C3
                (-1, 1, 4, 3),  # T3
            ],
            (-2, 2, 4, 3),  # projector_right_top
        ],
    }

    ctmrg_absorption_top_C1 = {
        "tensors": [["T4", "C1"], "projector_top_right"],
        "network": [
            [
                (-1, 4, 3, 1),  # T4
                (1, 2),  # C1
            ],
            (2, 3, 4, -2),  # projector_top_right
        ],
    }

    ctmrg_absorption_top_T1 = {
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

    ctmrg_absorption_top_C2 = {
        "tensors": [["C2", "T2"], "projector_top_left"],
        "network": [
            [
                (2, 1),  # C2
                (3, 4, -2, 1),  # T2
            ],
            (-1, 2, 3, 4),  # projector_top_left
        ],
    }

    ctmrg_absorption_bottom_C4 = {
        "tensors": [["C4", "T4"], "projector_bottom_right"],
        "network": [
            [
                (2, 1),  # C4
                (1, 4, 3, -2),  # T4
            ],
            (-1, 2, 4, 3),  # projector_bottom_right
        ],
    }

    ctmrg_absorption_bottom_T3 = {
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

    ctmrg_absorption_bottom_C3 = {
        "tensors": [["T2", "C3"], "projector_bottom_left"],
        "network": [
            [
                (3, 4, 1, -2),  # T2
                (2, 1),  # C3
            ],
            (2, 4, 3, -1),  # projector_bottom_left
        ],
    }

    ctmrg_gauge_fix_T1 = {
        "tensors": [["T1"], "left_unitary", "right_unitary"],
        "network": [
            [(1, -2, -3, 2)],  # T1
            (-1, 1),  # left_unitary
            (-4, 2),  # right_unitary
        ],
    }

    ctmrg_gauge_fix_C2 = {
        "tensors": [["C2"], "left_unitary", "bottom_unitary"],
        "network": [[(1, 2)], (-1, 1), (-2, 2)],  # C2  # left_unitary  # bottom_unitary
    }

    ctmrg_gauge_fix_T2 = {
        "tensors": [["T2"], "top_unitary", "bottom_unitary"],
        "network": [
            [(-1, -2, 1, 2)],  # T2
            (-4, 2),  # top_unitary
            (-3, 1),  # bottom_unitary
        ],
    }

    ctmrg_gauge_fix_T3 = {
        "tensors": [["T3"], "left_unitary", "right_unitary"],
        "network": [
            [(1, 2, -3, -4)],  # T3
            (1, -1),  # left_unitary
            (2, -2),  # right_unitary
        ],
    }

    ctmrg_gauge_fix_C4 = {
        "tensors": [["C4"], "top_unitary", "right_unitary"],
        "network": [[(1, 2)], (2, -2), (1, -1)],  # C4  # top_unitary  # right_unitary
    }

    ctmrg_gauge_fix_T4 = {
        "tensors": [["T4"], "top_unitary", "bottom_unitary"],
        "network": [
            [(1, -2, -3, 2)],  # T4
            (2, -4),  # top_unitary
            (1, -1),  # bottom_unitary
        ],
    }

    kagome_pess3_single_site_mapping = {
        "tensors": ["up_simplex", "down_simplex", "site1", "site2", "site3"],
        "network": [
            (1, 2, 3),  # up_simplex
            (-1, -7, 4),  # down_simplex
            (4, -3, 1),  # site1
            (-2, -4, 2),  # site2
            (-6, -5, 3),  # site3
        ],
    }

    kagome_downward_triangle_top_right = {
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

    kagome_downward_triangle_bottom_left = {
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

    kagome_downward_triangle_bottom_right = {
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
