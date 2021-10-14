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
