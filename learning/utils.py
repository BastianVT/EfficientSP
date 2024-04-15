from enum import Enum
from itertools import zip_longest

import numpy as np


class LossFunction(Enum):
    ARC_HAMMING = 1
    NODE_HAMMING = 2


MULTI_PRIZES = False
MAX_ITER = 400
MAX_NO_IMPROV = 50
INTENSITY = 1
CLASS_DOUBLEBRIDGE = False


def arc_hamming_distance(y, pred_y):
    return len({(a, b) for a, b in zip(y, y[1:])} ^ {(a, b) for a, b in zip(pred_y, pred_y[1:])})


def node_hamming_distance(y, pred_y):
    return sum(c1 != c2 for c1, c2 in zip_longest(y, pred_y))


def build_adjacency_matrix(sequence, n):
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for i in range(len(sequence) - 1):
        adjacency_matrix[sequence[i]][sequence[i + 1]] = 1
    return adjacency_matrix


def flatten_adjacency_matrix(sequence, n):
    return build_adjacency_matrix(sequence, n).flatten()


def compute_opposite_vector(vector, magnitude=1):
    # Normalize the given vector
    norm_vector = vector / np.linalg.norm(vector)
    # Create the opposite vector
    opposite_vector = -norm_vector * magnitude
    return opposite_vector
