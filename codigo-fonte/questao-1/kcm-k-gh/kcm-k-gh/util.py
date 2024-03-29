import copy as cp
import random
import numpy as np
from scipy.spatial.distance import euclidean
import math
from itertools import permutations


def take_random_elems(list, n):
    """
    Takes ``n`` random elements from ``list``.
    """
    shuffle_list = copy(list)
    random.shuffle(shuffle_list)
    return shuffle_list[:n]


def remove_none_and_zero_elems(list):
    """
    Remove ``None`` and ``0`` elements from ``list``.

    Examples:
        >>> remove_none_and_zero_elems([0.0, 1, 2, None, 0, None, 5])
        [1, 2, 5]

        >>> remove_none_and_zero_elems([0.0, None, 0, None])
        []
    """
    return [x for x in list if (x is not None) & (x != 0)]


def flatten_matrix(matrix):
    """
    Flatten the given matrix (n-d array) into an 1-d array.

    Examples:
        >>> flatten_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return np.array(matrix).flatten().tolist()


def calculate_distance_matrix(elements):
    """
    Calculate the matrix of distances between all ``data`` elements.

    Returns:
        ``n x n`` matrix, once ``n`` is the number of elements in ``list``.

    Examples:
        >>> calculate_distance_matrix([1, 2, 3])
        [[0.0, 1.0, 2.0], [None, 0.0, 1.0], [None, None, 0.0]]
    """
    size = len(elements)
    distances = list()
    range_i = range(size)

    for i in range_i:
        range_j = range(i + 1, size)

        for j in range_j:
            u = elements[i].data
            v = elements[j].data
            dist = euclidean(u, v)
            distances.append(dist)

    return distances


def parse_float(list):
    """
    Converts the items in a list to float data type.

    Examples:
        >>> parse_float(["1.123", "3", "0.789"])
        [1.123, 3.0, 0.789]
    """
    return np.array(list).astype('float').tolist()


def remove_cols(list, cols, axis=0):
    return np.delete(list, cols, axis).tolist()


def infinity():
    """
    Return the infinity number representation.

    Examples:
        >>> infinity()
        inf
    """
    return np.inf


def isnan(number):
    return math.isnan(number)


def copy(list):
    return cp.copy(list)


def range_list(start, end):
    return list(range(start, end))


def isempty(list):
    return len(list) == 0


def strnumber(number):
    return str(number).replace('.', ',')


if __name__ == '__main__':
    import doctest
    doctest.testmod()
