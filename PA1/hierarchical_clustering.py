import numpy as np


def calculate_distance_matrix(data, ord=2):
    """calculates the distance matrix for the given data, with the given norm

    Parameters
    ----------
    data : np.ndarray
        the data to calculate the distance matrix for
    ord : int, optional
        order of the norm to use. 1 = Manhattan, 2 = Euclidean, by default 2

    Returns
    -------
    np.ndarray
        the distance matrix
    """

    distance_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            distance = np.linalg.norm(data[i] - data[j], ord=ord)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def find_minimum(distance_matrix):
    """finds the minimum distance in the distance matrix

    Parameters
    ----------
    distance_matrix : np.ndarray
        the distance matrix to search

    Returns
    -------
    tuple
        the indices of the minimum distance
    """

    min_distance = np.inf
    min_indices = None
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[0]):
            if distance_matrix[i, j] < min_distance:
                min_distance = distance_matrix[i, j]
                min_indices = (i, j)

    return min_indices
