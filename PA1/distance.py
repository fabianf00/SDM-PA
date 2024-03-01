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

    distance_matrix = np.zeros((len(data), len(data)))
    # calculate the distances between point at position i and all other points
    for i in range(len(data)):
        distance_matrix[i, i:] = np.linalg.norm(data[i:] - data[i], axis=1, ord=ord)
        distance_matrix[i:, i] = distance_matrix[i, i:]
    return distance_matrix
