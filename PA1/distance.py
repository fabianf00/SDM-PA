import numpy as np
from numba import njit, prange


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


@njit()
def calculate_distance_matrix_numba(data, ord=2):
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


def calculate_distance_matrix_numpy(data, ord=2):
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
        distance_matrix[i] = np.linalg.norm(data - data[i], axis=1, ord=ord)
    return distance_matrix


def calculate_distance_matrix_numpy_v(data, ord=2):
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
    return np.linalg.norm(data[:, None] - data, axis=-1, ord=ord)


@njit(parallel=True)
def calculate_distance_matrix_numba_mt(data, ord=2):
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
    for i in prange(data.shape[0]):
        for j in prange(i + 1, data.shape[0]):
            distance = np.linalg.norm(data[i] - data[j], ord=ord)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def calculate_distance_matrix_numpy_v2(data, ord=2):
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
