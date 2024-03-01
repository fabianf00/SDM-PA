import numpy as np

from distance import calculate_distance_matrix


def find_minimum(distance_matrix: np.ndarray) -> tuple[int, int]:
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

    return np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)


def update_clusters(
    cluster_mapping: np.ndarray,
    pos_cluster_1: int,
    pos_cluster_2: int,
    new_cluster_id: int,
) -> None:
    """updates the cluster mapping to reflect the new cluster. cluster_1 will get the new id and
    cluster_2 will be marked as merged, by setting its id to -1

    Parameters
    ----------
    cluster_mapping : np.ndarray
        the current cluster mapping
    pos_cluster_1 : int
        the position in the mapping of the first cluster
    pos_cluster_2 : int
        the position int the mapping of the second cluster
    new_cluster_id : int
        the id of the new cluster
    """
    cluster_mapping[pos_cluster_1] = new_cluster_id
    cluster_mapping[pos_cluster_2] = -1


def update_distance_matrix(
    distance_matrix: np.ndarray, min_indices: tuple[int, int]
) -> None:
    """updates the distance matrix to reflect the new cluster

    Parameters
    ----------
    distance_matrix : np.ndarray
        the current distance matrix
    min_indices : tuple
        the indices of the minimum distance
    """
    minimum = np.minimum(
        distance_matrix[min_indices[0]], distance_matrix[min_indices[1]]
    )
    # update distance to the new clusters distance
    distance_matrix[min_indices[0]] = minimum
    distance_matrix[:, min_indices[0]] = minimum

    distance_matrix[min_indices[0], min_indices[0]] = np.inf

    # remove the second cluster from the distance matrix by setting its distance to infinity
    distance_matrix[min_indices[1]] = np.inf
    distance_matrix[:, min_indices[1]] = np.inf


def hierarchical_clustering_naive(data: np.ndarray, order: int = 2) -> np.ndarray:
    """performs hierarchical clustering on the given data
    using the naive implementation

    Parameters
    ----------
    data : np.ndarray
        the data to cluster
    order : int, optional
        order of the norm used to calulate the distance, by default 2

    Returns
    -------
    np.ndarray
        the linkage matrix
    """

    number_data_points = data.shape[0]
    cluster_mapping = np.arange(number_data_points)

    distance_matrix = calculate_distance_matrix(data, ord=order)
    np.fill_diagonal(distance_matrix, np.inf)
    linkage_matrix = np.zeros((number_data_points - 1, 4))

    for iteration in range(number_data_points - 1):
        min_indices = find_minimum(distance_matrix)
        min_distance = distance_matrix[min_indices]

        # get cluster ids
        cluster_1 = cluster_mapping[min_indices[0]]
        cluster_2 = cluster_mapping[min_indices[1]]

        cluster_1_size = (
            1
            if cluster_1 < number_data_points
            else linkage_matrix[cluster_1 - number_data_points, 3]
        )
        cluster_2_size = (
            1
            if cluster_2 < number_data_points
            else linkage_matrix[cluster_2 - number_data_points, 3]
        )

        # fill linkage matrix
        linkage_matrix[iteration] = np.array(
            [
                cluster_1,
                cluster_2,
                min_distance,
                cluster_1_size + cluster_2_size,
            ]
        )

        # update cluster mapping
        update_clusters(
            cluster_mapping,
            min_indices[0],
            min_indices[1],
            number_data_points + iteration,
        )

        # update distance matrix
        update_distance_matrix(distance_matrix, min_indices)

    return linkage_matrix
