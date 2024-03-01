import numpy as np
from distance import calculate_distance_matrix


def prim_algorithm(distance_matrix: np.ndarray) -> list[tuple[int, int, int]]:
    """performs the prim algorithm on the given distance matrix to find the minimum spanning tree

    Parameters
    ----------
    distance_matrix : np.ndarray
        the distance matrix to use

    Returns
    -------
    list[tuple[int,int,int]]
        the edges of the minimum spanning tree as a list of tuples
        the first two elements of the tuple are the indices of the connected nodes
        the third element is the weight of the edge
    """
    vertices = np.arange(distance_matrix.shape[0])
    costs = np.full(distance_matrix.shape[0], np.inf)
    previous = np.full(distance_matrix.shape[0], -1)
    costs[0] = 0

    while len(vertices) > 1:
        min_index = np.argmin(costs[vertices])
        min_node = vertices[min_index]

        vertices = np.delete(vertices, min_index)

        # update the costs and previous node for all vertices
        # if the distance to the new node is smaller than the current cost
        indices_to_update = np.where(
            distance_matrix[min_node, vertices] < costs[vertices]
        )[0]
        nodes_to_update = vertices[indices_to_update]

        costs[nodes_to_update] = distance_matrix[min_node][nodes_to_update]
        previous[nodes_to_update] = min_node

    return list(zip(previous[1:], np.arange(1, len(previous)), costs[1:]))


def hierarchical_clustering_MST(data: np.ndarray, order: int = 2) -> np.ndarray:
    """performs hierarchical clustering on the given data
    using the MST implementation

    Parameters
    ----------
    data : np.ndarray
        the data to cluster
    order : int, optional
        the order of the norm to use, by default 2

    Returns
    -------
    np.ndarray
        the linkage matrix
    """

    number_data_points = data.shape[0]
    distance_matrix = calculate_distance_matrix(data, ord=order)
    np.fill_diagonal(distance_matrix, np.inf)

    edges = prim_algorithm(distance_matrix)

    sorted_edges = sorted(edges, key=lambda x: x[2])
    cluster_mapping = np.arange(number_data_points)
    linkage_matrix = np.zeros((number_data_points - 1, 4))

    for iteration, (u, v, w) in enumerate(sorted_edges):

        cluster_1 = min(cluster_mapping[u], cluster_mapping[v])
        cluster_2 = max(cluster_mapping[u], cluster_mapping[v])

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

        linkage_matrix[iteration] = np.array(
            [
                cluster_1,
                cluster_2,
                w,
                cluster_1_size + cluster_2_size,
            ]
        )

        cluster_mapping[cluster_mapping == cluster_1] = number_data_points + iteration
        cluster_mapping[cluster_mapping == cluster_2] = number_data_points + iteration

    return linkage_matrix
