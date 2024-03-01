import numpy as np
from hierarchical_clustering_naive import hierarchical_clustering_naive
from hierarchical_clustering_mst import hierarchical_clustering_MST


def hierarchical_clustering(
    data: np.ndarray, metric: str = "euclidean", type: str = "naive"
) -> np.ndarray:
    """facade function for hierarchical clustering.
    calls the appropriate function based on the type parameter

    Parameters
    ----------
    data : np.ndarray
        the data to cluster
    metric : str, optional
        distance calculation metrix ,by default "euclidean"
    type : str, optional
        used type to perform clustering, by default "naive"

    Returns
    -------
    np.ndarray
        the linkage matrix
    """
    match (metric):
        case "euclidean":
            order = 2
        case "manhattan":
            order = 1
        case _:
            raise ValueError("metric must be either euclidean or manhattan")

    match (type):
        case "naive":
            return hierarchical_clustering_naive(data, order)
        case "MST":
            return hierarchical_clustering_MST(data, order)
        case _:
            raise ValueError("type must be either 'naive' or ...")
