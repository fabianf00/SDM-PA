# Task Description

Implement hierarchical clustering (naive and efficient implementations) and analyze the runtime and accuracy of each version. Test with different datasets and compare the results. All implementations will use single linkage.

## Task 1 Naive Implementation

First implement the naive version of hierarchical clustering. The naive version is to compute the distance between each pair of points and merge the two points with the smallest distance. Repeat this process until there is only one cluster left. Runtime complexity of this version is O(n^3).

## Task 2 Efficient Implementation

Then implement the efficient version of hierarchical clustering. Their are multiple ways to implement this version. One way is to use a priority queue to store the distances between clusters. The Runtime complexity of this version is O(n^2logn). Another way is to use find the minimum spanning tree of the graph and interpret the edges of the tree as merges . The runtime complexity of this version is O(n^2). The last version uses the nearest-neighbor chain algorithm. The runtime complexity of this version is O(n^2).

## Task 3 Analysis

Analyze the runtime and accuracy of each version. Test with different datasets and compare the results with public implementations (scipy or sklearn).
