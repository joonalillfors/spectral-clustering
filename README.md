# Description

This repository contains three graph partitioning algorithms using spectral clustering: 1) basic spectral clustering algorithm using normalized Laplacian matrix, 2) basic spectral clustering algorithm using unnormalized Laplacian matrix without the first eigenvector and 3) spectral clustering algorithm with the Fiedler vector.

The algorithms partition graphs described in text files with each edge connecting two vertices, e.g. "0 1" could be one edge connecting vertices 0 and 1. The algorithms output a similar text file that describes in which cluster each vertex is placed to, e.g. "0 0" would mean that vertex 0 is placed in cluster 0. A goodness of a clustering is measured using ratio-cut. Some sample graphs are provided in ```code/graphs_processed```.

# How to run

In code/ directory, run ```python partition.py {graphName}```, where graphName is ca-GrQc or Oregon-1. The code will print out the objective function scores and cluster sizes. Resulting clusterings can be found in results/ directory for each of the different algorithms.
