import numpy as np

def objective(clustering, edges, k):
    clusters = np.zeros(k, dtype=int)
    for i in clustering:
        clusters[i] += 1
    outgoing = np.zeros(k, dtype=int)
    for u, v in edges:
        cu = clustering[u]
        cv = clustering[v]
        if cu != cv:
            outgoing[cu] += 1
            outgoing[cv] += 1
    sum = 0
    for i, o in enumerate(outgoing):
        sum += o / clusters[i]
    print(clusters)
    return sum
