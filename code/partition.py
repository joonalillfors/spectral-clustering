import sys
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from objective import objective

def main():
    filename = sys.argv[1]
    E = []
    adjacency = []
    nofVertices = 0
    nofEdges = 0
    k = 0
    if not filename:
        print("no filename")
        return
    try:
        f = open("graphs_processed/"+filename, "r")
        for line in f:
            if line.startswith("#"):
                meta = line.split(" ")
                nofVertices = int(meta[2])
                nofEdges = int(meta[3])
                k = int(meta[4])
                adjacency = np.zeros((nofVertices, nofVertices), dtype=int)
                print(f"Graph: {meta[1]}")
                print(f"Nodes: {meta[2]}")
                print(f"Edges: {meta[3]}")
                print(f"k: {meta[4]}")
            else:
                u, v = map(lambda x: int(x), line.split(" "))
                adjacency[u][v] = 1
                adjacency[v][u] = 1
                E.append((u, v))
    except:
        print(f"no such file {filename}")
        return

    # Calculate degrees
    degree = np.array(list(map(lambda x: np.sum(x), adjacency)), dtype=float)
    D = np.diag(np.sqrt(1 / degree))
    DD = np.diag(degree)

    # Compute clusterings
    kmeans_fiedler = fiedler(adjacency, D, k)
    spec = spectral(adjacency, D, k)
    og_spec = ogSpectral(adjacency, DD, k)

    # Calculate scores
    kmeans_fiedler_score = objective(kmeans_fiedler, E, k)
    spec_score = objective(spec, E, k)
    og_spec_score = objective(og_spec, E, k)

    print("kmeans fiedler:", kmeans_fiedler_score)
    print("spec:", spec_score)
    print("og spec:", og_spec_score)

    writeRes("fiedler", filename, nofVertices, nofEdges, k, kmeans_fiedler)
    writeRes("normalized-spectral", filename, nofVertices, nofEdges, k, spec)
    writeRes("without-first-eigvec", filename, nofVertices, nofEdges, k, og_spec)

# Spectral clustering with Fiedler vector
def fiedler(A, D, k):
    UL = D - A
    # Get eigenvalues v and eigenvectors w
    v, w = np.linalg.eig(UL)
    # Sort eigenvalues
    idx = v.argsort()[::1]
    x2 = w[idx[:2][1]].T
    fiedler = x2.reshape(-1,1)
    res = KMeans(n_clusters=k).fit_predict(fiedler)
    return res


# Basic normalized spectral clustering
def spectral(A, D, k):
    L = np.identity(A.shape[0]) - D @ A @ D
    V, eig = eigsh(L, k)
    
    U = eig
    U_rowsums = U.sum(axis=1)
    U = U / U_rowsums[:, np.newaxis]

    res = KMeans(n_clusters=k).fit_predict(U)
    return res

# Spectral clustering without the first eigenvector
def ogSpectral(A, D, k):
    L = D - A
    w, eig = eigsh(L, k+1, which="SA")
    # Drop the first eigenvector
    U = eig.T[1::].T
    res = KMeans(n_clusters=k).fit_predict(U)
    return res

def writeRes(alg, name, nofV, nofE, k, clustering):
    try:
        f = open(f"../results/{alg}/{name}", "w")
        f.write(f"# {name} {nofV} {nofE} {k}\n")
        for v, c in enumerate(clustering, start=1):
            f.write(f"{v} {c+1}\n")
        f.close()
    except Exception as e:
        print(e)
        print("write failed")

main()