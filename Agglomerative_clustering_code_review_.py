import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(42) # set the seed for reproducibility
X = np.random.rand(10, 2)

def euclidean_distance(vector1, vector2):
    # Compute the Euclidean distance between two points.
    return np.linalg.norm(vector1 - vector2)

def find_closest_clusters(dist_matrix):
    """Find indices of the closest pair of clusters in a non-symmetric distance matrix."""
    # Mask the diagonal (self-distances) by setting it to infinity
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Find the index of the minimum value in the distance matrix
    min_index = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    
    return min_index

def agglomerative_clustering(X, n_clusters=2):
    clusters = [[i] for i in range(len(X))]
    
    dist_matrix = cdist(X, X, metric='euclidean')

    while len(clusters) > n_clusters:
        i, j = find_closest_clusters(dist_matrix)
        
        clusters[i].extend(clusters[j]) # Merge cluster j into cluster i
        del clusters[j]
        # Update distance matrix
        for m in range(len(clusters)):
            if m != i:  # No need to calculate distance for the same cluster
                # Update distance matrix with new distances
                distances = [euclidean_distance(X[p1], X[p2]) for p1 in clusters[i] for p2 in clusters[m]]
                dist_matrix[i][m] = dist_matrix[m][i] = np.min(distances)

        dist_matrix = np.delete(dist_matrix, j, axis=0)
        dist_matrix = np.delete(dist_matrix, j, axis=1)
        # Remove row and column for cluster j


    return clusters

clusters = agglomerative_clustering(X, n_clusters=4)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for idx, cluster in enumerate(clusters):
    cluster_points = X[cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[idx % len(colors)], label=f"Cluster {idx+1}")
    
    
plt.title('Agglomerative clustering(from scratch)')
plt.legend()
plt.show()

