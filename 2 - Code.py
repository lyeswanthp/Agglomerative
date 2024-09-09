import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(42)
X = np.random.rand(10, 2)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def find_closest_clusters(dist_matrix):
    min_dist = np.inf
    closest_pair = (0, 0)
    for i in range(len(dist_matrix)):
        for j in range(i+1, len(dist_matrix)):
            if dist_matrix[i][j] < min_dist:
                min_dist = dist_matrix[i][j]
                closest_pair = (i, j)
    return closest_pair

def agglomerative_clustering(X, n_clusters=2):
    clusters = [[i] for i in range(len(X))]
    
    dist_matrix = cdist(X, X, metric='euclidean')

    while len(clusters) > n_clusters:
        i, j = find_closest_clusters(dist_matrix)
        
        clusters[i].extend(clusters[j])
        del clusters[j]
        
        for m in range(len(clusters)):
            if m != i:  # No need to calculate distance for the same cluster
                distances = [euclidean_distance(X[p1], X[p2]) for p1 in clusters[i] for p2 in clusters[m]]
                dist_matrix[i][m] = dist_matrix[m][i] = np.min(distances)

        dist_matrix = np.delete(dist_matrix, j, axis=0)
        dist_matrix = np.delete(dist_matrix, j, axis=1)

    return clusters

clusters = agglomerative_clustering(X, n_clusters=4)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for idx, cluster in enumerate(clusters):
    cluster_points = X[cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[idx % len(colors)], label=f"Cluster {idx+1}")
    
    
plt.title('Agglomerative clustering(from scratch)')
plt.legend()
plt.show()
