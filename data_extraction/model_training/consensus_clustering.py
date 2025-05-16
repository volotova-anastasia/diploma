import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

def build_coassociation_matrix(labels_list):
    n_samples = len(labels_list[0])
    co_matrix = np.zeros((n_samples, n_samples))

    for labels in labels_list:
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    co_matrix[i, j] += 1

    co_matrix /= len(labels_list)
    return co_matrix

def consensus_spectral_clustering(labels_list, k):
    co_matrix = build_coassociation_matrix(labels_list)
    distance_matrix = 1 - co_matrix
    affinity = np.exp(-squareform(pdist(distance_matrix)))

    model = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
    consensus_labels = model.fit_predict(co_matrix)
    return consensus_labels

if __name__ == "__main__":
    dummy_labels = [[]]]
    labels = consensus_spectral_clustering(dummy_labels, k=2)
    print("Consensus Labels:", labels)
