from sklearn.metrics import silhouette_score

def calculate_silhouette(X, labels):
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    else:
        return -1  # Невозможно вычислить силуэт, если кластер один
