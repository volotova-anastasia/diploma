from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import load_users, load_transactions, merge_data
from utils import calculate_silhouette
import numpy as np


def grid_search_kmeans(users_path, transactions_path, param_grid):
    users = load_users(users_path)
    transactions = load_transactions(transactions_path)
    data = merge_data(users, transactions)

    # Пример признаков — можно расширить
    X = data[['amount', 'age', 'avg_monthly_income']].fillna(0)
    X = StandardScaler().fit_transform(X)

    best_score = -1
    best_params = None

    for k in param_grid['n_clusters']:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        score = calculate_silhouette(X, labels)
        print(f"KMeans k={k} Silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = {'n_clusters': k}

    print(f"Best params: {best_params} with silhouette score {best_score:.4f}")
    return best_params, best_score


if __name__ == "__main__":
    param_grid = {'n_clusters': [3, 4, 5, 6, 7]}
    grid_search_kmeans('path/to/users.csv', 'path/to/transactions.csv', param_grid)
