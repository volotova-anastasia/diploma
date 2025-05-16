import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import load_users, load_transactions, merge_data
from utils import calculate_silhouette
import time
import psutil
import os


def train_and_save_kmeans(users_path, transactions_path, n_clusters, model_path):
    users = load_users(users_path)
    transactions = load_transactions(transactions_path)
    data = merge_data(users, transactions)

    X = data[['amount', 'age', 'avg_monthly_income']].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    start_time = time.time()
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    elapsed = time.time() - start_time

    silhouette = calculate_silhouette(X_scaled, labels)

    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # MB

    print(f"KMeans trained with n_clusters={n_clusters}")
    print(f"Silhouette score: {silhouette:.4f}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Memory usage: {mem_usage:.2f} MB")

    joblib.dump({'model': model, 'scaler': scaler}, model_path)

    return silhouette, elapsed, mem_usage


if __name__ == "__main__":
    train_and_save_kmeans('path/to/users.csv', 'path/to/transactions.csv', 6, 'kmeans_model.joblib')
