import joblib
import pandas as pd
from data_loader import load_users, load_transactions, merge_data
from utils import calculate_silhouette
import time
import psutil
import os


def load_model_and_predict(model_path, users_path, transactions_path):
    data = merge_data(load_users(users_path), load_transactions(transactions_path))
    X = data[['amount', 'age', 'avg_monthly_income']].fillna(0)

    model_bundle = joblib.load(model_path)
    model = model_bundle['model']
    scaler = model_bundle['scaler']

    X_scaled = scaler.transform(X)

    start_time = time.time()
    labels = model.predict(X_scaled)
    elapsed = time.time() - start_time

    silhouette = calculate_silhouette(X_scaled, labels)
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    print(f"Prediction done")
    print(f"Silhouette score: {silhouette:.4f}")
    print(f"Prediction time: {elapsed:.2f} seconds")
    print(f"Memory usage: {mem_usage:.2f} MB")

    return labels, silhouette, elapsed, mem_usage


if __name__ == "__main__":
    labels, silhouette, elapsed, mem_usage = load_model_and_predict(
        'kmeans_model.joblib',
        'path/to/users.csv',
        'path/to/transactions.csv'
    )
