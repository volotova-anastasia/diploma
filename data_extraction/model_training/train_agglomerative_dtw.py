import joblib
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from data_loader import load_transactions
from utils import calculate_silhouette
import time
import psutil
import os

def prepare_time_series(transactions):
    transactions['date'] = transactions['date'].dt.strftime('%Y-%m-%d')
    pivot = transactions.pivot_table(index='user_id', columns='date', values='amount', aggfunc='sum').fillna(0)
    series = pivot.values[:, :, np.newaxis]
    series = TimeSeriesScalerMinMax().fit_transform(series)
    return series

def train_agglomerative_dtw(transactions_path, k, model_path):
    transactions = load_transactions(transactions_path)
    series = prepare_time_series(transactions)

    start_time = time.time()
    model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42)
    labels = model.fit_predict(series)
    elapsed = time.time() - start_time

    silhouette = calculate_silhouette(series.reshape(series.shape[0], -1), labels)
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    print(f"AggloDTW k={k} Silhouette={silhouette:.4f}, Time={elapsed:.2f}s, Mem={mem_usage:.2f}MB")

    joblib.dump({'model': model}, model_path)

if __name__ == "__main__":
    train_agglomerative_dtw('path/to/transactions.csv', 3, 'agglo_dtw_model.joblib')
