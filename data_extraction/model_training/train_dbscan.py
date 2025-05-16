import joblib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from data_loader import load_users, load_transactions, merge_data
from utils import calculate_silhouette
import time
import psutil
import os

def train_and_save_dbscan(users_path, transactions_path, eps, model_path):
    users = load_users(users_path)
    transactions = load_transactions(transactions_path)
    data = merge_data(users, transactions)

    X = data[['amount', 'age', 'avg_monthly_income', 'account_age_days']].fillna(0)
    X = StandardScaler().fit_transform(X)

    start_time = time.time()
    model = DBSCAN(eps=eps)
    labels = model.fit_predict(X)
    elapsed = time.time() - start_time

    silhouette = calculate_silhouette(X, labels)
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    print(f"DBSCAN eps={eps} Silhouette={silhouette:.4f}, Time={elapsed:.2f}s, Mem={mem_usage:.2f}MB")

    joblib.dump({'model': model}, model_path)

if __name__ == "__main__":
    train_and_save_dbscan('path/to/users.csv', 'path/to/transactions.csv', 0.7, 'dbscan_model.joblib')
