import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
import time
import psutil
from data_loader import load_prepared_data
from utils import measure_memory_time, save_results

X = load_prepared_data()

@measure_memory_time
def run_meanshift(X):
    model = MeanShift()
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
    return score, labels

score, labels, exec_time, mem_usage = run_meanshift(X)

print(f"Silhouette: {score:.4f}, Time: {exec_time:.2f}s, Memory: {mem_usage:.2f}MB")

save_results("meanshift", labels, score, exec_time, mem_usage)
