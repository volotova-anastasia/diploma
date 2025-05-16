import time
import psutil
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import os
import joblib

def measure_memory_time(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        end_mem = process.memory_info().rss / 1024 / 1024
        exec_time = end_time - start_time
        mem_usage = end_mem - start_mem
        if isinstance(result, tuple):
            return (*result, exec_time, mem_usage)
        else:
            return result, exec_time, mem_usage
    return wrapper

def save_results(name, labels, score, exec_time, mem_usage):
    os.makedirs("../data/cluster_labels", exist_ok=True)
    pd.DataFrame({"cluster": labels}).to_csv(f"../data/cluster_labels/{name}_labels.csv", index=False)
    with open(f"../data/cluster_labels/{name}_metrics.txt", "w") as f:
        f.write(f"silhouette={score:.4f}\ntime={exec_time:.2f}s\nmemory={mem_usage:.2f}MB")

def save_model(model, name):
    os.makedirs("../data/models", exist_ok=True)
    joblib.dump(model, f"../data/models/{name}.pkl")

def load_model(name):
    return joblib.load(f"../data/models/{name}.pkl")
