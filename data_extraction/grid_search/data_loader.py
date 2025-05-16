import pandas as pd

def load_users(path: str) -> pd.DataFrame:
    users = pd.read_csv(path, parse_dates=['registration_date'])
    return users

def load_transactions(path: str) -> pd.DataFrame:
    transactions = pd.read_csv(path, parse_dates=['date'])
    return transactions

def merge_data(users: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    data = transactions.merge(users, on='user_id', how='left')
    return data
