"""Data loading and preprocessing for MovieLens."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple
import urllib.request
import zipfile
import os


def download_movielens_100k(data_dir: str = "./data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extract_path = os.path.join(data_dir, "ml-100k")
    
    if not os.path.exists(extract_path):
        print("Downloading MovieLens 100K...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(zip_path)
    
    return extract_path


def load_ratings(data_path: str) -> pd.DataFrame:
    ratings_file = os.path.join(data_path, "u.data")
    return pd.read_csv(
        ratings_file,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )


def load_items(data_path: str) -> pd.DataFrame:
    items_file = os.path.join(data_path, "u.item")
    return pd.read_csv(
        items_file,
        sep='|',
        encoding='latin-1',
        header=None,
        usecols=[0, 1],
        names=['item_id', 'title']
    )


def create_sparse_matrix(
    df: pd.DataFrame,
    implicit: bool = False
) -> Tuple[csr_matrix, dict, dict]:
    """
    Convert ratings DataFrame to sparse CSR matrix.
    
    Returns rating matrix and mappings from original IDs to matrix indices.
    """
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_map = {iid: idx for idx, iid in enumerate(unique_items)}
    
    row_indices = df['user_id'].map(user_map).values
    col_indices = df['item_id'].map(item_map).values
    
    if implicit:
        data = np.ones(len(df))
    else:
        data = df['rating'].values.astype(np.float64)
    
    R = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_items))
    )
    
    return R, user_map, item_map


def train_test_split_by_user(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings per user to preserve user structure in test set."""
    rng = np.random.RandomState(random_state)
    
    train_data = []
    test_data = []
    
    for user_id in df['user_id'].unique():
        user_ratings = df[df['user_id'] == user_id]
        
        if len(user_ratings) < 5:
            train_data.append(user_ratings)
            continue
        
        n_test = max(1, int(len(user_ratings) * test_ratio))
        test_indices = rng.choice(user_ratings.index, n_test, replace=False)
        
        train_data.append(user_ratings.drop(test_indices))
        test_data.append(user_ratings.loc[test_indices])
    
    return pd.concat(train_data), pd.concat(test_data)
