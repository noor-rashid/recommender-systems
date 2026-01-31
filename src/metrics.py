"""Evaluation metrics for recommender systems."""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Set


def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return np.sqrt(np.mean((predictions - actuals) ** 2))


def compute_rmse_sparse(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    R_test: csr_matrix
) -> float:
    """Compute RMSE on sparse test matrix."""
    total_sq_error = 0.0
    n_ratings = 0
    
    for i in range(R_test.shape[0]):
        start = R_test.indptr[i]
        end = R_test.indptr[i + 1]
        
        if start == end:
            continue
        
        items = R_test.indices[start:end]
        true_ratings = R_test.data[start:end]
        predictions = user_factors[i] @ item_factors[items].T
        
        total_sq_error += np.sum((true_ratings - predictions) ** 2)
        n_ratings += len(items)
    
    return np.sqrt(total_sq_error / n_ratings)


def hit_rate_at_k(
    recommendations: np.ndarray,
    relevant_items: Set[int],
    k: int = 10
) -> float:
    """1 if any relevant item in top-k, else 0."""
    top_k = set(recommendations[:k])
    return 1.0 if len(top_k & relevant_items) > 0 else 0.0


def dcg_at_k(ranked_items: np.ndarray, relevant_items: Set[int], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(
    recommendations: np.ndarray,
    relevant_items: Set[int],
    k: int = 10
) -> float:
    """Normalised DCG at k."""
    if len(relevant_items) == 0:
        return 0.0
    
    dcg = dcg_at_k(recommendations, relevant_items, k)
    ideal_dcg = dcg_at_k(np.array(list(relevant_items)), relevant_items, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def evaluate_ranking(
    model,
    R_train: csr_matrix,
    R_test: csr_matrix,
    k_values: list = [5, 10, 20],
    relevance_threshold: float = 4.0
) -> dict:
    """Compute hit rate and NDCG across all users."""
    n_users = R_train.shape[0]
    
    results = {k: {'hit_rate': [], 'ndcg': []} for k in k_values}
    
    for user_idx in range(n_users):
        test_start = R_test.indptr[user_idx]
        test_end = R_test.indptr[user_idx + 1]
        
        if test_start == test_end:
            continue
        
        test_items = R_test.indices[test_start:test_end]
        test_ratings = R_test.data[test_start:test_end]
        relevant = set(test_items[test_ratings >= relevance_threshold])
        
        if len(relevant) == 0:
            continue
        
        recommendations = model.recommend(user_idx, n=max(k_values), exclude_seen=R_train)
        
        for k in k_values:
            results[k]['hit_rate'].append(hit_rate_at_k(recommendations, relevant, k))
            results[k]['ndcg'].append(ndcg_at_k(recommendations, relevant, k))
    
    return {
        k: {
            'hit_rate': np.mean(v['hit_rate']),
            'ndcg': np.mean(v['ndcg'])
        }
        for k, v in results.items()
    }
