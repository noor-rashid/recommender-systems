"""Alternating Least Squares for collaborative filtering."""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from joblib import Parallel, delayed
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class ALSConfig:
    n_factors: int = 64
    regularisation: float = 0.1
    n_iterations: int = 15
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True


class ALS:
    """
    Sparse ALS with parallel updates.
    
    Optimisations:
    - CSR/CSC sparse storage
    - Pre-computed Gram matrices  
    - Parallel user/item updates via joblib
    """
    
    def __init__(self, config: ALSConfig):
        self.config = config
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self._is_fitted = False
        self._item_popularity: Optional[np.ndarray] = None
    
    def _init_factors(self, n_users: int, n_items: int) -> None:
        rng = np.random.RandomState(self.config.random_state)
        k = self.config.n_factors
        scale = 1.0 / np.sqrt(k)
        self.user_factors = rng.normal(0, scale, (n_users, k))
        self.item_factors = rng.normal(0, scale, (n_items, k))
    
    def _compute_gram_matrix(self, factors: np.ndarray) -> np.ndarray:
        VtV = factors.T @ factors
        reg_matrix = self.config.regularisation * np.eye(self.config.n_factors)
        return VtV + reg_matrix
    
    def _update_single_user(
        self,
        user_idx: int,
        R_csr: csr_matrix,
        V: np.ndarray,
        VtV_reg: np.ndarray
    ) -> np.ndarray:
        start = R_csr.indptr[user_idx]
        end = R_csr.indptr[user_idx + 1]
        
        if start == end:
            return np.zeros(self.config.n_factors)
        
        rated_items = R_csr.indices[start:end]
        ratings = R_csr.data[start:end]
        
        V_rated = V[rated_items]
        rhs = V_rated.T @ ratings
        
        return np.linalg.solve(VtV_reg, rhs)
    
    def _update_users_parallel(
        self,
        R_csr: csr_matrix,
        V: np.ndarray
    ) -> np.ndarray:
        n_users = R_csr.shape[0]
        VtV_reg = self._compute_gram_matrix(V)
        
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._update_single_user)(i, R_csr, V, VtV_reg)
            for i in range(n_users)
        )
        
        return np.array(results)
    
    def _update_single_item(
        self,
        item_idx: int,
        R_csc: csc_matrix,
        U: np.ndarray,
        UtU_reg: np.ndarray
    ) -> np.ndarray:
        start = R_csc.indptr[item_idx]
        end = R_csc.indptr[item_idx + 1]
        
        if start == end:
            return np.zeros(self.config.n_factors)
        
        rated_users = R_csc.indices[start:end]
        ratings = R_csc.data[start:end]
        
        U_rated = U[rated_users]
        rhs = U_rated.T @ ratings
        
        return np.linalg.solve(UtU_reg, rhs)
    
    def _update_items_parallel(
        self,
        R_csc: csc_matrix,
        U: np.ndarray
    ) -> np.ndarray:
        n_items = R_csc.shape[1]
        UtU_reg = self._compute_gram_matrix(U)
        
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._update_single_item)(j, R_csc, U, UtU_reg)
            for j in range(n_items)
        )
        
        return np.array(results)
    
    def _compute_loss(self, R_csr: csr_matrix) -> float:
        total_sq_error = 0.0
        
        for i in range(R_csr.shape[0]):
            start = R_csr.indptr[i]
            end = R_csr.indptr[i + 1]
            
            if start == end:
                continue
            
            items = R_csr.indices[start:end]
            ratings = R_csr.data[start:end]
            predictions = self.user_factors[i] @ self.item_factors[items].T
            sq_errors = (ratings - predictions) ** 2
            total_sq_error += np.sum(sq_errors)
        
        reg_term = self.config.regularisation * (
            np.sum(self.user_factors ** 2) + 
            np.sum(self.item_factors ** 2)
        )
        
        return total_sq_error + reg_term
    
    def fit(self, R: csr_matrix) -> 'ALS':
        n_users, n_items = R.shape
        
        if self.config.verbose:
            print(f"Training ALS: {n_users} users, {n_items} items")
            print(f"Factors: {self.config.n_factors}, λ: {self.config.regularisation}")
        
        self._init_factors(n_users, n_items)
        
        R_csr = R.tocsr()
        R_csc = R.tocsc()
        self._item_popularity = np.array(R_csr.sum(axis=0)).flatten()
        
        for iteration in range(self.config.n_iterations):
            iter_start = time.time()
            
            self.user_factors = self._update_users_parallel(R_csr, self.item_factors)
            self.item_factors = self._update_items_parallel(R_csc, self.user_factors)
            
            if self.config.verbose:
                loss = self._compute_loss(R_csr)
                print(f"Iteration {iteration + 1:2d} | Loss: {loss:,.0f} | Time: {time.time() - iter_start:.2f}s")
        
        self._is_fitted = True
        return self
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])
    
    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: Optional[csr_matrix] = None
    ) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        if np.allclose(self.user_factors[user_idx], 0):
            return np.argsort(-self._item_popularity)[:n]
        
        scores = self.user_factors[user_idx] @ self.item_factors.T
        
        if exclude_seen is not None:
            seen_items = exclude_seen[user_idx].indices
            scores[seen_items] = -np.inf
        
        return np.argsort(-scores)[:n]
    
    def save(self, filepath: str) -> None:
        np.savez(
            filepath,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            item_popularity=self._item_popularity,
            n_factors=self.config.n_factors,
            regularisation=self.config.regularisation
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'ALS':
        data = np.load(filepath)
        config = ALSConfig(
            n_factors=int(data['n_factors']),
            regularisation=float(data['regularisation'])
        )
        model = cls(config)
        model.user_factors = data['user_factors']
        model.item_factors = data['item_factors']
        model._item_popularity = data['item_popularity']
        model._is_fitted = True
        return model
