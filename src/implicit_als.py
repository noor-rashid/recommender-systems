"""Implicit feedback ALS with confidence weighting."""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from joblib import Parallel, delayed
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class ImplicitALSConfig:
    n_factors: int = 64
    regularisation: float = 0.01
    alpha: float = 40.0
    n_iterations: int = 15
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True


class ImplicitALS:
    """
    ALS for implicit feedback with confidence weighting.
    
    Uses the Hu, Koren, Volinsky approach:
    - Preference p_ui = 1 if r_ui > 0, else 0
    - Confidence c_ui = 1 + alpha * r_ui
    - Loss weighted by confidence
    """
    
    def __init__(self, config: ImplicitALSConfig):
        self.config = config
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def _init_factors(self, n_users: int, n_items: int) -> None:
        rng = np.random.RandomState(self.config.random_state)
        k = self.config.n_factors
        scale = 1.0 / np.sqrt(k)
        self.user_factors = rng.normal(0, scale, (n_users, k))
        self.item_factors = rng.normal(0, scale, (n_items, k))
    
    def _update_single_user(
        self,
        user_idx: int,
        P: csr_matrix,
        R: csr_matrix,
        V: np.ndarray,
        VtV: np.ndarray
    ) -> np.ndarray:
        k = self.config.n_factors
        
        start = R.indptr[user_idx]
        end = R.indptr[user_idx + 1]
        
        if start == end:
            # No interactions: solve (VtV + λI) u = 0
            A = VtV + self.config.regularisation * np.eye(k)
            return np.zeros(k)
        
        interacted_items = R.indices[start:end]
        interaction_counts = R.data[start:end]
        
        # Confidence: c = 1 + alpha * r (but baseline is 1 for all items)
        # Correction: only add (c - 1) for interacted items
        delta_c = self.config.alpha * interaction_counts
        
        V_int = V[interacted_items]
        correction = V_int.T @ (delta_c[:, np.newaxis] * V_int)
        
        A = VtV + correction + self.config.regularisation * np.eye(k)
        
        # RHS: V^T @ (c * p) where p=1 for interacted, c=1+alpha*r
        c_p = 1 + self.config.alpha * interaction_counts
        b = V_int.T @ c_p
        
        return np.linalg.solve(A, b)
    
    def _update_users_parallel(
        self,
        P: csr_matrix,
        R: csr_matrix,
        V: np.ndarray
    ) -> np.ndarray:
        n_users = R.shape[0]
        VtV = V.T @ V
        
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._update_single_user)(i, P, R, V, VtV)
            for i in range(n_users)
        )
        
        return np.array(results)
    
    def _update_single_item(
        self,
        item_idx: int,
        P: csc_matrix,
        R: csc_matrix,
        U: np.ndarray,
        UtU: np.ndarray
    ) -> np.ndarray:
        k = self.config.n_factors
        
        start = R.indptr[item_idx]
        end = R.indptr[item_idx + 1]
        
        if start == end:
            A = UtU + self.config.regularisation * np.eye(k)
            return np.zeros(k)
        
        interacted_users = R.indices[start:end]
        interaction_counts = R.data[start:end]
        
        delta_c = self.config.alpha * interaction_counts
        
        U_int = U[interacted_users]
        correction = U_int.T @ (delta_c[:, np.newaxis] * U_int)
        
        A = UtU + correction + self.config.regularisation * np.eye(k)
        
        c_p = 1 + self.config.alpha * interaction_counts
        b = U_int.T @ c_p
        
        return np.linalg.solve(A, b)
    
    def _update_items_parallel(
        self,
        P: csc_matrix,
        R: csc_matrix,
        U: np.ndarray
    ) -> np.ndarray:
        n_items = R.shape[1]
        UtU = U.T @ U
        
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._update_single_item)(j, P, R, U, UtU)
            for j in range(n_items)
        )
        
        return np.array(results)
    
    def _compute_loss(self, P: csr_matrix, R: csr_matrix) -> float:
        """Weighted squared loss on all entries."""
        total_loss = 0.0
        
        for i in range(R.shape[0]):
            start = R.indptr[i]
            end = R.indptr[i + 1]
            
            predictions = self.user_factors[i] @ self.item_factors.T
            
            # Non-interacted items: c=1, p=0
            loss_zeros = np.sum(predictions ** 2)
            
            if start < end:
                items = R.indices[start:end]
                counts = R.data[start:end]
                
                # Remove contribution from interacted items (will add weighted version)
                loss_zeros -= np.sum(predictions[items] ** 2)
                
                # Interacted items: c=1+alpha*r, p=1
                c = 1 + self.config.alpha * counts
                errors = (1 - predictions[items]) ** 2
                total_loss += np.sum(c * errors)
            
            total_loss += loss_zeros
        
        reg_term = self.config.regularisation * (
            np.sum(self.user_factors ** 2) + 
            np.sum(self.item_factors ** 2)
        )
        
        return total_loss + reg_term
    
    def fit(self, R: csr_matrix) -> 'ImplicitALS':
        """
        Train implicit ALS.
        
        Args:
            R: Interaction matrix (counts or binary)
        """
        n_users, n_items = R.shape
        
        if self.config.verbose:
            print(f"Training Implicit ALS: {n_users} users, {n_items} items")
            print(f"Factors: {self.config.n_factors}, λ: {self.config.regularisation}, α: {self.config.alpha}")
        
        self._init_factors(n_users, n_items)
        
        P = (R > 0).astype(np.float64)
        P_csr = P.tocsr()
        P_csc = P.tocsc()
        R_csr = R.tocsr()
        R_csc = R.tocsc()
        
        for iteration in range(self.config.n_iterations):
            iter_start = time.time()
            
            self.user_factors = self._update_users_parallel(P_csr, R_csr, self.item_factors)
            self.item_factors = self._update_items_parallel(P_csc, R_csc, self.user_factors)
            
            if self.config.verbose:
                loss = self._compute_loss(P_csr, R_csr)
                print(f"Iteration {iteration + 1:2d} | Loss: {loss:,.0f} | Time: {time.time() - iter_start:.2f}s")
        
        self._is_fitted = True
        return self
    
    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: Optional[csr_matrix] = None
    ) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        scores = self.user_factors[user_idx] @ self.item_factors.T
        
        if exclude_seen is not None:
            start = exclude_seen.indptr[user_idx]
            end = exclude_seen.indptr[user_idx + 1]
            seen_items = exclude_seen.indices[start:end]
            scores[seen_items] = -np.inf
        
        return np.argsort(-scores)[:n]
