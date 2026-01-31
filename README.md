# Recommender Systems from Scratch

Implementing collaborative filtering algorithms from first principles using NumPy and SciPy. The goal is not to use library black boxes but to understand the mathematics deeply enough to make informed engineering decisions when these systems scale.

## Algorithms Implemented

**Explicit ALS** (Alternating Least Squares)
Matrix factorisation with closed form updates. Optimised with sparse matrices and parallel computation via joblib.

**Implicit ALS**
Confidence weighted ALS for implicit feedback (clicks, views, purchases) following the Hu, Koren, Volinsky approach.

## Project Structure

```
src/
  als.py           # Explicit ALS implementation
  implicit_als.py  # Implicit ALS with confidence weighting
  data.py          # Data loading and preprocessing
  metrics.py       # RMSE, hit rate, NDCG evaluation

train_als.py       # Training script for MovieLens 100K
```

## Usage

```python
from src import ALS, ALSConfig, download_movielens_100k, load_ratings, create_sparse_matrix

# Load data
data_path = download_movielens_100k()
df = load_ratings(data_path)
R, user_map, item_map = create_sparse_matrix(df)

# Train
config = ALSConfig(n_factors=64, regularisation=10.0, n_iterations=15)
model = ALS(config)
model.fit(R)

# Recommend
recommendations = model.recommend(user_idx=0, n=10, exclude_seen=R)
```

## Results on MovieLens 100K

| Metric | Value |
|--------|-------|
| Test RMSE | 0.97 |
| Hit Rate at 10 | 0.65 |
| NDCG at 10 | 0.18 |

## Requirements

```
numpy
scipy
pandas
joblib
```

## Blog Posts

Detailed explanations of the implementations are available on the [project site](https://noor-rashid.github.io/noor-rashid-portfolio/projects/recommender-system/).
