"""Train and evaluate ALS on MovieLens 100K."""

from src import (
    ALS, ALSConfig,
    download_movielens_100k,
    load_ratings,
    load_items,
    create_sparse_matrix,
    train_test_split_by_user,
    compute_rmse_sparse,
    evaluate_ranking
)


def main():
    # Load data
    data_path = download_movielens_100k()
    df = load_ratings(data_path)
    items_df = load_items(data_path)
    
    print(f"Loaded {len(df):,} ratings")
    
    # Train/test split
    train_df, test_df = train_test_split_by_user(df, test_ratio=0.2)
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Create sparse matrices
    R_train, user_map, item_map = create_sparse_matrix(train_df)
    R_test, _, _ = create_sparse_matrix(test_df)
    
    print(f"Matrix shape: {R_train.shape}")
    print(f"Sparsity: {100 * (1 - R_train.nnz / (R_train.shape[0] * R_train.shape[1])):.1f}%")
    
    # Train model
    config = ALSConfig(
        n_factors=64,
        regularisation=10.0,
        n_iterations=15
    )
    
    model = ALS(config)
    model.fit(R_train)
    
    # Evaluate RMSE
    test_rmse = compute_rmse_sparse(model.user_factors, model.item_factors, R_test)
    print(f"\nTest RMSE: {test_rmse:.4f}")
    
    # Evaluate ranking metrics
    results = evaluate_ranking(model, R_train, R_test)
    
    print("\nRanking Metrics:")
    for k, metrics in results.items():
        print(f"  K={k:2d} | Hit Rate: {metrics['hit_rate']:.3f} | NDCG: {metrics['ndcg']:.3f}")
    
    # Show recommendations for user 0
    print("\nTop 10 for user 0:")
    reverse_item_map = {v: k for k, v in item_map.items()}
    
    recommendations = model.recommend(0, n=10, exclude_seen=R_train)
    for rank, item_idx in enumerate(recommendations, 1):
        original_id = reverse_item_map[item_idx]
        title = items_df[items_df['item_id'] == original_id]['title'].values[0]
        score = model.predict(0, item_idx)
        print(f"  {rank:2d}. {title[:50]:50s} ({score:.2f})")
    
    # Save model
    model.save("./data/als_model.npz")
    print("\nModel saved to ./data/als_model.npz")


if __name__ == "__main__":
    main()
