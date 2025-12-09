"""
K-Means Clustering for Donor Segmentation
Cluster contributors into donor archetypes for KNN-based predictions
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle
import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_training_data import load_data_splits, prepare_clustering_data


def train_kmeans_model(X_train, n_clusters, batch_size=10000):
    """
    Train Mini-Batch K-Means model
    
    Args:
        X_train: Training data
        n_clusters: Number of clusters
        batch_size: Batch size for mini-batch K-Means
        
    Returns:
        Trained model and metrics
    """
    print(f"\nTraining Mini-Batch K-Means with {n_clusters} clusters...")
    
    start_time = time.time()
    
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        random_state=42,
        verbose=1,
        n_init=3
    )
    
    model.fit(X_train)
    
    train_time = time.time() - start_time
    
    # Compute metrics
    print("\nComputing cluster quality metrics...")
    
    # Get cluster assignments
    labels = model.labels_
    
    # Inertia (within-cluster sum of squares)
    inertia = model.inertia_
    
    # Silhouette score (sample if too large)
    if len(X_train) > 50000:
        sample_idx = np.random.choice(len(X_train), 50000, replace=False)
        X_sample = X_train[sample_idx]
        labels_sample = labels[sample_idx]
        silhouette = silhouette_score(X_sample, labels_sample, sample_size=10000)
    else:
        silhouette = silhouette_score(X_train, labels, sample_size=min(10000, len(X_train)))
    
    # Davies-Bouldin Index (lower is better)
    db_index = davies_bouldin_score(X_train, labels)
    
    # Cluster size statistics
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"Cluster sizes: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.0f}, std={np.std(counts):.0f}")
    
    metrics = {
        'n_clusters': n_clusters,
        'inertia': inertia,
        'silhouette_score': silhouette,
        'davies_bouldin_index': db_index,
        'cluster_sizes': cluster_sizes,
        'train_time': train_time
    }
    
    return model, metrics


def evaluate_elbow_method(X_train, k_values=[50, 100, 150, 200, 300, 500]):
    """
    Evaluate different values of K using elbow method
    
    Args:
        X_train: Training data
        k_values: List of K values to try
        
    Returns:
        DataFrame with results for each K
    """
    print("ELBOW METHOD - Finding Optimal Number of Clusters")
    
    results = []
    
    for k in k_values:
        print(f"\nTesting K = {k}")
        
        model, metrics = train_kmeans_model(X_train, n_clusters=k)
        
        results.append({
            'k': k,
            'inertia': metrics['inertia'],
            'silhouette_score': metrics['silhouette_score'],
            'davies_bouldin_index': metrics['davies_bouldin_index'],
            'train_time': metrics['train_time']
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nELBOW METHOD RESULTS")
    print(results_df.to_string(index=False))
    
    return results_df


def plot_elbow_curves(results_df, save_dir=None):
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))
        save_dir = os.path.join(base_dir, 'report', 'model_comparison')
    """
    Plot elbow curves for selecting optimal K
    """
    print("\nGenerating elbow plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Inertia plot
    axes[0].plot(results_df['k'], results_df['inertia'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    axes[0].set_title('Elbow Method: Inertia', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette score plot
    axes[1].plot(results_df['k'], results_df['silhouette_score'], 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Davies-Bouldin Index plot
    axes[2].plot(results_df['k'], results_df['davies_bouldin_index'], 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Index (lower is better)', fontsize=12)
    axes[2].set_title('Davies-Bouldin Index vs K', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'kmeans_elbow_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Elbow plots saved to: {save_path}")
    plt.close()


def select_optimal_k(results_df):
    """
    Select optimal K based on elbow method results
    
    Uses a simple heuristic: best silhouette score
    """
    # Find K with best silhouette score
    best_idx = results_df['silhouette_score'].idxmax()
    optimal_k = results_df.loc[best_idx, 'k']
    
    print(f"\nRecommended K based on Silhouette Score: {optimal_k}")
    
    return int(optimal_k)


def save_kmeans_model(model, metrics, save_path):
    """
    Save trained K-Means model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'metrics': metrics,
            'cluster_centers': model.cluster_centers_,
            'n_clusters': model.n_clusters
        }, f)
    
    print(f"\nK-Means model saved to: {save_path}")


def train_final_model(X_train, X_val, X_test, train_df, val_df, test_df, 
                      n_clusters=None, run_elbow=True):
    """
    Train final K-Means model
    
    Args:
        X_train, X_val, X_test: Feature matrices
        train_df, val_df, test_df: Original DataFrames with labels
        n_clusters: Number of clusters (if None, will run elbow method)
        run_elbow: If True, run elbow method to find optimal K
    """
    print("K-MEANS CLUSTERING FOR DONOR SEGMENTATION")
    
    print(f"\nTraining set size: {len(X_train):,}")
    print(f"Validation set size: {len(X_val):,}")
    print(f"Test set size: {len(X_test):,}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Run elbow method if requested
    if run_elbow or n_clusters is None:
        results_df = evaluate_elbow_method(X_train)
        plot_elbow_curves(results_df)
        
        # Save elbow results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))
        results_path = os.path.join(base_dir, 'models', 'clustering', 'elbow_results.csv')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nElbow results saved to: {results_path}")
        
        if n_clusters is None:
            n_clusters = select_optimal_k(results_df)
    
    # Train final model with selected K
    print(f"\nTRAINING FINAL MODEL WITH K={n_clusters}")
    
    final_model, final_metrics = train_kmeans_model(X_train, n_clusters=n_clusters)
    
    # Assign clusters to all data
    print("\nAssigning clusters to all data...")
    train_clusters = final_model.predict(X_train)
    val_clusters = final_model.predict(X_val)
    test_clusters = final_model.predict(X_test)
    
    # Add cluster assignments to DataFrames
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['cluster'] = train_clusters
    val_df['cluster'] = val_clusters
    test_df['cluster'] = test_clusters
    
    # Determine absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    clustering_dir = os.path.join(base_dir, 'models', 'clustering')
    assignments_dir = os.path.join(clustering_dir, 'cluster_assignments')
    
    # Save model
    model_path = os.path.join(clustering_dir, 'kmeans_model.pkl')
    save_kmeans_model(final_model, final_metrics, model_path)
    
    # Save cluster assignments
    print("\nSaving cluster assignments...")
    os.makedirs(assignments_dir, exist_ok=True)
    
    train_df[['id', 'cluster']].to_csv(os.path.join(assignments_dir, 'train_clusters.csv'), index=False)
    val_df[['id', 'cluster']].to_csv(os.path.join(assignments_dir, 'val_clusters.csv'), index=False)
    test_df[['id', 'cluster']].to_csv(os.path.join(assignments_dir, 'test_clusters.csv'), index=False)
    
    print(f"Cluster assignments saved to: {assignments_dir}")
    
    return final_model, final_metrics, train_df, val_df, test_df


def main():
    """
    Main K-Means training script
    """
    # Load data
    print("Loading data splits...")
    train_df, val_df, test_df = load_data_splits()
    
    # Prepare clustering features
    clustering_data = prepare_clustering_data(train_df, val_df, test_df)
    
    X_train = clustering_data['X_train']
    X_val = clustering_data['X_val']
    X_test = clustering_data['X_test']
    
    # Train K-Means
    # Set run_elbow=True to find optimal K, or specify n_clusters directly
    model, metrics, train_df_clustered, val_df_clustered, test_df_clustered = train_final_model(
        X_train, X_val, X_test,
        train_df, val_df, test_df,
        n_clusters=None,  # Will run elbow method
        run_elbow=True
    )
    
    print("\nK-MEANS CLUSTERING COMPLETE")
    print(f"Model trained with {model.n_clusters} clusters")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print("\nNext steps:")
    print("  1. Run profile_clusters.py to analyze cluster characteristics")
    print("  2. Run build_knn_index.py to build FAISS indexes")


if __name__ == "__main__":
    main()
