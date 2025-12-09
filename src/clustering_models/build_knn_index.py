"""
Build FAISS KNN Index for Fast Similarity Search
Create FAISS indexes for efficient K-Nearest Neighbors queries
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_training_data import load_data_splits, prepare_clustering_data


def build_faiss_index(X, use_gpu=False):
    """
    Build FAISS index for fast similarity search
    
    Args:
        X: Feature matrix (numpy array)
        use_gpu: Whether to use GPU (if available)
        
    Returns:
        FAISS index
    """
    try:
        import faiss
    except ImportError:
        print("FAISS not installed. Installing with: pip install faiss-cpu")
        print("For GPU support, use: pip install faiss-gpu")
        raise
    
    print(f"\nBuilding FAISS index for {len(X):,} vectors with {X.shape[1]} dimensions...")
    
    start_time = time.time()
    
    # Convert to float32 (required by FAISS)
    X = X.astype('float32')
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(X)
    
    # Create index
    # Using IndexFlatL2 for exact search (good for moderate scale)
    # For larger scale, use IndexIVFFlat or IndexHNSWFlat
    dimension = X.shape[1]
    
    if len(X) < 100000:
        # Small dataset: use exact search
        index = faiss.IndexFlatL2(dimension)
        print("Using IndexFlatL2 (exact search)")
    else:
        # Larger dataset: use approximate search with IVF
        nlist = min(int(np.sqrt(len(X))), 1000)  # Number of clusters for IVF
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        print(f"Using IndexIVFFlat with {nlist} clusters (approximate search)")
        
        # Train the index
        print("Training index...")
        index.train(X)
    
    # Add vectors to index
    print("Adding vectors to index...")
    index.add(X)
    
    build_time = time.time() - start_time
    
    print(f"Index built in {build_time:.2f}s")
    print(f"Index contains {index.ntotal} vectors")
    
    return index


def build_cluster_specific_indexes(X_train, cluster_assignments):
    """
    Build separate FAISS index for each cluster
    
    Args:
        X_train: Feature matrix
        cluster_assignments: Array of cluster IDs
        
    Returns:
        Dictionary mapping cluster_id -> (index, indices_in_cluster)
    """
    print("\nBUILDING CLUSTER-SPECIFIC INDEXES")
    
    cluster_indexes = {}
    unique_clusters = np.unique(cluster_assignments)
    
    for cluster_id in unique_clusters:
        # Get data for this cluster
        cluster_mask = cluster_assignments == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        X_cluster = X_train[cluster_mask]
        
        if len(X_cluster) < 10:
            print(f"Skipping cluster {cluster_id} (only {len(X_cluster)} samples)")
            continue
        
        print(f"\nCluster {cluster_id}: {len(X_cluster):,} contributors")
        
        # Build index for this cluster
        index = build_faiss_index(X_cluster)
        
        # Store index and original indices
        cluster_indexes[int(cluster_id)] = {
            'index': index,
            'indices': cluster_indices,
            'size': len(X_cluster)
        }
        
        if cluster_id % 20 == 0:
            print(f"Progress: {cluster_id}/{len(unique_clusters)} clusters")
    
    print(f"\nBuilt indexes for {len(cluster_indexes)} clusters")
    
    return cluster_indexes


def build_global_index(X_train):
    """
    Build a single global FAISS index for all data
    
    Args:
        X_train: Feature matrix
        
    Returns:
        FAISS index
    """
    print("\nBUILDING GLOBAL INDEX")
    
    index = build_faiss_index(X_train)
    
    return index


def save_indexes(cluster_indexes, global_index, save_dir=None):
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))
        save_dir = os.path.join(base_dir, 'models', 'clustering', 'faiss_indexes')
    """
    Save FAISS indexes to disk
    
    Args:
        cluster_indexes: Dictionary of cluster-specific indexes
        global_index: Global index
        save_dir: Directory to save indexes
    """
    try:
        import faiss
    except ImportError:
        print("FAISS not available")
        return
    
    print("\nSAVING INDEXES")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save global index
    global_path = os.path.join(save_dir, 'global_index.faiss')
    faiss.write_index(global_index, global_path)
    print(f"Global index saved to: {global_path}")
    
    # Save cluster-specific indexes
    cluster_dir = os.path.join(save_dir, 'clusters')
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Save index metadata
    cluster_metadata = {}
    
    for cluster_id, data in cluster_indexes.items():
        # Save FAISS index
        index_path = os.path.join(cluster_dir, f'cluster_{cluster_id}.faiss')
        faiss.write_index(data['index'], index_path)
        
        # Save indices mapping
        indices_path = os.path.join(cluster_dir, f'cluster_{cluster_id}_indices.npy')
        np.save(indices_path, data['indices'])
        
        # Store metadata
        cluster_metadata[cluster_id] = {
            'size': data['size'],
            'index_path': index_path,
            'indices_path': indices_path
        }
    
    # Save metadata
    metadata_path = os.path.join(save_dir, 'index_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'cluster_metadata': cluster_metadata,
            'num_clusters': len(cluster_indexes),
            'total_vectors': global_index.ntotal
        }, f)
    
    print(f"Cluster indexes saved to: {cluster_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total: {len(cluster_indexes)} cluster indexes + 1 global index")


def test_index_performance(cluster_indexes, global_index, X_test, cluster_assignments_test, k=20):
    """
    Test index query performance
    
    Args:
        cluster_indexes: Cluster-specific indexes
        global_index: Global index
        X_test: Test feature matrix
        cluster_assignments_test: Test cluster assignments
        k: Number of neighbors to retrieve
    """
    print("\nTESTING INDEX PERFORMANCE")
    
    try:
        import faiss
    except ImportError:
        print("FAISS not available")
        return
    
    # Sample test queries
    n_test_queries = min(1000, len(X_test))
    test_indices = np.random.choice(len(X_test), n_test_queries, replace=False)
    X_test_sample = X_test[test_indices].astype('float32')
    
    # Normalize
    faiss.normalize_L2(X_test_sample)
    
    # Test global index
    print(f"\nTesting global index with {n_test_queries} queries...")
    start_time = time.time()
    distances, indices = global_index.search(X_test_sample, k)
    global_time = time.time() - start_time
    
    print(f"Global index query time: {global_time:.3f}s")
    print(f"Average per query: {global_time/n_test_queries*1000:.2f}ms")
    
    # Test cluster-specific indexes
    print(f"\nTesting cluster-specific indexes...")
    cluster_times = []
    
    for i in test_indices[:100]:  # Test first 100 queries
        cluster_id = int(cluster_assignments_test[i])
        
        if cluster_id not in cluster_indexes:
            continue
        
        query = X_test[i:i+1].astype('float32')
        faiss.normalize_L2(query)
        
        start_time = time.time()
        distances, indices = cluster_indexes[cluster_id]['index'].search(query, k)
        query_time = time.time() - start_time
        cluster_times.append(query_time)
    
    if cluster_times:
        avg_cluster_time = np.mean(cluster_times)
        print(f"Cluster-specific average query time: {avg_cluster_time*1000:.2f}ms")
        print(f"Speedup: {(global_time/n_test_queries)/avg_cluster_time:.1f}x")
    
    print("\nIndex performance test complete")


def main():
    """
    Main FAISS index building script
    """
    print("BUILDING FAISS KNN INDEXES")
    
    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_data_splits()
    
    # Prepare clustering features
    clustering_data = prepare_clustering_data(train_df, val_df, test_df)
    
    X_train = clustering_data['X_train']
    X_test = clustering_data['X_test']
    
    # Load cluster assignments
    print("\nLoading cluster assignments...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    train_cluster_path = os.path.join(base_dir, 'models', 'clustering', 'cluster_assignments', 'train_clusters.csv')
    test_cluster_path = os.path.join(base_dir, 'models', 'clustering', 'cluster_assignments', 'test_clusters.csv')
    
    train_clusters = pd.read_csv(train_cluster_path)
    test_clusters = pd.read_csv(test_cluster_path)
    
    cluster_assignments_train = train_clusters['cluster'].values
    cluster_assignments_test = test_clusters['cluster'].values
    
    # Build global index
    global_index = build_global_index(X_train)
    
    # Build cluster-specific indexes
    cluster_indexes = build_cluster_specific_indexes(X_train, cluster_assignments_train)
    
    # Save indexes
    save_indexes(cluster_indexes, global_index)
    
    # Test performance
    test_index_performance(cluster_indexes, global_index, X_test, cluster_assignments_test)
    
    print("\nFAISS INDEX BUILDING COMPLETE")
    print("\nIndexes saved to: ../../models/clustering/faiss_indexes")
    print("Next step: Run knn_pipeline.py to create KNN prediction pipeline")


if __name__ == "__main__":
    main()
