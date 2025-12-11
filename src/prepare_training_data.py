"""
Data Preparation Module
Load data from database, create train/val/test splits for both prediction approaches
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from datetime import datetime
from db_utils import db_connection
from feature_engineering import extract_features, prepare_features_for_modeling, PARTIES


def load_all_contributors(min_donations=1, sample_size=None):
    """
    Load all contributors from database
    
    Args:
        min_donations: Minimum number of donations to include
        sample_size: If specified, randomly sample this many contributors
        
    Returns:
        DataFrame with all contributor data
    """
    print("Loading contributors from database...")
    
    query = """
    SELECT 
        id,
        donor_key,
        first_name,
        last_name,
        city,
        state,
        zip_code,
        employer,
        occupation,
        total_donations,
        total_amount,
        donations_last_2years,
        amount_last_2years,
        recency_days,
        avg_donation_amount,
        first_donation_date,
        last_donation_date,
        unique_committees,
        dem_donations,
        dem_amount,
        rep_donations,
        rep_amount,
        lib_donations,
        lib_amount,
        gre_donations,
        gre_amount,
        ind_donations,
        ind_amount,
        other_donations,
        other_amount,
        dem_pct,
        rep_pct,
        primary_party
    FROM contributors
    WHERE total_donations >= %s
        AND primary_party IN ('DEM', 'REP', 'LIB', 'GRE', 'IND', 'OTHER')
        AND total_amount > 0
    """ % min_donations
    
    if sample_size:
        query += " ORDER BY RAND() LIMIT %s" % sample_size
    
    with db_connection(use_dict_cursor=False) as conn:
        df = pd.read_sql(query, conn)
    
    print(f"Loaded {len(df):,} contributors")
    print(f"Party distribution:\n{df['primary_party'].value_counts()}")
    
    return df


def create_train_val_test_splits(df, test_size=0.15, val_size=0.15, random_state=42, 
                                 use_temporal_split=True):
    """
    Create train/validation/test splits (temporal or random)
    
    Args:
        df: DataFrame with all data
        test_size: Proportion for test set (only used if not temporal)
        val_size: Proportion for validation set (only used if not temporal)
        random_state: Random seed
        use_temporal_split: If True, split by year (train<=2023, val=2024, test=2025)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print(f"\nCreating train/val/test splits...")
    
    if use_temporal_split:
        print("Using TEMPORAL splits to prevent data leakage:")
        print("  - Train: donations through 2023")
        print("  - Validation: donations in 2024")
        print("  - Test: donations in 2025")
        
        # Convert last_donation_date to datetime, handling bad dates
        df['last_donation_date'] = pd.to_datetime(df['last_donation_date'], errors='coerce')
        
        # Filter out invalid dates (before 1980 or after 2030)
        valid_date_mask = (
            df['last_donation_date'].notna() &
            (df['last_donation_date'].dt.year >= 1980) &
            (df['last_donation_date'].dt.year <= 2030)
        )
        
        print(f"\nFiltering dates: {(~valid_date_mask).sum():,} invalid dates removed")
        df = df[valid_date_mask].copy()
        
        df['last_donation_year'] = df['last_donation_date'].dt.year
        
        # Temporal split by year
        train_df = df[df['last_donation_year'] <= 2023].copy()
        val_df = df[df['last_donation_year'] == 2024].copy()
        test_df = df[df['last_donation_year'] == 2025].copy()
        
        print(f"\nTrain set: {len(train_df):,} contributors (through 2023)")
        print(f"Val set: {len(val_df):,} contributors (2024)")
        print(f"Test set: {len(test_df):,} contributors (2025)")
        
        # Show year distribution
        print(f"\nYear distribution in training set:")
        if len(train_df) > 0:
            year_dist = train_df['last_donation_year'].value_counts().sort_index()
            for year, count in year_dist.tail(5).items():
                print(f"  {year}: {count:,} contributors")
        
    else:
        print("Using RANDOM stratified splits...")
        
        # First split: separate out test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['primary_party'],
            random_state=random_state
        )
        
        # Second split: separate train and validation
        val_proportion = val_size / (1 - test_size)  # Adjust proportion
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_proportion,
            stratify=train_val_df['primary_party'],
            random_state=random_state
        )
        
        print(f"Train set: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val set: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify party distribution
    print("\nParty distribution in train set:")
    print(train_df['primary_party'].value_counts(normalize=True) * 100)
    
    print("\nParty distribution in validation set:")
    print(val_df['primary_party'].value_counts(normalize=True) * 100)
    
    print("\nParty distribution in test set:")
    print(test_df['primary_party'].value_counts(normalize=True) * 100)
    
    return train_df, val_df, test_df


def get_models_dir():
    """Get absolute path to models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), 'models')


def prepare_supervised_learning_data(train_df, val_df, test_df, 
                                     include_history=True,
                                     save_dir=None):
    """
    Prepare data for supervised learning models
    
    Args:
        train_df, val_df, test_df: Data splits
        include_history: Whether to include historical features
        save_dir: Directory to save preprocessors
        
    Returns:
        Dictionary with prepared datasets
    """
    if save_dir is None:
        save_dir = os.path.join(get_models_dir(), 'supervised')
    
    print(f"\nPreparing supervised learning data (include_history={include_history})...")
    
    # Prepare features and fit transformer on training data
    X_train, transformer, feature_names = prepare_features_for_modeling(
        train_df, 
        include_history=include_history,
        include_party_amounts=False,  # Don't include amounts as features for Stage 1
        fit_transformer=True
    )
    
    # Transform validation and test sets
    X_val, _, _ = prepare_features_for_modeling(
        val_df,
        include_history=include_history,
        include_party_amounts=False,
        transformer=transformer,
        fit_transformer=False
    )
    
    X_test, _, _ = prepare_features_for_modeling(
        test_df,
        include_history=include_history,
        include_party_amounts=False,
        transformer=transformer,
        fit_transformer=False
    )
    
    # Prepare targets for Stage 1 (party classification)
    y_train_party = train_df['primary_party'].values
    y_val_party = val_df['primary_party'].values
    y_test_party = test_df['primary_party'].values
    
    # Prepare targets for Stage 2 (party-specific amounts)
    party_amount_targets = {}
    for party in PARTIES:
        party_col = f'{party.lower()}_amount'
        party_amount_targets[party] = {
            'train': train_df[party_col].values,
            'val': val_df[party_col].values,
            'test': test_df[party_col].values
        }
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Save transformer
    os.makedirs(save_dir, exist_ok=True)
    history_suffix = '_full' if include_history else '_demo'
    transformer_path = os.path.join(save_dir, f'feature_transformer{history_suffix}.pkl')
    
    with open(transformer_path, 'wb') as f:
        pickle.dump({
            'transformer': transformer,
            'feature_names': feature_names,
            'include_history': include_history
        }, f)
    print(f"Saved transformer to {transformer_path}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train_party': y_train_party,
        'y_val_party': y_val_party,
        'y_test_party': y_test_party,
        'party_amounts': party_amount_targets,
        'transformer': transformer,
        'feature_names': feature_names,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }


def prepare_clustering_data(train_df, val_df, test_df, save_dir=None):
    """
    Prepare data for K-Means clustering
    Uses full feature set for clustering
    
    Args:
        train_df, val_df, test_df: Data splits
        save_dir: Directory to save preprocessors
        
    Returns:
        Dictionary with prepared datasets
    """
    if save_dir is None:
        save_dir = os.path.join(get_models_dir(), 'clustering')
    
    print("\nPreparing clustering data...")
    
    # Use full history for clustering
    X_train, transformer, feature_names = prepare_features_for_modeling(
        train_df,
        include_history=True,
        include_party_amounts=True,  # Include amounts for better clustering
        fit_transformer=True
    )
    
    X_val, _, _ = prepare_features_for_modeling(
        val_df,
        include_history=True,
        include_party_amounts=True,
        transformer=transformer,
        fit_transformer=False
    )
    
    X_test, _, _ = prepare_features_for_modeling(
        test_df,
        include_history=True,
        include_party_amounts=True,
        transformer=transformer,
        fit_transformer=False
    )
    
    print(f"Clustering feature matrix shape: {X_train.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Save transformer
    os.makedirs(save_dir, exist_ok=True)
    transformer_path = os.path.join(save_dir, 'feature_transformer_clustering.pkl')
    
    with open(transformer_path, 'wb') as f:
        pickle.dump({
            'transformer': transformer,
            'feature_names': feature_names,
            'include_history': True,
            'include_party_amounts': True
        }, f)
    print(f"Saved clustering transformer to {transformer_path}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'transformer': transformer,
        'feature_names': feature_names,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }


def save_data_splits(train_df, val_df, test_df, save_dir=None):
    """
    Save train/val/test splits to disk
    
    Args:
        train_df, val_df, test_df: Data splits
        save_dir: Directory to save splits
    """
    if save_dir is None:
        save_dir = os.path.join(get_models_dir(), 'data_splits')
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving data splits to {save_dir}...")
    
    # Convert date columns to datetime before saving
    date_columns = ['first_donation_date', 'last_donation_date']
    for df in [train_df, val_df, test_df]:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Save as parquet for efficiency with large datasets
    train_df.to_parquet(os.path.join(save_dir, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(save_dir, 'val.parquet'), index=False)
    test_df.to_parquet(os.path.join(save_dir, 'test.parquet'), index=False)
    
    print("Data splits saved successfully")


def load_data_splits(save_dir=None):
    """
    Load previously saved data splits
    
    Args:
        save_dir: Directory with saved splits (uses absolute path by default)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if save_dir is None:
        # Use absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        save_dir = os.path.join(base_dir, 'models', 'data_splits')
    
    print(f"Loading data splits from {save_dir}...")
    
    train_df = pd.read_parquet(os.path.join(save_dir, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(save_dir, 'val.parquet'))
    test_df = pd.read_parquet(os.path.join(save_dir, 'test.parquet'))
    
    print(f"Loaded train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,}")
    
    return train_df, val_df, test_df


def main():
    """
    Main data preparation pipeline
    """
    print("DATA PREPARATION FOR TWO-STAGE PREDICTIVE MODELS")
    
    # Configuration
    SAMPLE_SIZE = 500000  # 500K contributors (faster training, still good accuracy)
    # SAMPLE_SIZE = 4000000  # 4M contributors for better coverage
    # SAMPLE_SIZE = None  # Full dataset (slower but most accurate)
    MIN_DONATIONS = 1
    USE_TEMPORAL_SPLIT = True  # Train<=2023, Val=2024, Test=2025 (prevents data leakage)
    
    # Load data
    df = load_all_contributors(min_donations=MIN_DONATIONS, sample_size=SAMPLE_SIZE)
    
    # Create splits (temporal to avoid data snooping)
    train_df, val_df, test_df = create_train_val_test_splits(
        df, 
        use_temporal_split=USE_TEMPORAL_SPLIT
    )
    
    # Save splits for reuse
    save_data_splits(train_df, val_df, test_df)
    
    # Prepare for supervised learning (full features)
    print("\nSUPERVISED LEARNING - FULL FEATURES")
    supervised_full = prepare_supervised_learning_data(
        train_df, val_df, test_df,
        include_history=True
    )
    
    # Prepare for supervised learning (demographics only)
    print("\nSUPERVISED LEARNING - DEMOGRAPHICS ONLY")
    supervised_demo = prepare_supervised_learning_data(
        train_df, val_df, test_df,
        include_history=False
    )
    
    # Prepare for clustering
    print("\nCLUSTERING (K-MEANS + KNN)")
    clustering_data = prepare_clustering_data(
        train_df, val_df, test_df
    )
    
    # Summary
    print("\nDATA PREPARATION COMPLETE!")
    print(f"Total contributors: {len(df):,}")
    print(f"Training set: {len(train_df):,}")
    print(f"Validation set: {len(val_df):,}")
    print(f"Test set: {len(test_df):,}")
    print("\nFeatures prepared for:")
    print(f"  - Supervised learning (full): {supervised_full['X_train'].shape[1]} features")
    print(f"  - Supervised learning (demo): {supervised_demo['X_train'].shape[1]} features")
    print(f"  - Clustering: {clustering_data['X_train'].shape[1]} features")
    print(f"\nAll preprocessors and data splits saved to {get_models_dir()}")


if __name__ == "__main__":
    main()
