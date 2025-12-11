"""
Feature Engineering Module
Shared feature extraction and transformation for both prediction approaches
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle
import os


# Party mapping
PARTIES = ['DEM', 'REP', 'LIB', 'GRE', 'IND', 'OTHER']


def extract_temporal_features(df):
    """
    Extract temporal features from donation dates
    
    Args:
        df: DataFrame with first_donation_date and last_donation_date columns
        
    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    
    # Convert to datetime if not already (use errors='coerce' to handle invalid dates)
    if 'first_donation_date' in df.columns:
        df['first_donation_date'] = pd.to_datetime(df['first_donation_date'], errors='coerce')
        df['last_donation_date'] = pd.to_datetime(df['last_donation_date'], errors='coerce')
        
        # Calculate donor tenure (days between first and last donation)
        df['donor_tenure_days'] = (df['last_donation_date'] - df['first_donation_date']).dt.days
        
        # Calculate donation frequency (donations per year)
        df['donation_frequency'] = df['total_donations'] / (df['donor_tenure_days'] / 365.25 + 1)
        
        # Extract year and month features
        df['first_donation_year'] = df['first_donation_date'].dt.year
        df['last_donation_year'] = df['last_donation_date'].dt.year
        df['first_donation_month'] = df['first_donation_date'].dt.month
        
        # Fill NaN tenure with 0 (single donation)
        df['donor_tenure_days'].fillna(0, inplace=True)
        df['donation_frequency'].fillna(0, inplace=True)
    
    # Fill recency_days if missing
    if 'recency_days' not in df.columns and 'last_donation_date' in df.columns:
        reference_date = datetime.now()
        df['recency_days'] = (reference_date - df['last_donation_date']).dt.days
    
    return df


def extract_behavioral_features(df):
    """
    Extract behavioral features from donation patterns
    
    Args:
        df: DataFrame with donation history
        
    Returns:
        DataFrame with additional behavioral features
    """
    df = df.copy()
    
    # Donor loyalty score (higher = more loyal to single party)
    party_columns = [f'{party.lower()}_amount' for party in PARTIES]
    if all(col in df.columns for col in party_columns):
        # Calculate party diversity using entropy
        party_amounts = df[party_columns].fillna(0)
        total_amounts = party_amounts.sum(axis=1).replace(0, 1)  # Avoid division by zero
        party_proportions = party_amounts.div(total_amounts, axis=0)
        
        # Shannon entropy (higher = more diverse giving)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        party_proportions_safe = party_proportions + epsilon
        df['party_diversity'] = -(party_proportions_safe * np.log2(party_proportions_safe)).sum(axis=1)
        
        # Loyalty score (inverse of diversity, normalized)
        max_entropy = np.log2(len(PARTIES))
        df['party_loyalty'] = 1 - (df['party_diversity'] / max_entropy)
    
    # Donor type: one-time vs repeat
    if 'total_donations' in df.columns:
        df['is_repeat_donor'] = (df['total_donations'] > 1).astype(int)
        df['is_frequent_donor'] = (df['total_donations'] >= 5).astype(int)
    
    # Average donation consistency (coefficient of variation would require transaction-level data)
    # For now, use ratio of recent to total activity
    if 'amount_last_2years' in df.columns and 'total_amount' in df.columns:
        df['recent_activity_ratio'] = df['amount_last_2years'] / (df['total_amount'] + 1)
    
    return df


def extract_geographic_features(df):
    """
    Extract geographic features from location data
    
    Args:
        df: DataFrame with state and zip_code columns
        
    Returns:
        DataFrame with additional geographic features
    """
    df = df.copy()
    
    # Extract 3-digit zip code prefix for regional clustering
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype(str).str.strip()
        df['zip_prefix'] = df['zip_code'].str[:3]
        # Fill empty with 'UNK'
        df['zip_prefix'] = df['zip_prefix'].replace('', 'UNK').fillna('UNK')
    
    # State feature - already in df
    if 'state' in df.columns:
        df['state'] = df['state'].fillna('UNK').replace('', 'UNK')
    
    return df


def create_demographic_features(df, top_n_categories=100):
    """
    Create demographic features from employer and occupation
    
    Args:
        df: DataFrame with employer and occupation columns
        top_n_categories: Keep only top N most frequent categories
        
    Returns:
        DataFrame with cleaned demographic features
    """
    df = df.copy()
    
    # Clean employer and occupation
    if 'employer' in df.columns:
        df['employer'] = df['employer'].fillna('UNKNOWN').replace('', 'UNKNOWN')
        df['employer'] = df['employer'].str.upper().str.strip()
        
        # Group rare employers into 'OTHER'
        employer_counts = df['employer'].value_counts()
        top_employers = employer_counts.head(top_n_categories).index
        df['employer_grouped'] = df['employer'].apply(
            lambda x: x if x in top_employers else 'OTHER'
        )
    
    if 'occupation' in df.columns:
        df['occupation'] = df['occupation'].fillna('UNKNOWN').replace('', 'UNKNOWN')
        df['occupation'] = df['occupation'].str.upper().str.strip()
        
        # Group rare occupations into 'OTHER'
        occupation_counts = df['occupation'].value_counts()
        top_occupations = occupation_counts.head(top_n_categories).index
        df['occupation_grouped'] = df['occupation'].apply(
            lambda x: x if x in top_occupations else 'OTHER'
        )
    
    return df


def extract_features(df, include_history=True, include_party_amounts=False):
    """
    Main feature extraction pipeline
    
    Args:
        df: DataFrame with contributor data
        include_history: If True, include historical donation features
        include_party_amounts: If True, include party-specific donation amounts (for Stage 2)
        
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Apply all feature engineering steps
    df = extract_geographic_features(df)
    df = create_demographic_features(df)
    
    if include_history:
        df = extract_temporal_features(df)
        df = extract_behavioral_features(df)
    
    return df


def get_feature_columns(include_history=True, include_party_amounts=False):
    """
    Get list of feature columns for modeling
    
    Args:
        include_history: If True, include historical features
        include_party_amounts: If True, include party-specific amounts
        
    Returns:
        Dictionary with categorical and numerical feature lists
    """
    # Categorical features
    categorical_features = ['state', 'zip_prefix', 'employer_grouped', 'occupation_grouped']
    
    # Numerical features - demographics only
    numerical_features = []
    
    if include_history:
        # Add historical numerical features
        numerical_features.extend([
            'total_donations',
            'total_amount',
            'avg_donation_amount',
            'donations_last_2years',
            'amount_last_2years',
            'recency_days',
            'donor_tenure_days',
            'donation_frequency',
            'unique_committees',
            'party_diversity',
            'party_loyalty',
            'is_repeat_donor',
            'is_frequent_donor',
            'recent_activity_ratio',
            'first_donation_year',
            'last_donation_year',
            'first_donation_month'
        ])
        
        # Add party-specific donation counts
        for party in PARTIES:
            numerical_features.append(f'{party.lower()}_donations')
    
    if include_party_amounts:
        # Add party-specific amounts
        for party in PARTIES:
            numerical_features.append(f'{party.lower()}_amount')
    
    return {
        'categorical': categorical_features,
        'numerical': numerical_features
    }


def create_feature_transformer(categorical_features, numerical_features):
    """
    Create sklearn ColumnTransformer for preprocessing
    
    Args:
        categorical_features: List of categorical column names
        numerical_features: List of numerical column names
        
    Returns:
        ColumnTransformer object
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def prepare_features_for_modeling(df, include_history=True, include_party_amounts=False, 
                                   transformer=None, fit_transformer=True):
    """
    Prepare features for modeling
    
    Args:
        df: DataFrame with raw contributor data
        include_history: Include historical features
        include_party_amounts: Include party-specific amounts
        transformer: Pre-fitted transformer (optional)
        fit_transformer: If True, fit new transformer on this data
        
    Returns:
        Tuple of (X, transformer, feature_names)
    """
    # Extract features
    df_features = extract_features(df, include_history=include_history, 
                                   include_party_amounts=include_party_amounts)
    
    # Get feature columns
    feature_cols = get_feature_columns(include_history=include_history, 
                                       include_party_amounts=include_party_amounts)
    
    # Select only existing columns
    categorical_features = [col for col in feature_cols['categorical'] if col in df_features.columns]
    numerical_features = [col for col in feature_cols['numerical'] if col in df_features.columns]
    
    # Handle missing values in numerical features
    for col in numerical_features:
        if col in df_features.columns:
            df_features[col] = df_features[col].fillna(df_features[col].median())
    
    # Create or use provided transformer
    if transformer is None and fit_transformer:
        transformer = create_feature_transformer(categorical_features, numerical_features)
        X = transformer.fit_transform(df_features)
    elif transformer is not None:
        X = transformer.transform(df_features)
    else:
        # Return raw features
        all_features = categorical_features + numerical_features
        X = df_features[all_features].values
    
    # Get feature names after transformation
    if transformer is not None and fit_transformer:
        feature_names = []
        if len(categorical_features) > 0:
            feature_names.extend(transformer.named_transformers_['cat'].get_feature_names_out(categorical_features))
        feature_names.extend(numerical_features)
    else:
        feature_names = categorical_features + numerical_features
    
    return X, transformer, feature_names


def get_zip_demographics(zip_code, df_all_contributors):
    """
    Get aggregated demographics for a ZIP code (for new contributors)
    
    Args:
        zip_code: ZIP code string
        df_all_contributors: DataFrame with all contributors
        
    Returns:
        Dictionary with ZIP-level statistics
    """
    zip_prefix = str(zip_code)[:3]
    
    # Filter to ZIP prefix
    zip_data = df_all_contributors[df_all_contributors['zip_code'].str.startswith(zip_prefix)]
    
    if len(zip_data) == 0:
        # Return defaults if no data
        return {
            'avg_donation': 0,
            'avg_donations_count': 0,
            'dem_pct': 0.5,
            'rep_pct': 0.5,
            'primary_party': 'DEM'
        }
    
    stats = {
        'avg_donation': zip_data['avg_donation_amount'].mean(),
        'avg_donations_count': zip_data['total_donations'].mean(),
        'dem_pct': zip_data['dem_pct'].mean(),
        'rep_pct': zip_data['rep_pct'].mean(),
        'primary_party': zip_data['primary_party'].mode()[0] if len(zip_data) > 0 else 'DEM'
    }
    
    return stats


def create_new_contributor_features(zip_code, state, df_all_contributors=None):
    """
    Create feature vector for a new contributor (not in database)
    
    Args:
        zip_code: ZIP code string
        state: State abbreviation
        df_all_contributors: DataFrame with all contributors (for ZIP demographics)
        
    Returns:
        DataFrame with single row of features
    """
    # Create basic demographic features
    data = {
        'state': state if state else 'UNK',
        'zip_code': str(zip_code),
        'employer_grouped': 'UNKNOWN',
        'occupation_grouped': 'UNKNOWN',
        'zip_prefix': str(zip_code)[:3] if zip_code else 'UNK'
    }
    
    # If we have contributor data, use ZIP-level statistics
    if df_all_contributors is not None:
        zip_stats = get_zip_demographics(zip_code, df_all_contributors)
        # Can use these as features or for imputation
        data['avg_donation_amount'] = zip_stats['avg_donation']
        data['total_donations'] = zip_stats['avg_donations_count']
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    return df


def save_transformer(transformer, feature_names, filepath):
    """
    Save feature transformer and metadata
    
    Args:
        transformer: Fitted ColumnTransformer
        feature_names: List of feature names
        filepath: Path to save pickle file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump({
            'transformer': transformer,
            'feature_names': feature_names
        }, f)
    
    print(f"Transformer saved to {filepath}")


def load_transformer(filepath):
    """
    Load feature transformer and metadata
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Tuple of (transformer, feature_names)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data['transformer'], data['feature_names']


if __name__ == "__main__":
    # Test feature engineering
    from db_utils import db_connection
    
    print("Testing feature engineering...")
    
    # Load sample data
    query = """
    SELECT * FROM contributors
    WHERE total_donations > 0
    LIMIT 1000
    """
    
    with db_connection() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"Loaded {len(df)} contributors")
    
    # Test feature extraction
    df_features = extract_features(df, include_history=True, include_party_amounts=False)
    print(f"\nFeatures extracted: {df_features.shape}")
    print(f"Columns: {df_features.columns.tolist()[:10]}...")
    
    # Test feature preparation
    X, transformer, feature_names = prepare_features_for_modeling(
        df, include_history=True, fit_transformer=True
    )
    
    print(f"\nPrepared features shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"First 10 feature names: {feature_names[:10]}")
    
    print("\nFeature engineering test complete!")



