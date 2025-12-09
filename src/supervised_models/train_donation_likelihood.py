"""
Donation Likelihood Prediction Model (FIXED - No Data Leakage)

Predicts:
1. Likelihood that an EXISTING donor will donate AGAIN to a committee
2. Expected donation amount if they do donate

KEY FIX: 
- Both positives and negatives have pre-cutoff donation history
- Positives: Donated before AND after cutoff (repeat donors)
- Negatives: Donated before but NOT after cutoff (lapsed donors)
- Features computed from PRE-CUTOFF data only

Input: contributor_id, cmte_id (with existing relationship)
Output: likelihood_score (0-1), predicted_amount
"""

import logging
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_utils import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('donation_likelihood_training.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / 'models' / 'donation_likelihood'
REPORT_DIR = Path(__file__).parent.parent.parent / 'report' / 'donation_likelihood'

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def compute_point_in_time_features(conn, cutoff_date, sample_contributor_committee_pairs):
    """
    Compute features using ONLY pre-cutoff donations.
    This prevents data leakage from post-cutoff information.
    """
    if len(sample_contributor_committee_pairs) == 0:
        return pd.DataFrame()
    
    # Create temp table for the pairs we want to compute features for
    cursor = conn.cursor()
    
    # Build values for IN clause
    pairs_list = [(int(c), str(cm)) for c, cm in sample_contributor_committee_pairs]
    
    # Compute features from contributions table with date filter
    # We'll do this in chunks to avoid memory issues
    all_features = []
    chunk_size = 10000
    
    for i in range(0, len(pairs_list), chunk_size):
        chunk_pairs = pairs_list[i:i+chunk_size]
        
        # Build the query with explicit pairs
        pairs_str = ','.join([f"({c}, '{cm}')" for c, cm in chunk_pairs])
        
        query = f"""
            WITH pair_data AS (
                SELECT * FROM (VALUES {pairs_str}) AS t(contributor_id, cmte_id)
            ),
            pre_cutoff_donations AS (
                SELECT 
                    c.contributor_id,
                    c.cmte_id,
                    c.transaction_amt,
                    c.transaction_dt,
                    c.transaction_pgi
                FROM contributions c
                JOIN pair_data p ON c.contributor_id = p.contributor_id AND c.cmte_id = p.cmte_id
                WHERE c.transaction_dt < '{cutoff_date}'
            )
            SELECT 
                contributor_id,
                cmte_id,
                COUNT(*) as cc_total_donations,
                SUM(transaction_amt) as cc_total_amount,
                AVG(transaction_amt) as cc_avg_amount,
                STDDEV(transaction_amt) as cc_stddev_amount,
                MIN(transaction_dt) as first_donation_date,
                MAX(transaction_dt) as last_donation_date,
                DATEDIFF('{cutoff_date}', MAX(transaction_dt)) as cc_days_since_last,
                DATEDIFF(MAX(transaction_dt), MIN(transaction_dt)) as cc_donation_span,
                CASE WHEN COUNT(*) > 1 
                     THEN DATEDIFF(MAX(transaction_dt), MIN(transaction_dt)) / (COUNT(*) - 1)
                     ELSE 0 END as cc_avg_frequency,
                SUM(CASE WHEN transaction_pgi = 'P' THEN 1 ELSE 0 END) as cc_donations_primary,
                SUM(CASE WHEN transaction_pgi = 'G' THEN 1 ELSE 0 END) as cc_donations_general,
                CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END as cc_is_recurring
            FROM pre_cutoff_donations
            GROUP BY contributor_id, cmte_id
        """
        
        try:
            chunk_df = pd.read_sql(query, conn)
            all_features.append(chunk_df)
        except Exception as e:
            # Fallback: query each pair individually (slower but works)
            logger.warning(f"Chunk query failed, using individual queries: {e}")
            for contrib_id, cmte_id in chunk_pairs:
                single_query = f"""
                    SELECT 
                        {contrib_id} as contributor_id,
                        '{cmte_id}' as cmte_id,
                        COUNT(*) as cc_total_donations,
                        SUM(transaction_amt) as cc_total_amount,
                        AVG(transaction_amt) as cc_avg_amount,
                        STDDEV(transaction_amt) as cc_stddev_amount,
                        MIN(transaction_dt) as first_donation_date,
                        MAX(transaction_dt) as last_donation_date,
                        DATEDIFF('{cutoff_date}', MAX(transaction_dt)) as cc_days_since_last,
                        DATEDIFF(MAX(transaction_dt), MIN(transaction_dt)) as cc_donation_span,
                        CASE WHEN COUNT(*) > 1 
                             THEN DATEDIFF(MAX(transaction_dt), MIN(transaction_dt)) / (COUNT(*) - 1)
                             ELSE 0 END as cc_avg_frequency,
                        SUM(CASE WHEN transaction_pgi = 'P' THEN 1 ELSE 0 END) as cc_donations_primary,
                        SUM(CASE WHEN transaction_pgi = 'G' THEN 1 ELSE 0 END) as cc_donations_general,
                        CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END as cc_is_recurring
                    FROM contributions
                    WHERE contributor_id = {contrib_id} 
                      AND cmte_id = '{cmte_id}'
                      AND transaction_dt < '{cutoff_date}'
                """
                try:
                    single_df = pd.read_sql(single_query, conn)
                    if len(single_df) > 0 and single_df['cc_total_donations'].iloc[0] > 0:
                        all_features.append(single_df)
                except:
                    pass
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()


def fetch_training_data_fixed(sample_size=100000, cutoff_date='2024-01-01'):
    """
    Fetch training data with PROPER temporal split - NO data leakage.
    
    POSITIVES: Donor-committee pairs that donated BEFORE cutoff AND AFTER cutoff
    NEGATIVES: Donor-committee pairs that donated BEFORE cutoff but NOT AFTER cutoff
    
    Uses donor_totals table (already has first/last donation dates) for FAST sampling.
    Features are computed from PRE-CUTOFF data only!
    """
    logger.info(f"Fetching training data with cutoff: {cutoff_date}")
    logger.info("FIXED: Both positives and negatives have pre-cutoff history")
    logger.info("OPTIMIZED: Using donor_totals for fast sampling")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # === STEP 1: Find REPEAT donors using donor_totals (FAST!) ===
    # first_donation_date < cutoff AND last_donation_date >= cutoff = donated both before AND after
    logger.info("Finding repeat donor-committee pairs (using donor_totals)...")
    
    # Get ID range for efficient sampling
    cursor.execute(f"""
        SELECT MIN(id), MAX(id), COUNT(*) 
        FROM donor_totals 
        WHERE first_donation_date < '{cutoff_date}' 
          AND last_donation_date >= '{cutoff_date}'
    """)
    repeat_min, repeat_max, repeat_count = cursor.fetchone()
    logger.info(f"Found {repeat_count:,} repeat donor relationships in donor_totals")
    
    # Sample using ID ranges (much faster than ORDER BY RAND())
    repeat_pairs = []
    if repeat_count and repeat_count > 0:
        # Sample in chunks across the ID range
        chunk_size = sample_size // 10
        id_range = (repeat_max - repeat_min) if repeat_max and repeat_min else 1
        
        for i in range(10):
            chunk_start = repeat_min + (i * id_range // 10)
            chunk_end = repeat_min + ((i + 1) * id_range // 10)
            
            cursor.execute(f"""
                SELECT contributor_id, cmte_id 
                FROM donor_totals 
                WHERE first_donation_date < '{cutoff_date}' 
                  AND last_donation_date >= '{cutoff_date}'
                  AND id >= {chunk_start} AND id < {chunk_end}
                LIMIT {chunk_size}
            """)
            chunk_pairs = cursor.fetchall()
            repeat_pairs.extend(chunk_pairs)
            logger.info(f"  Chunk {i+1}/10: {len(chunk_pairs):,} repeat pairs")
        
        repeat_pairs = repeat_pairs[:sample_size]
    
    logger.info(f"Sampled {len(repeat_pairs):,} repeat donor-committee pairs")
    
    # === STEP 2: Find LAPSED donors using donor_totals (FAST!) ===
    # last_donation_date < cutoff = only donated before cutoff, not after
    logger.info("Finding lapsed donor-committee pairs (using donor_totals)...")
    
    cursor.execute(f"""
        SELECT MIN(id), MAX(id), COUNT(*) 
        FROM donor_totals 
        WHERE last_donation_date < '{cutoff_date}'
    """)
    lapsed_min, lapsed_max, lapsed_count = cursor.fetchone()
    logger.info(f"Found {lapsed_count:,} lapsed donor relationships in donor_totals")
    
    lapsed_pairs = []
    if lapsed_count and lapsed_count > 0:
        chunk_size = sample_size // 10
        id_range = (lapsed_max - lapsed_min) if lapsed_max and lapsed_min else 1
        
        for i in range(10):
            chunk_start = lapsed_min + (i * id_range // 10)
            chunk_end = lapsed_min + ((i + 1) * id_range // 10)
            
            cursor.execute(f"""
                SELECT contributor_id, cmte_id 
                FROM donor_totals 
                WHERE last_donation_date < '{cutoff_date}'
                  AND id >= {chunk_start} AND id < {chunk_end}
                LIMIT {chunk_size}
            """)
            chunk_pairs = cursor.fetchall()
            lapsed_pairs.extend(chunk_pairs)
            logger.info(f"  Chunk {i+1}/10: {len(chunk_pairs):,} lapsed pairs")
        
        lapsed_pairs = lapsed_pairs[:sample_size]
    
    logger.info(f"Sampled {len(lapsed_pairs):,} lapsed donor-committee pairs")
    
    # === STEP 3: Compute PRE-CUTOFF features ===
    # For LAPSED donors: use donor_totals directly (all their data is pre-cutoff anyway)
    # For REPEAT donors: compute from contributions table (to exclude post-cutoff data)
    logger.info("Computing point-in-time features...")
    
    # LAPSED: Get features directly from donor_totals (safe - all data is pre-cutoff)
    # Use PARALLEL workers for speed
    logger.info("  Getting features for lapsed donors from donor_totals (parallel)...")
    
    def fetch_lapsed_chunk(chunk_pairs, cutoff):
        """Worker function to fetch lapsed features."""
        if not chunk_pairs:
            return pd.DataFrame()
        
        conn_local = get_db_connection()
        try:
            # Use contributor_id IN (...) which is much faster than OR chains
            contrib_ids = list(set(p[0] for p in chunk_pairs))
            contrib_in = ','.join(str(c) for c in contrib_ids)
            
            # Build a set for filtering
            pairs_set = set((int(p[0]), str(p[1])) for p in chunk_pairs)
            
            query = f"""
                SELECT 
                    contributor_id, cmte_id,
                    total_donations as cc_total_donations,
                    total_amount as cc_total_amount,
                    avg_amount as cc_avg_amount,
                    COALESCE(stddev_amount, 0) as cc_stddev_amount,
                    DATEDIFF('{cutoff}', last_donation_date) as cc_days_since_last,
                    COALESCE(donation_span_days, 0) as cc_donation_span,
                    COALESCE(avg_days_between_donations, 0) as cc_avg_frequency,
                    COALESCE(donations_primary, 0) as cc_donations_primary,
                    COALESCE(donations_general, 0) as cc_donations_general,
                    is_recurring as cc_is_recurring,
                    avg_amount as target_amount
                FROM donor_totals
                WHERE contributor_id IN ({contrib_in})
                  AND last_donation_date < '{cutoff}'
            """
            df = pd.read_sql(query, conn_local)
            # Filter to exact pairs we want
            df = df[df.apply(lambda r: (int(r['contributor_id']), str(r['cmte_id'])) in pairs_set, axis=1)]
            df['donated_after_cutoff'] = 0
            return df
        finally:
            conn_local.close()
    
    if lapsed_pairs:
        lapsed_dfs = []
        chunk_size = 5000  # Smaller chunks for parallel
        chunks = [lapsed_pairs[i:i+chunk_size] for i in range(0, len(lapsed_pairs), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=48) as executor:
            futures = {executor.submit(fetch_lapsed_chunk, chunk, cutoff_date): i 
                       for i, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    if len(result) > 0:
                        lapsed_dfs.append(result)
                    if chunk_idx % 10 == 0:
                        logger.info(f"    Lapsed chunk {chunk_idx+1}/{len(chunks)}: {len(result)} rows")
                except Exception as e:
                    logger.warning(f"    Chunk {chunk_idx} failed: {e}")
        
        lapsed_features_df = pd.concat(lapsed_dfs, ignore_index=True) if lapsed_dfs else pd.DataFrame()
        logger.info(f"  Got {len(lapsed_features_df):,} lapsed donor features")
    else:
        lapsed_features_df = pd.DataFrame()
    
    # REPEAT: Compute from contributions (pre-cutoff only) to avoid leakage
    # Use PARALLEL workers for speed
    logger.info("  Computing pre-cutoff features for repeat donors from contributions (parallel)...")
    
    def fetch_repeat_chunk(chunk_pairs, cutoff):
        """Worker function to fetch repeat donor features from contributions."""
        if not chunk_pairs:
            return pd.DataFrame()
        
        conn_local = get_db_connection()
        try:
            # Use contributor_id IN (...) for faster query
            contrib_ids = list(set(p[0] for p in chunk_pairs))
            contrib_in = ','.join(str(c) for c in contrib_ids)
            
            # Build a set for filtering
            pairs_set = set((int(p[0]), str(p[1])) for p in chunk_pairs)
            
            # Get pre-cutoff features
            query = f"""
                SELECT 
                    contributor_id, cmte_id,
                    COUNT(*) as cc_total_donations,
                    SUM(transaction_amt) as cc_total_amount,
                    AVG(transaction_amt) as cc_avg_amount,
                    COALESCE(STDDEV(transaction_amt), 0) as cc_stddev_amount,
                    DATEDIFF('{cutoff}', MAX(transaction_dt)) as cc_days_since_last,
                    DATEDIFF(MAX(transaction_dt), MIN(transaction_dt)) as cc_donation_span,
                    CASE WHEN COUNT(*) > 1 
                         THEN DATEDIFF(MAX(transaction_dt), MIN(transaction_dt)) / (COUNT(*) - 1)
                         ELSE 0 END as cc_avg_frequency,
                    SUM(CASE WHEN transaction_pgi = 'P' THEN 1 ELSE 0 END) as cc_donations_primary,
                    SUM(CASE WHEN transaction_pgi = 'G' THEN 1 ELSE 0 END) as cc_donations_general,
                    CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END as cc_is_recurring
                FROM contributions
                WHERE contributor_id IN ({contrib_in})
                  AND transaction_dt < '{cutoff}'
                GROUP BY contributor_id, cmte_id
            """
            df = pd.read_sql(query, conn_local)
            
            # Get FIRST donation amount AFTER cutoff (this is what we're predicting - NO LEAKAGE)
            target_query = f"""
                SELECT contributor_id, cmte_id, transaction_amt as target_amount
                FROM (
                    SELECT contributor_id, cmte_id, transaction_amt,
                           ROW_NUMBER() OVER (PARTITION BY contributor_id, cmte_id ORDER BY transaction_dt) as rn
                    FROM contributions
                    WHERE contributor_id IN ({contrib_in})
                      AND transaction_dt >= '{cutoff}'
                ) ranked
                WHERE rn = 1
            """
            target_df = pd.read_sql(target_query, conn_local)
            df = df.merge(target_df, on=['contributor_id', 'cmte_id'], how='inner')
            # Filter to exact pairs we want
            df = df[df.apply(lambda r: (int(r['contributor_id']), str(r['cmte_id'])) in pairs_set, axis=1)]
            df['donated_after_cutoff'] = 1
            return df
        finally:
            conn_local.close()
    
    if repeat_pairs:
        repeat_dfs = []
        chunk_size = 2000  # Smaller chunks since contributions query is heavier
        chunks = [repeat_pairs[i:i+chunk_size] for i in range(0, len(repeat_pairs), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=48) as executor:
            futures = {executor.submit(fetch_repeat_chunk, chunk, cutoff_date): i 
                       for i, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    if len(result) > 0:
                        repeat_dfs.append(result)
                    if chunk_idx % 10 == 0:
                        logger.info(f"    Repeat chunk {chunk_idx+1}/{len(chunks)}: {len(result)} rows")
                except Exception as e:
                    logger.warning(f"    Chunk {chunk_idx} failed: {e}")
        
        repeat_features_df = pd.concat(repeat_dfs, ignore_index=True) if repeat_dfs else pd.DataFrame()
        logger.info(f"  Got {len(repeat_features_df):,} repeat donor features")
    else:
        repeat_features_df = pd.DataFrame()
    
    # Combine
    features_df = pd.concat([lapsed_features_df, repeat_features_df], ignore_index=True)
    logger.info(f"Total features computed: {len(features_df):,}")
    
    logger.info(f"Labels: {features_df['donated_after_cutoff'].sum():,} positives, "
                f"{(features_df['donated_after_cutoff']==0).sum():,} negatives")
    
    # === STEP 4: Add committee and donor geographic features (parallel) ===
    logger.info("Adding committee and donor features (parallel)...")
    
    def fetch_committee_info(cmte_chunk):
        """Fetch committee info for a chunk."""
        if not cmte_chunk:
            return pd.DataFrame()
        conn_local = get_db_connection()
        try:
            cmte_values = ','.join([f"'{c}'" for c in cmte_chunk])
            query = f"""
                SELECT 
                    cm.cmte_id,
                    cm.cmte_pty_affiliation as cmte_party,
                    cm.cmte_tp as cmte_type,
                    cm.cmte_st as cmte_state,
                    MAX(ca.cand_pty_affiliation) as cand_party,
                    CASE WHEN MAX(ca.cand_id) IS NOT NULL THEN 1 ELSE 0 END as has_candidate
                FROM committees cm
                LEFT JOIN candidate_committee_links ccl ON cm.cmte_id = ccl.cmte_id
                LEFT JOIN candidates ca ON ccl.cand_id = ca.cand_id
                WHERE cm.cmte_id IN ({cmte_values})
                GROUP BY cm.cmte_id, cm.cmte_pty_affiliation, cm.cmte_tp, cm.cmte_st
            """
            return pd.read_sql(query, conn_local)
        finally:
            conn_local.close()
    
    def fetch_contributor_info(contrib_chunk, max_retries=3):
        """Fetch contributor geographic and demographic info for a chunk with retry."""
        if not contrib_chunk:
            return pd.DataFrame()
        
        for attempt in range(max_retries):
            try:
                conn_local = get_db_connection()
                contrib_values = ','.join([str(c) for c in contrib_chunk])
                query = f"""
                    SELECT 
                        id as contributor_id,
                        state as donor_state,
                        LEFT(zip_code, 3) as donor_zip3,
                        COALESCE(occupation, 'UNKNOWN') as occupation,
                        COALESCE(employer, 'UNKNOWN') as employer
                    FROM contributors
                    WHERE id IN ({contrib_values})
                """
                result = pd.read_sql(query, conn_local)
                conn_local.close()
                return result
            except Exception as e:
                if conn_local:
                    try:
                        conn_local.close()
                    except:
                        pass
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch contributor chunk after {max_retries} attempts: {e}")
                    return pd.DataFrame()
    
    # Fetch committee info in parallel
    cmte_ids = features_df['cmte_id'].unique().tolist()
    cmte_chunks = [cmte_ids[i:i+2000] for i in range(0, len(cmte_ids), 2000)]
    
    cmte_dfs = []
    with ThreadPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(fetch_committee_info, chunk) for chunk in cmte_chunks]
        for future in as_completed(futures):
            result = future.result()
            if len(result) > 0:
                cmte_dfs.append(result)
    
    cmte_df = pd.concat(cmte_dfs, ignore_index=True) if cmte_dfs else pd.DataFrame()
    features_df = features_df.merge(cmte_df, on='cmte_id', how='left')
    logger.info(f"  Added committee info for {len(cmte_df):,} committees")
    
    # Fetch contributor info in parallel (reduced workers to avoid DB connection exhaustion)
    contrib_ids = features_df['contributor_id'].unique().tolist()
    contrib_chunks = [contrib_ids[i:i+25000] for i in range(0, len(contrib_ids), 25000)]  # Smaller chunks
    
    contrib_dfs = []
    with ThreadPoolExecutor(max_workers=16) as executor:  # Reduced from 48 to avoid connection issues
        futures = [executor.submit(fetch_contributor_info, chunk) for chunk in contrib_chunks]
        for future in as_completed(futures):
            result = future.result()
            if len(result) > 0:
                contrib_dfs.append(result)
    
    contrib_df = pd.concat(contrib_dfs, ignore_index=True) if contrib_dfs else pd.DataFrame()
    features_df = features_df.merge(contrib_df, on='contributor_id', how='left')
    logger.info(f"  Added donor info for {len(contrib_df):,} contributors")
    
    # Add same_state feature
    features_df['same_state'] = (features_df['donor_state'] == features_df['cmte_state']).astype(int)
    
    cursor.close()
    conn.close()
    
    # === STEP 5: Split into train/test ===
    # Shuffle and split 80/20
    features_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx].copy()
    test_df = features_df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(train_df):,} ({train_df['donated_after_cutoff'].sum():,} positives)")
    logger.info(f"Test: {len(test_df):,} ({test_df['donated_after_cutoff'].sum():,} positives)")
    
    return train_df, test_df


def prepare_features(df, scaler=None, label_encoders=None, fit=True):
    """Prepare features for modeling."""
    logger.info("Preparing features...")
    
    df = df.copy()
    
    # Encode categorical variables
    if label_encoders is None:
        label_encoders = {}
    
    categorical_cols = ['cmte_party', 'cmte_type', 'cmte_state', 'cand_party', 'donor_state', 'donor_zip3', 'occupation', 'employer']
    
    # Clean occupation and employer - group rare values
    if 'occupation' in df.columns:
        # Normalize common variations
        df['occupation'] = df['occupation'].fillna('UNKNOWN').astype(str).str.strip()
        df['occupation'] = df['occupation'].replace({
            'NOT EMPLOYED': 'NOT EMPLOYED', 'SELF-EMPLOYED': 'SELF-EMPLOYED',
            'SELF EMPLOYED': 'SELF-EMPLOYED', 
            'INFORMATION REQUESTED': 'UNKNOWN', 'NONE': 'UNKNOWN', 'N/A': 'UNKNOWN', '': 'UNKNOWN'
        })
        # Keep top occupations, group others
        top_occupations = df['occupation'].value_counts().head(100).index.tolist()
        df['occupation'] = df['occupation'].apply(lambda x: x if x in top_occupations else 'OTHER')
    
    if 'employer' in df.columns:
        df['employer'] = df['employer'].fillna('UNKNOWN').astype(str).str.strip()
        df['employer'] = df['employer'].replace({
            'SELF-EMPLOYED': 'SELF-EMPLOYED', 'SELF EMPLOYED': 'SELF-EMPLOYED',
            'SELF': 'SELF-EMPLOYED', 'NONE': 'UNKNOWN', 'N/A': 'UNKNOWN', 
            'INFORMATION REQUESTED': 'UNKNOWN', '': 'UNKNOWN'
        })
        # Keep top employers, group others
        top_employers = df['employer'].value_counts().head(100).index.tolist()
        df['employer'] = df['employer'].apply(lambda x: x if x in top_employers else 'OTHER')
    
    for col in categorical_cols:
        df[col] = df[col].fillna('UNKNOWN').astype(str)
        if fit:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col + '_encoded'] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    
    # Feature columns - all computed from PRE-CUTOFF data
    # NOTE: cc_avg_amount and cc_total_amount REMOVED to prevent leakage in amount prediction
    feature_cols = [
        # Contributor-Committee relationship (from pre-cutoff only)
        'cc_total_donations',  # Count of donations (not amount)
        'cc_days_since_last', 'cc_donation_span', 'cc_avg_frequency',
        'cc_is_recurring', 'cc_stddev_amount',
        'cc_donations_primary', 'cc_donations_general',
        # Committee features (static - safe)
        'cmte_party_encoded', 'cmte_type_encoded', 'cmte_state_encoded',
        'cand_party_encoded', 'has_candidate',
        # Donor geographic features (static - safe)
        'donor_state_encoded', 'donor_zip3_encoded', 'same_state',
        # Donor demographic features (static - safe)
        'occupation_encoded', 'employer_encoded'
    ]
    
    # Fill NAs
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    X = df[feature_cols].values
    y_class = df['donated_after_cutoff'].values
    y_amount = df['target_amount'].fillna(0).values
    
    # Scale features
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y_class, y_amount, feature_cols, scaler, label_encoders


def train_classifier(X_train, y_train, X_test=None, y_test=None, n_jobs=-1):
    """Train likelihood classifier with progressive visualization."""
    logger.info("Training likelihood classifier (RandomForest)...")
    
    # Train progressively to show improvement
    tree_stages = [1, 10, 50, 200]
    stage_results = []
    
    clf = RandomForestClassifier(
        n_estimators=1,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=n_jobs,
        random_state=42,
        warm_start=True,
        verbose=0
    )
    
    for n_trees in tree_stages:
        clf.n_estimators = n_trees
        clf.fit(X_train, y_train)
        
        # Calculate metrics at this stage
        train_acc = clf.score(X_train, y_train)
        if X_test is not None and y_test is not None:
            test_acc = clf.score(X_test, y_test)
            test_proba = clf.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, test_proba)
        else:
            test_acc = None
            test_auc = None
        
        stage_results.append({
            'n_trees': n_trees,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_auc': test_auc
        })
        logger.info(f"  Trees: {n_trees:3d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")
    
    # Generate forest progression visualization
    if X_test is not None:
        generate_forest_progression_viz(clf, stage_results, X_test, y_test)
    
    return clf


def generate_forest_progression_viz(clf, stage_results, X_test, y_test):
    """Generate visualization showing how the forest improves with more trees."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Random Forest: How More Trees Improve Predictions', fontsize=16, fontweight='bold')
    
    # 1. Accuracy vs Number of Trees
    ax1 = axes[0, 0]
    n_trees = [r['n_trees'] for r in stage_results]
    train_accs = [r['train_acc'] for r in stage_results]
    test_accs = [r['test_acc'] for r in stage_results]
    test_aucs = [r['test_auc'] for r in stage_results]
    
    ax1.plot(n_trees, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=10)
    ax1.plot(n_trees, test_accs, 'g-s', label='Test Accuracy', linewidth=2, markersize=10)
    ax1.plot(n_trees, test_aucs, 'r-^', label='Test ROC-AUC', linewidth=2, markersize=10)
    ax1.set_xlabel('Number of Trees', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Model Performance vs. Forest Size', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xscale('log')
    ax1.set_xticks(n_trees)
    ax1.set_xticklabels([str(n) for n in n_trees])
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # 2. Prediction confidence distribution at different stages
    ax2 = axes[0, 1]
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
    
    # Get predictions at final stage
    proba = clf.predict_proba(X_test)[:, 1]
    
    for i, (n_tree, color) in enumerate(zip([1, 10, 50, 200], colors)):
        # Simulate what predictions would look like with fewer trees
        if n_tree < 200:
            # Use subset of trees
            subset_proba = np.mean([tree.predict_proba(X_test)[:, 1] 
                                    for tree in clf.estimators_[:n_tree]], axis=0)
        else:
            subset_proba = proba
        
        ax2.hist(subset_proba, bins=30, alpha=0.5, label=f'{n_tree} trees', color=color, density=True)
    
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision threshold')
    
    # 3. Single tree vs Full forest decision boundary (using 2 most important features)
    ax3 = axes[1, 0]
    
    # Get feature importances
    importances = clf.feature_importances_
    top2_idx = np.argsort(importances)[-2:]
    
    # Single tree prediction
    single_tree = clf.estimators_[0]
    single_pred = single_tree.predict(X_test)
    single_correct = (single_pred == y_test)
    
    ax3.scatter(X_test[single_correct, top2_idx[0]], X_test[single_correct, top2_idx[1]], 
                c='green', alpha=0.3, s=20, label='Correct')
    ax3.scatter(X_test[~single_correct, top2_idx[0]], X_test[~single_correct, top2_idx[1]], 
                c='red', alpha=0.3, s=20, label='Wrong')
    ax3.set_xlabel('Top Feature 1 (scaled)', fontsize=12)
    ax3.set_ylabel('Top Feature 2 (scaled)', fontsize=12)
    ax3.set_title(f'Single Tree: {single_correct.mean():.1%} Accuracy', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # 4. Full forest prediction
    ax4 = axes[1, 1]
    forest_pred = clf.predict(X_test)
    forest_correct = (forest_pred == y_test)
    
    ax4.scatter(X_test[forest_correct, top2_idx[0]], X_test[forest_correct, top2_idx[1]], 
                c='green', alpha=0.3, s=20, label='Correct')
    ax4.scatter(X_test[~forest_correct, top2_idx[0]], X_test[~forest_correct, top2_idx[1]], 
                c='red', alpha=0.3, s=20, label='Wrong')
    ax4.set_xlabel('Top Feature 1 (scaled)', fontsize=12)
    ax4.set_ylabel('Top Feature 2 (scaled)', fontsize=12)
    ax4.set_title(f'Full Forest (200 trees): {forest_correct.mean():.1%} Accuracy', fontsize=12, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'forest_progression.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {REPORT_DIR / 'forest_progression.png'}")


def train_regressor(X_train, y_train, y_class_train):
    """Train amount regressor (only on positive samples)."""
    logger.info("Training amount regressor (GradientBoosting)...")
    
    # Filter to only positive samples
    mask = y_class_train == 1
    X_pos = X_train[mask]
    y_pos = y_train[mask]
    
    if len(y_pos) == 0:
        logger.warning("No positive samples for regression!")
        return None
    
    logger.info(f"Training on {len(y_pos):,} positive samples")
    
    reg = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        learning_rate=0.1,
        random_state=42,
        verbose=1
    )
    reg.fit(X_pos, y_pos)
    return reg


def evaluate_models(clf, reg, X_test, y_class_test, y_amount_test):
    """Evaluate both models."""
    results = {}
    
    # Classification metrics
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_class = clf.predict(X_test)
    
    results['classification'] = {
        'roc_auc': roc_auc_score(y_class_test, y_pred_proba),
        'precision': precision_score(y_class_test, y_pred_class),
        'recall': recall_score(y_class_test, y_pred_class),
        'f1': f1_score(y_class_test, y_pred_class)
    }
    
    logger.info("Classification Results:")
    logger.info(f"  ROC-AUC: {results['classification']['roc_auc']:.4f}")
    logger.info(f"  Precision: {results['classification']['precision']:.4f}")
    logger.info(f"  Recall: {results['classification']['recall']:.4f}")
    logger.info(f"  F1: {results['classification']['f1']:.4f}")
    
    # Regression metrics (only on actual positives in test)
    if reg is not None:
        mask = y_class_test == 1
        if mask.sum() > 0:
            X_pos = X_test[mask]
            y_pos = y_amount_test[mask]
            y_pred_amount = reg.predict(X_pos)
            
            results['regression'] = {
                'mae': mean_absolute_error(y_pos, y_pred_amount),
                'rmse': np.sqrt(mean_squared_error(y_pos, y_pred_amount)),
                'r2': r2_score(y_pos, y_pred_amount)
            }
            
            logger.info("Regression Results (on positive samples):")
            logger.info(f"  MAE: ${results['regression']['mae']:.2f}")
            logger.info(f"  RMSE: ${results['regression']['rmse']:.2f}")
            logger.info(f"  R2: {results['regression']['r2']:.4f}")
    
    return results


def generate_training_visualizations(clf, reg, X_train, y_train, X_test, y_test, 
                                      y_amount_test, feature_cols):
    """Generate visualizations during training to understand model behavior."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating training visualizations...")
    
    # Get predictions
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_class = clf.predict(X_test)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Feature Importance
    ax1 = fig.add_subplot(2, 3, 1)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax1.barh(range(len(indices)), importances[indices], color=colors)
    ax1.set_yticks(range(len(indices)))
    ax1.set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
    ax1.set_xlabel('Importance')
    ax1.set_title('Feature Importance (Top 15)', fontsize=12, fontweight='bold')
    
    # 2. ROC Curve
    ax2 = fig.add_subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(2, 3, 3)
    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Lapsed', 'Repeat'], yticklabels=['Lapsed', 'Repeat'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # 4. Prediction Distribution
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='Lapsed (actual)', color='red', density=True)
    ax4.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Repeat (actual)', color='green', density=True)
    ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Density')
    ax4.set_title('Prediction Distribution by Class', fontsize=12, fontweight='bold')
    ax4.legend()
    
    # 5. Calibration - Predicted vs Actual Rate
    ax5 = fig.add_subplot(2, 3, 5)
    # Bin predictions and compute actual rate in each bin
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, 9)
    
    bin_means = []
    bin_actuals = []
    bin_counts = []
    for i in range(10):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(y_pred_proba[mask].mean())
            bin_actuals.append(y_test[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_means.append((bins[i] + bins[i+1]) / 2)
            bin_actuals.append(0)
            bin_counts.append(0)
    
    ax5.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax5.scatter(bin_means, bin_actuals, s=[c/10 for c in bin_counts], alpha=0.7, c='steelblue')
    ax5.plot(bin_means, bin_actuals, 'o-', color='steelblue', label='Model')
    ax5.set_xlabel('Mean Predicted Probability')
    ax5.set_ylabel('Actual Positive Rate')
    ax5.set_title('Calibration Plot', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    
    # 6. Sample Tree Visualization (simplified)
    ax6 = fig.add_subplot(2, 3, 6)
    # Get one tree from the forest
    sample_tree = clf.estimators_[0]
    plot_tree(sample_tree, max_depth=3, feature_names=feature_cols, 
              class_names=['Lapsed', 'Repeat'], filled=True, rounded=True,
              fontsize=7, ax=ax6)
    ax6.set_title('Sample Decision Tree (depth 3)', fontsize=12, fontweight='bold')
    
    plt.suptitle('RandomForest Training Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'training_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {REPORT_DIR / 'training_analysis.png'}")
    
    # Additional plot: Feature distributions by class
    fig2, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get top features
    top_features_idx = np.argsort(importances)[-12:]
    
    # Create DataFrame for plotting
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    X_test_df['class'] = y_test
    
    for i, feat_idx in enumerate(top_features_idx):
        ax = axes[i]
        feat_name = feature_cols[feat_idx]
        
        # Plot distribution for each class
        for label, color, name in [(0, 'red', 'Lapsed'), (1, 'green', 'Repeat')]:
            data = X_test_df[X_test_df['class'] == label][feat_name]
            ax.hist(data, bins=30, alpha=0.5, label=name, color=color, density=True)
        
        ax.set_xlabel(feat_name, fontsize=9)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
    
    plt.suptitle('Feature Distributions: Lapsed vs Repeat Donors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {REPORT_DIR / 'feature_distributions.png'}")
    
    # Additional plot: Amount prediction (if regressor exists)
    if reg is not None:
        fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Filter to positive samples
        mask = y_test == 1
        if mask.sum() > 0:
            X_pos = X_test[mask]
            y_actual = y_amount_test[mask]
            y_pred_amt = reg.predict(X_pos)
            
            # Scatter plot
            ax = axes[0]
            max_val = min(np.percentile(y_actual, 95), np.percentile(y_pred_amt, 95), 5000)
            ax.scatter(y_actual, y_pred_amt, alpha=0.3, s=10)
            ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect')
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_xlabel('Actual Amount ($)')
            ax.set_ylabel('Predicted Amount ($)')
            ax.set_title('Actual vs Predicted Amount', fontweight='bold')
            ax.legend()
            
            # Residuals
            ax = axes[1]
            residuals = y_pred_amt - y_actual
            ax.hist(residuals, bins=50, color='coral', edgecolor='white', alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='--')
            ax.set_xlabel('Residual (Predicted - Actual)')
            ax.set_ylabel('Count')
            ax.set_title('Residuals Distribution', fontweight='bold')
            
            # Prediction vs Actual distribution
            ax = axes[2]
            ax.hist(y_actual, bins=50, alpha=0.5, label='Actual', color='blue', density=True)
            ax.hist(y_pred_amt, bins=50, alpha=0.5, label='Predicted', color='orange', density=True)
            ax.set_xlabel('Amount ($)')
            ax.set_ylabel('Density')
            ax.set_title('Amount Distribution', fontweight='bold')
            ax.legend()
        
        plt.suptitle('Amount Prediction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'amount_prediction_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {REPORT_DIR / 'amount_prediction_analysis.png'}")
    
    logger.info("Training visualizations complete!")


def save_models(clf, reg, scaler, label_encoders, feature_cols, results):
    """Save all model artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        'classifier': clf,
        'regressor': reg,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'results': results
    }
    
    model_path = MODEL_DIR / 'donation_likelihood_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    logger.info(f"Models saved to {model_path}")
    return model_path


class DonationLikelihoodPredictor:
    """Predictor class for making predictions on new data."""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_DIR / 'donation_likelihood_model.pkl'
        
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.classifier = artifacts['classifier']
        self.regressor = artifacts['regressor']
        self.scaler = artifacts['scaler']
        self.label_encoders = artifacts['label_encoders']
        self.feature_cols = artifacts['feature_cols']
    
    def predict(self, contributor_id, cmte_id):
        """
        Predict likelihood and amount for a contributor-committee pair.
        
        Returns:
            dict with 'likelihood' (0-1) and 'predicted_amount'
        """
        conn = get_db_connection()
        
        # Get pre-cutoff features from contributions
        query = """
            SELECT 
                COUNT(*) as cc_total_donations,
                SUM(transaction_amt) as cc_total_amount,
                AVG(transaction_amt) as cc_avg_amount,
                STDDEV(transaction_amt) as cc_stddev_amount,
                DATEDIFF(CURDATE(), MAX(transaction_dt)) as cc_days_since_last,
                DATEDIFF(MAX(transaction_dt), MIN(transaction_dt)) as cc_donation_span,
                CASE WHEN COUNT(*) > 1 
                     THEN DATEDIFF(MAX(transaction_dt), MIN(transaction_dt)) / (COUNT(*) - 1)
                     ELSE 0 END as cc_avg_frequency,
                SUM(CASE WHEN transaction_pgi = 'P' THEN 1 ELSE 0 END) as cc_donations_primary,
                SUM(CASE WHEN transaction_pgi = 'G' THEN 1 ELSE 0 END) as cc_donations_general,
                CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END as cc_is_recurring
            FROM contributions
            WHERE contributor_id = %s AND cmte_id = %s
        """
        
        df = pd.read_sql(query, conn, params=(contributor_id, cmte_id))
        
        if len(df) == 0 or df['cc_total_donations'].iloc[0] == 0:
            conn.close()
            return {'likelihood': 0.1, 'predicted_amount': 0, 'note': 'No prior relationship'}
        
        # Get committee and donor info
        cmte_query = """
            SELECT 
                cm.cmte_pty_affiliation as cmte_party,
                cm.cmte_tp as cmte_type,
                cm.cmte_st as cmte_state,
                MAX(ca.cand_pty_affiliation) as cand_party,
                CASE WHEN MAX(ca.cand_id) IS NOT NULL THEN 1 ELSE 0 END as has_candidate
            FROM committees cm
            LEFT JOIN candidate_committee_links ccl ON cm.cmte_id = ccl.cmte_id
            LEFT JOIN candidates ca ON ccl.cand_id = ca.cand_id
            WHERE cm.cmte_id = %s
            GROUP BY cm.cmte_id, cm.cmte_pty_affiliation, cm.cmte_tp, cm.cmte_st
        """
        cmte_df = pd.read_sql(cmte_query, conn, params=(cmte_id,))
        
        contrib_query = """
            SELECT state as donor_state, LEFT(zip_code, 3) as donor_zip3,
                   COALESCE(occupation, 'UNKNOWN') as occupation,
                   COALESCE(employer, 'UNKNOWN') as employer
            FROM contributors WHERE id = %s
        """
        contrib_df = pd.read_sql(contrib_query, conn, params=(contributor_id,))
        conn.close()
        
        # Merge info
        for col in cmte_df.columns:
            df[col] = cmte_df[col].iloc[0] if len(cmte_df) > 0 else None
        for col in contrib_df.columns:
            df[col] = contrib_df[col].iloc[0] if len(contrib_df) > 0 else None
        
        df['same_state'] = 1 if df['donor_state'].iloc[0] == df['cmte_state'].iloc[0] else 0
        
        # Encode categoricals
        for col in ['cmte_party', 'cmte_type', 'cmte_state', 'cand_party', 'donor_state', 'donor_zip3', 'occupation', 'employer']:
            val = str(df[col].iloc[0]) if df[col].iloc[0] else 'UNKNOWN'
            if col in self.label_encoders and val in self.label_encoders[col].classes_:
                df[col + '_encoded'] = self.label_encoders[col].transform([val])[0]
            else:
                df[col + '_encoded'] = 0
        
        # Build feature vector
        X = df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        # Predict
        likelihood = self.classifier.predict_proba(X_scaled)[0, 1]
        predicted_amount = self.regressor.predict(X_scaled)[0] if self.regressor else 0
        
        return {
            'likelihood': float(likelihood),
            'predicted_amount': float(max(0, predicted_amount))
        }


def main():
    parser = argparse.ArgumentParser(description='Train donation likelihood model')
    parser.add_argument('--sample-size', type=int, default=100000,
                        help='Number of samples per class')
    parser.add_argument('--cutoff-date', type=str, default='2024-01-01',
                        help='Temporal split cutoff date (YYYY-MM-DD)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for training')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DONATION LIKELIHOOD MODEL TRAINING (FIXED)")
    print("=" * 60)
    print(f"Sample size: {args.sample_size:,} per class")
    print(f"Temporal cutoff: {args.cutoff_date}")
    print(f"Parallel jobs: {args.n_jobs}")
    print()
    print("DATA SETUP (NO LEAKAGE):")
    print("  - POSITIVES: Donated BEFORE and AFTER cutoff (repeat donors)")
    print("  - NEGATIVES: Donated BEFORE but NOT AFTER cutoff (lapsed donors)")
    print("  - Features computed from PRE-CUTOFF data only")
    print()
    
    start_time = time.time()
    
    # Fetch data with fixed temporal split
    train_df, test_df = fetch_training_data_fixed(
        sample_size=args.sample_size, 
        cutoff_date=args.cutoff_date
    )
    
    # Prepare features
    X_train, y_class_train, y_amount_train, feature_cols, scaler, label_encoders = prepare_features(
        train_df, fit=True
    )
    
    X_test, y_class_test, y_amount_test, _, _, _ = prepare_features(
        test_df, scaler=scaler, label_encoders=label_encoders, fit=False
    )
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    logger.info(f"Features: {len(feature_cols)}")
    
    # Train models
    clf = train_classifier(X_train, y_class_train, X_test, y_class_test, args.n_jobs)
    reg = train_regressor(X_train, y_amount_train, y_class_train)
    
    # Evaluate
    eval_results = evaluate_models(clf, reg, X_test, y_class_test, y_amount_test)
    
    # Generate visualizations
    generate_training_visualizations(
        clf, reg, X_train, y_class_train, X_test, y_class_test, 
        y_amount_test, feature_cols
    )
    
    # Feature importance
    logger.info("\nFeature Importance (top 10):")
    importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances[:10]:
        logger.info(f"  {feat}: {imp:.4f}")
    
    # Save
    model_path = save_models(clf, reg, scaler, label_encoders, feature_cols, eval_results)
    
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"TRAINING COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
