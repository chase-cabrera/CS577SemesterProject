"""
Supervised Learning - Stage 2: Party-Specific Donation Amount Prediction
Train 6 separate regression models (one per party) to predict donation amounts
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_training_data import load_data_splits, prepare_supervised_learning_data
from feature_engineering import PARTIES


def get_models_dir():
    """Get absolute path to models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    return os.path.join(project_root, 'models')


def compute_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error
    Handles zero values gracefully
    """
    # Filter out zero values
    mask = y_true != 0
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_ridge_regression(X_train, y_train, X_val, y_val, party):
    """
    Train Ridge Regression model (baseline with L2 regularization)
    """
    print(f"\n  Training Ridge Regression for {party}...")
    
    start_time = time.time()
    
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Clip negative predictions to 0
    y_train_pred = np.maximum(0, y_train_pred)
    y_val_pred = np.maximum(0, y_val_pred)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mape = compute_mape(y_val, y_val_pred)
    
    print(f"    Train MAE: ${train_mae:.2f}, Val MAE: ${val_mae:.2f}")
    print(f"    Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")
    
    metrics = {
        'model_name': 'Ridge Regression',
        'party': party,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'val_mape': val_mape,
        'train_time': train_time
    }
    
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val, party):
    """
    Train Random Forest Regressor
    """
    print(f"\n  Training Random Forest for {party}...")
    
    start_time = time.time()
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=25,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Clip negative predictions to 0
    y_train_pred = np.maximum(0, y_train_pred)
    y_val_pred = np.maximum(0, y_val_pred)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mape = compute_mape(y_val, y_val_pred)
    
    print(f"    Train MAE: ${train_mae:.2f}, Val MAE: ${val_mae:.2f}")
    print(f"    Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")
    
    metrics = {
        'model_name': 'Random Forest',
        'party': party,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'val_mape': val_mape,
        'train_time': train_time
    }
    
    return model, metrics


def train_gradient_boosting(X_train, y_train, X_val, y_val, party):
    """
    Train Gradient Boosting Regressor
    """
    print(f"\n  Training Gradient Boosting for {party}...")
    
    try:
        import lightgbm as lgb
        use_lightgbm = True
    except ImportError:
        use_lightgbm = False
    
    start_time = time.time()
    
    if use_lightgbm:
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Clip negative predictions to 0
    y_train_pred = np.maximum(0, y_train_pred)
    y_val_pred = np.maximum(0, y_val_pred)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mape = compute_mape(y_val, y_val_pred)
    
    print(f"    Train MAE: ${train_mae:.2f}, Val MAE: ${val_mae:.2f}")
    print(f"    Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")
    
    model_name = 'LightGBM' if use_lightgbm else 'Gradient Boosting'
    
    metrics = {
        'model_name': model_name,
        'party': party,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'val_mape': val_mape,
        'train_time': train_time
    }
    
    return model, metrics


def add_party_probabilities(X, y_party, party_model):
    """
    Add predicted party probabilities as features for Stage 2
    
    Args:
        X: Feature matrix
        y_party: Actual party labels
        party_model: Trained Stage 1 party classification model
        
    Returns:
        Enhanced feature matrix with party probabilities
    """
    # Get party probabilities
    party_probs = party_model.predict_proba(X)
    
    # Concatenate with existing features
    X_enhanced = np.hstack([X, party_probs])
    
    return X_enhanced


def train_party_model(party, X_train, y_train, X_val, y_val):
    """
    Train all models for a specific party
    
    Args:
        party: Party name (e.g., 'DEM', 'REP')
        X_train, y_train, X_val, y_val: Training and validation data
        
    Returns:
        Dictionary with all models and metrics for this party
    """
    print(f"\nTraining Models for {party}")
    print(f"Training samples: {len(y_train):,}")
    print(f"Non-zero donations: {(y_train > 0).sum():,} ({(y_train > 0).sum()/len(y_train)*100:.1f}%)")
    print(f"Validation samples: {len(y_val):,}")
    print(f"Non-zero donations: {(y_val > 0).sum():,} ({(y_val > 0).sum()/len(y_val)*100:.1f}%)")
    print(f"Mean donation: ${y_train[y_train > 0].mean():.2f}")
    
    all_models = {}
    all_metrics = []
    
    # Train Ridge
    ridge_model, ridge_metrics = train_ridge_regression(X_train, y_train, X_val, y_val, party)
    all_models['ridge'] = ridge_model
    all_metrics.append(ridge_metrics)
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val, party)
    all_models['random_forest'] = rf_model
    all_metrics.append(rf_metrics)
    
    # Train Gradient Boosting (optional - skip if dependencies missing)
    try:
        gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_val, y_val, party)
        all_models['gradient_boosting'] = gb_model
        all_metrics.append(gb_metrics)
    except Exception as e:
        print(f"\n  Gradient Boosting skipped for {party}: {str(e)[:80]}")
    
    # Select best model based on validation MAE
    best_metrics = min(all_metrics, key=lambda x: x['val_mae'])
    best_model_name = best_metrics['model_name']
    
    if 'Ridge' in best_model_name:
        best_model = all_models['ridge']
    elif 'Random' in best_model_name:
        best_model = all_models['random_forest']
    elif 'gradient_boosting' in all_models:
        best_model = all_models['gradient_boosting']
    else:
        # Default to Random Forest if gradient boosting unavailable
        best_model = all_models['random_forest']
        print(f"  Defaulting to Random Forest")
    
    print(f"\n  Best model for {party}: {best_model_name} (Val MAE: ${best_metrics['val_mae']:.2f})")
    
    return {
        'models': all_models,
        'metrics': all_metrics,
        'best_model': best_model,
        'best_metrics': best_metrics
    }


def save_party_models(party, models_dict, save_dir):
    """
    Save all models for a party
    """
    party_lower = party.lower()
    
    for model_key, model in models_dict['models'].items():
        save_path = os.path.join(save_dir, f'stage2_{party_lower}_{model_key}.pkl')
        
        # Find corresponding metrics
        metrics = [m for m in models_dict['metrics'] if model_key.replace('_', ' ').title() in m['model_name'] 
                   or m['model_name'].replace(' ', '_').lower() == model_key][0]
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'metrics': metrics,
                'party': party,
                'trained_at': datetime.now().isoformat()
            }, f)
    
    # Save best model separately
    best_save_path = os.path.join(save_dir, f'stage2_{party_lower}_best.pkl')
    with open(best_save_path, 'wb') as f:
        pickle.dump({
            'model': models_dict['best_model'],
            'metrics': models_dict['best_metrics'],
            'party': party,
            'trained_at': datetime.now().isoformat()
        }, f)
    
    print(f"  Models for {party} saved to {save_dir}")


def train_all_party_models(load_stage1=True):
    """
    Train Stage 2 models for all parties
    
    Args:
        load_stage1: If True, load Stage 1 model to add party probabilities as features
    """
    print("STAGE 2: PARTY-SPECIFIC DONATION AMOUNT PREDICTION")
    
    # Load data
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_data_splits()
    
    # Prepare features (full features)
    data = prepare_supervised_learning_data(
        train_df, val_df, test_df,
        include_history=True,
        save_dir=os.path.join(get_models_dir(), 'supervised')
    )
    
    X_train = data['X_train']
    X_val = data['X_val']
    party_amounts = data['party_amounts']
    
    # Optionally add Stage 1 party probabilities as features
    if load_stage1:
        print("\nLoading Stage 1 model to add party probabilities as features...")
        try:
            stage1_path = os.path.join(get_models_dir(), 'supervised', 'stage1_party_best_full.pkl')
            with open(stage1_path, 'rb') as f:
                stage1_data = pickle.load(f)
                stage1_model = stage1_data['model']
            
            # Add party probabilities
            X_train = add_party_probabilities(X_train, data['y_train_party'], stage1_model)
            X_val = add_party_probabilities(X_val, data['y_val_party'], stage1_model)
            
            print(f"Enhanced features shape: {X_train.shape}")
        except Exception as e:
            print(f"Could not load Stage 1 model: {e}")
            print("Proceeding without party probability features")
    
    # Train models for each party
    all_party_results = {}
    all_metrics_list = []
    
    for party in PARTIES:
        y_train = party_amounts[party]['train']
        y_val = party_amounts[party]['val']
        
        # Train models for this party
        party_results = train_party_model(party, X_train, y_train, X_val, y_val)
        all_party_results[party] = party_results
        all_metrics_list.extend(party_results['metrics'])
        
        # Save models
        save_dir = os.path.join(get_models_dir(), 'supervised')
        save_party_models(party, party_results, save_dir)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(all_metrics_list)
    
    print("\nSTAGE 2 MODEL COMPARISON - ALL PARTIES")
    print(comparison_df[['party', 'model_name', 'val_mae', 'val_rmse', 'val_r2', 'train_time']].to_string(index=False))
    
    # Save comparison
    comparison_path = '../../models/supervised/stage2_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    # Summary by party
    print("\nBEST MODELS BY PARTY")
    
    for party in PARTIES:
        best = all_party_results[party]['best_metrics']
        print(f"{party}: {best['model_name']} (MAE: ${best['val_mae']:.2f}, R2: {best['val_r2']:.4f})")
    
    print("\nSTAGE 2 TRAINING COMPLETE")
    
    return all_party_results, comparison_df


def main():
    """
    Main training script for Stage 2
    """
    train_all_party_models(load_stage1=True)


if __name__ == "__main__":
    main()
