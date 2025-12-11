"""
Model Comparison and Evaluation
Comprehensive evaluation comparing Supervised Learning vs K-Means + KNN approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
import pickle
import os
import sys
import time
import json

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'supervised_models'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clustering_models'))

from prepare_training_data import load_data_splits
from supervised_models.supervised_pipeline import SupervisedPredictionPipeline
from clustering_models.knn_pipeline import KNNPredictionPipeline
from feature_engineering import PARTIES as ALL_PARTIES

# Simplified party categories for comparison
PARTIES = ['DEM', 'REP', 'OTHER']
OTHER_PARTIES = ['LIB', 'GRE', 'IND', 'OTHER']  # These get grouped into OTHER


def map_to_simplified_party(party):
    """Map a party to DEM, REP, or OTHER"""
    if party in ['DEM', 'REP']:
        return party
    return 'OTHER'


def simplify_party_predictions(y_pred, y_true=None):
    """Convert party predictions to simplified categories"""
    y_pred_simplified = np.array([map_to_simplified_party(p) for p in y_pred])
    if y_true is not None:
        y_true_simplified = np.array([map_to_simplified_party(p) for p in y_true])
        return y_pred_simplified, y_true_simplified
    return y_pred_simplified


def aggregate_other_amounts(amount_preds):
    """Aggregate amount predictions for OTHER parties"""
    simplified_amounts = {
        'DEM': amount_preds['DEM'],
        'REP': amount_preds['REP'],
        'OTHER': sum(amount_preds[p] for p in OTHER_PARTIES if p in amount_preds)
    }
    return simplified_amounts


def evaluate_party_classification(y_true, y_pred, y_proba, model_name):
    """
    Evaluate party classification performance
    
    Returns:
        Dictionary with metrics
    """
    metrics = {'model_name': model_name}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    metrics['precision_weighted'] = precision
    metrics['recall_weighted'] = recall
    metrics['f1_weighted'] = f1
    
    # Per-party metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=PARTIES
    )
    
    for i, party in enumerate(PARTIES):
        metrics[f'{party}_precision'] = precision_per[i] if i < len(precision_per) else 0
        metrics[f'{party}_recall'] = recall_per[i] if i < len(recall_per) else 0
        metrics[f'{party}_f1'] = f1_per[i] if i < len(f1_per) else 0
        metrics[f'{party}_support'] = support_per[i] if i < len(support_per) else 0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=PARTIES)
    metrics['confusion_matrix'] = cm
    
    return metrics


def evaluate_amount_prediction(y_true, y_pred, party, model_name):
    """
    Evaluate donation amount prediction performance
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'model_name': model_name,
        'party': party
    }
    
    # Only evaluate on non-zero actual amounts
    mask = y_true > 0
    if mask.sum() == 0:
        return metrics
    
    y_true_nonzero = y_true[mask]
    y_pred_nonzero = y_pred[mask]
    
    # Overall metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Non-zero metrics
    metrics['mae_nonzero'] = mean_absolute_error(y_true_nonzero, y_pred_nonzero)
    metrics['rmse_nonzero'] = np.sqrt(mean_squared_error(y_true_nonzero, y_pred_nonzero))
    metrics['r2_nonzero'] = r2_score(y_true_nonzero, y_pred_nonzero)
    
    # MAPE
    mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
    metrics['mape'] = mape
    
    return metrics


def evaluate_supervised_model(test_df, save_dir='../report/model_comparison'):
    """
    Evaluate supervised learning models on test set
    """
    print("\nEVALUATING SUPERVISED LEARNING MODELS")
    
    # Load pipeline
    pipeline = SupervisedPredictionPipeline(use_demographics_only=False)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    start_time = time.time()
    predictions = pipeline.predict(test_df)
    pred_time = time.time() - start_time
    
    print(f"Prediction time: {pred_time:.2f}s for {len(test_df):,} contributors")
    print(f"Average per prediction: {pred_time/len(test_df)*1000:.2f}ms")
    
    # Extract predictions (using ALL_PARTIES from model output)
    if isinstance(predictions, list):
        y_pred_party_raw = np.array([p['primary_party'] for p in predictions])
        amount_preds_raw = {party: np.array([p['donation_amounts'][party] for p in predictions]) 
                       for party in ALL_PARTIES}
    else:
        y_pred_party_raw = np.array([predictions['primary_party']])
        amount_preds_raw = {party: np.array([predictions['donation_amounts'][party]]) 
                       for party in ALL_PARTIES}
    
    # Get true values and simplify to DEM, REP, OTHER
    y_true_party_raw = test_df['primary_party'].values
    y_pred_party, y_true_party = simplify_party_predictions(y_pred_party_raw, y_true_party_raw)
    
    # Aggregate amount predictions for OTHER
    amount_preds = aggregate_other_amounts(amount_preds_raw)
    
    # Create simplified party probabilities
    party_probs = np.zeros((len(y_pred_party), len(PARTIES)))
    
    # Evaluate party classification
    print("\nEvaluating party classification (DEM, REP, OTHER)...")
    party_metrics = evaluate_party_classification(
        y_true_party, y_pred_party, party_probs, 'Supervised Learning'
    )
    
    print(f"Accuracy: {party_metrics['accuracy']:.4f}")
    print(f"F1 Score (weighted): {party_metrics['f1_weighted']:.4f}")
    
    # Evaluate amount predictions
    print("\nEvaluating donation amount predictions...")
    amount_metrics = []
    for party in PARTIES:
        if party == 'OTHER':
            # Sum up other party amounts from test data
            y_true_amount = sum(test_df[f'{p.lower()}_amount'].values for p in OTHER_PARTIES 
                               if f'{p.lower()}_amount' in test_df.columns)
        else:
            party_col = f'{party.lower()}_amount'
            y_true_amount = test_df[party_col].values
        y_pred_amount = amount_preds[party]
        
        metrics = evaluate_amount_prediction(
            y_true_amount, y_pred_amount, party, 'Supervised Learning'
        )
        amount_metrics.append(metrics)
        
        print(f"{party}: MAE=${metrics['mae']:.2f}, R²={metrics['r2']:.4f}")
    
    results = {
        'party_metrics': party_metrics,
        'amount_metrics': amount_metrics,
        'prediction_time': pred_time,
        'avg_time_per_prediction': pred_time / len(test_df)
    }
    
    return results


def evaluate_knn_model(test_df, k=20, save_dir='../report/model_comparison'):
    """
    Evaluate KNN models on test set
    """
    print("\nEVALUATING K-MEANS + KNN MODELS")
    
    # Load pipeline
    pipeline = KNNPredictionPipeline()
    
    # Make predictions
    print(f"\nMaking predictions on test set (k={k})...")
    start_time = time.time()
    predictions = pipeline.predict(test_df, k=k)
    pred_time = time.time() - start_time
    
    print(f"Prediction time: {pred_time:.2f}s for {len(test_df):,} contributors")
    print(f"Average per prediction: {pred_time/len(test_df)*1000:.2f}ms")
    
    # Extract predictions (using ALL_PARTIES from model output)
    if isinstance(predictions, list):
        y_pred_party_raw = np.array([p['primary_party'] for p in predictions])
        amount_preds_raw = {party: np.array([p['donation_amounts'][party] for p in predictions]) 
                       for party in ALL_PARTIES}
    else:
        y_pred_party_raw = np.array([predictions['primary_party']])
        amount_preds_raw = {party: np.array([predictions['donation_amounts'][party]]) 
                       for party in ALL_PARTIES}
    
    # Get true values and simplify to DEM, REP, OTHER
    y_true_party_raw = test_df['primary_party'].values
    y_pred_party, y_true_party = simplify_party_predictions(y_pred_party_raw, y_true_party_raw)
    
    # Aggregate amount predictions for OTHER
    amount_preds = aggregate_other_amounts(amount_preds_raw)
    
    # Create simplified party probabilities
    party_probs = np.zeros((len(y_pred_party), len(PARTIES)))
    
    # Evaluate party classification
    print("\nEvaluating party classification (DEM, REP, OTHER)...")
    party_metrics = evaluate_party_classification(
        y_true_party, y_pred_party, party_probs, 'K-Means + KNN'
    )
    
    print(f"Accuracy: {party_metrics['accuracy']:.4f}")
    print(f"F1 Score (weighted): {party_metrics['f1_weighted']:.4f}")
    
    # Evaluate amount predictions
    print("\nEvaluating donation amount predictions...")
    amount_metrics = []
    for party in PARTIES:
        if party == 'OTHER':
            # Sum up other party amounts from test data
            y_true_amount = sum(test_df[f'{p.lower()}_amount'].values for p in OTHER_PARTIES 
                               if f'{p.lower()}_amount' in test_df.columns)
        else:
            party_col = f'{party.lower()}_amount'
            y_true_amount = test_df[party_col].values
        y_pred_amount = amount_preds[party]
        
        metrics = evaluate_amount_prediction(
            y_true_amount, y_pred_amount, party, 'K-Means + KNN'
        )
        amount_metrics.append(metrics)
        
        print(f"{party}: MAE=${metrics['mae']:.2f}, R²={metrics['r2']:.4f}")
    
    results = {
        'party_metrics': party_metrics,
        'amount_metrics': amount_metrics,
        'prediction_time': pred_time,
        'avg_time_per_prediction': pred_time / len(test_df)
    }
    
    return results


def create_comparison_visualizations(supervised_results, knn_results, 
                                     save_dir='../report/model_comparison'):
    """
    Create comparison visualizations
    """
    print("\nCREATING COMPARISON VISUALIZATIONS")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Party classification comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Party Classification Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    ax1 = axes[0]
    models = ['Supervised', 'KNN']
    accuracies = [
        supervised_results['party_metrics']['accuracy'],
        knn_results['party_metrics']['accuracy']
    ]
    bars = ax1.bar(models, accuracies, color=['steelblue', 'forestgreen'], alpha=0.7)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Overall Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11)
    
    # F1 score comparison
    ax2 = axes[1]
    f1_scores = [
        supervised_results['party_metrics']['f1_weighted'],
        knn_results['party_metrics']['f1_weighted']
    ]
    bars = ax2.bar(models, f1_scores, color=['steelblue', 'forestgreen'], alpha=0.7)
    ax2.set_ylabel('F1 Score (weighted)', fontsize=12)
    ax2.set_title('F1 Score', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11)
    
    # Inference speed comparison
    ax3 = axes[2]
    times = [
        supervised_results['avg_time_per_prediction'] * 1000,  # ms
        knn_results['avg_time_per_prediction'] * 1000
    ]
    bars = ax3.bar(models, times, color=['steelblue', 'forestgreen'], alpha=0.7)
    ax3.set_ylabel('Time (ms)', fontsize=12)
    ax3.set_title('Inference Speed (per prediction)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'party_classification_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # 2. Amount prediction comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Donation Amount Prediction Comparison', fontsize=16, fontweight='bold')
    
    # MAE comparison by party
    ax1 = axes[0]
    supervised_maes = [m['mae'] for m in supervised_results['amount_metrics']]
    knn_maes = [m['mae'] for m in knn_results['amount_metrics']]
    
    x = np.arange(len(PARTIES))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, supervised_maes, width, label='Supervised', 
                    color='steelblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, knn_maes, width, label='KNN',
                    color='forestgreen', alpha=0.7)
    
    ax1.set_xlabel('Party', fontsize=12)
    ax1.set_ylabel('MAE ($)', fontsize=12)
    ax1.set_title('Mean Absolute Error by Party', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(PARTIES)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # R² comparison by party
    ax2 = axes[1]
    supervised_r2 = [m['r2'] for m in supervised_results['amount_metrics']]
    knn_r2 = [m['r2'] for m in knn_results['amount_metrics']]
    
    bars1 = ax2.bar(x - width/2, supervised_r2, width, label='Supervised',
                    color='steelblue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, knn_r2, width, label='KNN',
                    color='forestgreen', alpha=0.7)
    
    ax2.set_xlabel('Party', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Score by Party', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(PARTIES)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'amount_prediction_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    print("\nVisualizations created")


def save_comparison_results(supervised_results, knn_results, save_dir='../report/model_comparison'):
    """
    Save comparison results to files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Party classification comparison
    party_comparison = pd.DataFrame([
        {
            'Model': 'Supervised Learning',
            'Accuracy': supervised_results['party_metrics']['accuracy'],
            'Precision': supervised_results['party_metrics']['precision_weighted'],
            'Recall': supervised_results['party_metrics']['recall_weighted'],
            'F1 Score': supervised_results['party_metrics']['f1_weighted'],
            'Avg Time (ms)': supervised_results['avg_time_per_prediction'] * 1000
        },
        {
            'Model': 'K-Means + KNN',
            'Accuracy': knn_results['party_metrics']['accuracy'],
            'Precision': knn_results['party_metrics']['precision_weighted'],
            'Recall': knn_results['party_metrics']['recall_weighted'],
            'F1 Score': knn_results['party_metrics']['f1_weighted'],
            'Avg Time (ms)': knn_results['avg_time_per_prediction'] * 1000
        }
    ])
    
    party_path = os.path.join(save_dir, 'party_classification_comparison.csv')
    party_comparison.to_csv(party_path, index=False)
    print(f"\nSaved party comparison: {party_path}")
    
    # Amount prediction comparison
    amount_comparison = []
    for i, party in enumerate(PARTIES):
        amount_comparison.append({
            'Party': party,
            'Model': 'Supervised Learning',
            'MAE': supervised_results['amount_metrics'][i]['mae'],
            'RMSE': supervised_results['amount_metrics'][i]['rmse'],
            'R²': supervised_results['amount_metrics'][i]['r2']
        })
        amount_comparison.append({
            'Party': party,
            'Model': 'K-Means + KNN',
            'MAE': knn_results['amount_metrics'][i]['mae'],
            'RMSE': knn_results['amount_metrics'][i]['rmse'],
            'R²': knn_results['amount_metrics'][i]['r2']
        })
    
    amount_df = pd.DataFrame(amount_comparison)
    amount_path = os.path.join(save_dir, 'amount_prediction_comparison.csv')
    amount_df.to_csv(amount_path, index=False)
    print(f"Saved amount comparison: {amount_path}")


def main():
    """
    Main comparison script
    """
    print("MODEL COMPARISON: SUPERVISED LEARNING VS K-MEANS + KNN")
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_df = load_data_splits()
    
    # Sample for faster testing
    # test_df = test_df.sample(min(10000, len(test_df)), random_state=42)
    print(f"Test set size: {len(test_df):,}")
    
    # Evaluate supervised model
    supervised_results = evaluate_supervised_model(test_df)
    
    # Evaluate KNN model
    knn_results = evaluate_knn_model(test_df)
    
    # Create visualizations
    create_comparison_visualizations(supervised_results, knn_results)
    
    # Save results
    save_comparison_results(supervised_results, knn_results)
    
    print("\nMODEL COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
