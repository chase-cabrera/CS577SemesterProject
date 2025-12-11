"""
Supervised Learning - Stage 1: Party Classification
Train multiple classification models to predict party affiliation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             confusion_matrix, classification_report, roc_auc_score)
import pickle
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_training_data import load_data_splits, prepare_supervised_learning_data
from feature_engineering import PARTIES


def get_models_dir():
    """Get absolute path to models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    return os.path.join(project_root, 'models')


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression model (baseline)
    
    Returns:
        Trained model and metrics
    """
    print("\nTraining Logistic Regression (Baseline)")
    
    start_time = time.time()
    
    # Multi-class logistic regression with balanced class weights
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, average='weighted'
    )
    
    print(f"Validation F1 (weighted): {f1:.4f}")
    print(f"Validation Precision (weighted): {precision:.4f}")
    print(f"Validation Recall (weighted): {recall:.4f}")
    
    metrics = {
        'model_name': 'Logistic Regression',
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_f1_weighted': f1,
        'val_precision_weighted': precision,
        'val_recall_weighted': recall,
        'train_time': train_time
    }
    
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest model
    
    Returns:
        Trained model and metrics
    """
    print("\nTraining Random Forest")
    
    start_time = time.time()
    
    # Random Forest with balanced class weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=100,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, average='weighted'
    )
    
    print(f"Validation F1 (weighted): {f1:.4f}")
    print(f"Validation Precision (weighted): {precision:.4f}")
    print(f"Validation Recall (weighted): {recall:.4f}")
    
    metrics = {
        'model_name': 'Random Forest',
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_f1_weighted': f1,
        'val_precision_weighted': precision,
        'val_recall_weighted': recall,
        'train_time': train_time
    }
    
    return model, metrics


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """
    Train Gradient Boosting model (LightGBM)
    Falls back to sklearn GradientBoosting if LightGBM not available
    
    Returns:
        Trained model and metrics
    """
    print("\nTraining Gradient Boosting")
    
    try:
        import lightgbm as lgb
        use_lightgbm = True
        print("Using LightGBM")
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        use_lightgbm = False
        print("LightGBM not available, using sklearn GradientBoostingClassifier")
    
    start_time = time.time()
    
    if use_lightgbm:
        # LightGBM
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        # Sklearn GradientBoosting
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, average='weighted'
    )
    
    print(f"Validation F1 (weighted): {f1:.4f}")
    print(f"Validation Precision (weighted): {precision:.4f}")
    print(f"Validation Recall (weighted): {recall:.4f}")
    
    model_name = 'LightGBM' if use_lightgbm else 'Gradient Boosting'
    
    metrics = {
        'model_name': model_name,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_f1_weighted': f1,
        'val_precision_weighted': precision,
        'val_recall_weighted': recall,
        'train_time': train_time
    }
    
    return model, metrics

def train_svm(X_train, y_train, X_val, y_val):
    """
    Train SVM model (RBF kernel)
    
    Returns:
        Trained model and metrics
    """
    print("\nTraining SVM (RBF kernel)")
    
    start_time = time.time()
    
    # RBF SVM with balanced class weights and probability for ROC-AUC
    model = SVC(
        kernel='rbf', # RBF kernel
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, average='weighted'
    )
    
    print(f"Validation F1 (weighted): {f1:.4f}")
    print(f"Validation Precision (weighted): {precision:.4f}")
    print(f"Validation Recall (weighted): {recall:.4f}")
    
    metrics = {
        'model_name': 'Support Vector Machine',
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_f1_weighted': f1,
        'val_precision_weighted': precision,
        'val_recall_weighted': recall,
        'train_time': train_time
    }
    
    return model, metrics

def print_detailed_evaluation(model, X_val, y_val, model_name):
    """
    Print detailed evaluation metrics
    """
    print(f"\nDetailed Evaluation: {model_name}")
    
    y_val_pred = model.predict(X_val)
    
    # Classification report - only use labels present in validation data
    unique_labels = sorted(set(y_val) | set(y_val_pred))
    label_names = [label for label in PARTIES if label in unique_labels]
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, labels=unique_labels, target_names=label_names, zero_division=0))
    
    # Confusion matrix - only for labels present in data
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)
    
    # ROC-AUC (one-vs-rest)
    if hasattr(model, 'predict_proba'):
        y_val_proba = model.predict_proba(X_val)
        try:
            roc_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='weighted')
            print(f"\nROC-AUC (weighted, one-vs-rest): {roc_auc:.4f}")
        except Exception as e:
            print(f"\nCould not compute ROC-AUC: {e}")


def save_model(model, metrics, save_path):
    """
    Save trained model and metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat()
        }, f)
    
    print(f"\nModel saved to: {save_path}")


def compare_models(all_metrics):
    """
    Compare all trained models
    """
    print("\nMODEL COMPARISON")
    
    df = pd.DataFrame(all_metrics)
    df = df.sort_values('val_f1_weighted', ascending=False)
    
    print("\nRanked by Validation F1 Score:")
    print(df.to_string(index=False))
    
    best_model_name = df.iloc[0]['model_name']
    best_f1 = df.iloc[0]['val_f1_weighted']
    
    print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")
    
    return df


def train_all_models(include_history=True):
    """
    Train all Stage 1 models
    
    Args:
        include_history: If True, use full features; if False, demographics only
    """
    suffix = '_full' if include_history else '_demo'
    feature_type = "Full Features" if include_history else "Demographics Only"
    
    print(f"STAGE 1: PARTY CLASSIFICATION - {feature_type}")
    
    # Load data
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_data_splits()
    
    # Prepare features
    data = prepare_supervised_learning_data(
        train_df, val_df, test_df,
        include_history=include_history,
        save_dir=os.path.join(get_models_dir(), 'supervised')
    )
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train_party']
    y_val = data['y_val_party']
    
    print(f"\nTraining set size: {len(y_train):,}")
    print(f"Validation set size: {len(y_val):,}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train all models
    all_models = {}
    all_metrics = []
    
    # 1. Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    all_models['logistic_regression'] = lr_model
    all_metrics.append(lr_metrics)
    print_detailed_evaluation(lr_model, X_val, y_val, "Logistic Regression")
    
    # 2. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    all_models['random_forest'] = rf_model
    all_metrics.append(rf_metrics)
    print_detailed_evaluation(rf_model, X_val, y_val, "Random Forest")
    
    # 3. Gradient Boosting (optional - skip if dependencies missing)
    try:
        gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_val, y_val)
        all_models['gradient_boosting'] = gb_model
        all_metrics.append(gb_metrics)
        print_detailed_evaluation(gb_model, X_val, y_val, "Gradient Boosting")
    except Exception as e:
        print(f"\nGradient Boosting skipped: {str(e)[:100]}")
        print("Continuing with Logistic Regression and Random Forest models...")

    # 4. SVM
    svm_model, svm_metrics = train_svm(X_train, y_train, X_val, y_val)
    svm_metrics['model_key'] = 'support_vector_machine'
    all_models['support_vector_machine'] = svm_model
    all_metrics.append(svm_metrics)
    print_detailed_evaluation(svm_model, X_val, y_val, "Support Vector Machine")
    
    # Compare models
    comparison_df = compare_models(all_metrics)
    
    # Select best model
    best_model_type = comparison_df.iloc[0]['model_name']
    if 'Logistic' in best_model_type:
        best_model = all_models['logistic_regression']
        best_key = 'logistic_regression'
    elif 'Random' in best_model_type:
        best_model = all_models['random_forest']
        best_key = 'random_forest'
    elif 'Support Vector Machine' in best_model_type:
        best_model = all_models['support_vector_machine']
        best_key = 'support_vector_machine'
    elif 'gradient_boosting' in all_models:
        best_model = all_models['gradient_boosting']
        best_key = 'gradient_boosting'
    else:
        # Default to logistic regression if gradient boosting unavailable
        best_model = all_models['logistic_regression']
        best_key = 'logistic_regression'
    
    best_metrics = comparison_df.iloc[0].to_dict()
    
    # Save all models
    save_dir = os.path.join(get_models_dir(), 'supervised')
    os.makedirs(save_dir, exist_ok=True)
    
    for model_key, model in all_models.items():
        metrics = [m for m in all_metrics if model_key.replace('_', ' ').title() in m['model_name'] 
                   or m['model_name'].replace(' ', '_').lower() == model_key][0]
        save_path = os.path.join(save_dir, f'stage1_party_{model_key}{suffix}.pkl')
        save_model(model, metrics, save_path)
    
    # Save best model separately
    best_save_path = os.path.join(save_dir, f'stage1_party_best{suffix}.pkl')
    save_model(best_model, best_metrics, best_save_path)
    
    # Save comparison
    comparison_path = os.path.join(save_dir, f'stage1_comparison{suffix}.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nModel comparison saved to: {comparison_path}")
    
    print("\nSTAGE 1 TRAINING COMPLETE")
    
    return best_model, best_metrics, all_models, comparison_df


def main():
    """
    Main training script
    """
    # Train models with full features
    print("\nTRAINING WITH FULL FEATURES (Including History)")
    train_all_models(include_history=True)
    
    # Train models with demographics only
    print("\nTRAINING WITH DEMOGRAPHICS ONLY (For New Contributors)")
    train_all_models(include_history=False)
    
    print("\nALL STAGE 1 MODELS TRAINED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
