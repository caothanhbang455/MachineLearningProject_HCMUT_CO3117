"""
Model Evaluation Utilities for Machine Learning

This module contains functions for:
- Comparing multiple models
- Calculating performance metrics
- Plotting model comparison results
- Cross-validation utilities

"""

import time
import yaml
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (cross_val_score, learning_curve,
                                    validation_curve, GridSearchCV,
                                    GroupShuffleSplit, GroupKFold, GridSearchCV)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_curve, auc, classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings('ignore')

def load_models(config_path="config/models.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    models_setup = []

    for name, m in config.items():
        module_name, class_name = m['class'].rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        model = cls(**(m.get('kwargs') or {}))
        params = m.get('params') or {}

        models_setup.append((name, model, params))

    return models_setup


def compare_models(models_with_params, X, y, groups=None, cv=5, scoring='accuracy'):
    results = {
        'Model': [], 'Mean_Score': [], 'Std_Score': [], 'Training_Time': [], 'Best_Params': []
    }

    if groups is not None:
        cv_splitter = GroupKFold(n_splits=cv)
    else:
        cv_splitter = cv

    for name, model, param_grid in models_with_params:
        import time
        start_time = time.time()

        try:
            grid = GridSearchCV(model, param_grid, cv=cv_splitter, scoring=scoring, n_jobs=-1)

            if groups is not None:
                grid.fit(X, y, groups=groups)
            else:
                grid.fit(X, y)

            best_score = grid.best_score_
            std_score = grid.cv_results_['std_test_score'][grid.best_index_]
            best_params = grid.best_params_
            best_model = grid.best_estimator_

        except Exception as e:
            print(f"Error in trainning {name}: {e}")
            best_score, std_score, best_params, best_model = 0, 0, "Failed", None

        training_time = time.time() - start_time

        results['Model'].append(name)
        results['Mean_Score'].append(best_score)
        results['Std_Score'].append(std_score)
        results['Training_Time'].append(training_time)
        results['Best_Params'].append(best_params)

        print(f"{name:<20}: {best_score:.4f} (Time: {training_time:.2f}s)")

    return pd.DataFrame(results).sort_values('Mean_Score', ascending=False)



def calculate_model_metrics(y_true, y_pred, task_type='regression'):
    """
    Calculate comprehensive metrics for model evaluation.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    task_type : str
        'regression' or 'classification'

    Returns:
    --------
    dict
        Dictionary of calculated metrics
    """
    metrics = {}

    if task_type == 'regression':
        metrics['R²'] = r2_score(y_true, y_pred)
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    else:  # classification
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['F1'] = f1_score(y_true, y_pred, average='weighted')

    return metrics


def evaluate_song_level_accuracy(model, X_test, y_test, groups_test, label_encoder):

    # Segment-level predictions
    y_pred_segments = model.predict(X_test)

    print("\n===== SEGMENT-LEVEL METRICS =====")
    print(f"Segment Accuracy: {accuracy_score(y_test, y_pred_segments):.4f}")
    print("\nSegment Classification Report:")
    print(classification_report(y_test, y_pred_segments, target_names=label_encoder.classes_))

    # Build dataframe aligned by file
    results_df = pd.DataFrame({
        'file_name': groups_test,
        'true_label': y_test,
        'pred_label': y_pred_segments
    })

    # Majority vote per file
    vote_results = results_df.groupby('file_name').agg({
        'true_label': 'first',
        'pred_label': lambda x: x.mode()[0]
    }).reset_index()

    # Song-level metrics
    y_true_song = vote_results['true_label']
    y_pred_song = vote_results['pred_label']

    song_acc = accuracy_score(y_true_song, y_pred_song)

    print("\n===== SONG-LEVEL METRICS (MAJORITY VOTE) =====")
    print(f"Song Accuracy: {song_acc:.4f}")
    print("\nSong Classification Report:")
    print(classification_report(y_true_song, y_pred_song, target_names=label_encoder.classes_))

    return song_acc, vote_results


def plot_model_comparison(comparison_df, figsize=(12, 8), title="Model Comparison"):
    """
    Create a beautiful visualization of model comparison results.

    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame from compare_models function
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))

    # 1. Model Scores
    ax1 = axes[0, 0]
    bars = ax1.bar(comparison_df['Model'], comparison_df['Mean_Score'],
                   color=colors, alpha=0.8)
    ax1.errorbar(comparison_df['Model'], comparison_df['Mean_Score'],
                 yerr=comparison_df['Std_Score'], fmt='none', color='black', capsize=5)
    ax1.set_title('Model Performance Scores', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, score in zip(bars, comparison_df['Mean_Score']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Training Time
    ax2 = axes[0, 1]
    bars = ax2.bar(comparison_df['Model'], comparison_df['Training_Time'],
                   color=colors, alpha=0.8)
    ax2.set_title('Training Time', fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, time_val in zip(bars, comparison_df['Training_Time']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')

        # 3. Score vs Time scatter
    ax3 = axes[1, 0]
    ax3.scatter(comparison_df['Training_Time'], comparison_df['Mean_Score'], 
                c=colors, s=100, alpha=0.8)
    ax3.set_xlabel('Training Time (seconds)')
    ax3.set_ylabel('Score')
    ax3.set_title('Score vs Training Time', fontweight='bold')

    # Add model labels
    for i, model in enumerate(comparison_df['Model']):
        ax3.annotate(model, (comparison_df['Training_Time'].iloc[i],
                             comparison_df['Mean_Score'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    # 4. Model ranking
    ax4 = axes[1, 1]
    y_pos = np.arange(len(comparison_df))
    bars = ax4.barh(y_pos, comparison_df['Mean_Score'], color=colors, alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(comparison_df['Model'])
    ax4.set_xlabel('Score')
    ax4.set_title('Model Ranking', fontweight='bold')

    # Add value labels
    for bar, score in zip(bars, comparison_df['Mean_Score']):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_enhanced(y_true, y_pred, class_names=None,
                                   title="Confusion Matrix", figsize=(8, 6)):
    """
    Plot an enhanced confusion matrix with better visualization.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list
        Names of classes
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')

    # Add accuracy information
    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.3f}',
             transform=plt.gca().transAxes, ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_roc_curves(models, X_test, y_test, figsize=(10, 8)):
    """
    Plot ROC curves for multiple binary classification models.

    Parameters:
    -----------
    models : dict
        Dictionary of model_name: fitted_model
    X_test : array-like
        Test features
    y_test : array-like
        Test targets (binary)
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

    for i, (name, model) in enumerate(models.items()):
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.decision_function(X_test)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curves Comparison', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_performance_metrics(metrics_dict, figsize=(10, 6)):
    """
    Plot performance metrics in a beautiful bar chart.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metric_name: value
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)

    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    bars = plt.bar(metrics, values, color=colors, alpha=0.8)

    plt.title('Performance Metrics', fontweight='bold', fontsize=14)
    plt.ylabel('Score', fontweight='bold')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()
