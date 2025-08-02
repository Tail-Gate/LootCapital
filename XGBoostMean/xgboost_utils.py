import xgboost as xgb
import numpy as np
import pandas as pd
import shap
from typing import Tuple, Optional, Dict, Any
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    num_rounds: int = 100,
    early_stopping_rounds: int = 10,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    num_classes: int = 2
) -> xgb.Booster:
    """
    Train an XGBoost model with optional early stopping.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: XGBoost parameters (defaults to reasonable values if None)
        num_rounds: Maximum number of training rounds
        early_stopping_rounds: Number of rounds for early stopping
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        num_classes: Number of classes (2 for binary, 3+ for multi-class)
        
    Returns:
        Trained XGBoost model
    """
    if params is None:
        if num_classes == 2:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0
            }
        else:
            params = {
                'objective': 'multi:softmax',
                'eval_metric': 'mlogloss',
                'num_class': num_classes,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0
            }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_rounds,
            watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
    else:
        model = xgb.train(params, dtrain, num_rounds, verbose_eval=False)
    
    return model

def predict_xgboost(
    model: xgb.Booster,
    X: pd.DataFrame,
    probability_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using a trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X: Features to predict on
        probability_threshold: Threshold for binary classification
        
    Returns:
        Tuple of (probabilities, binary predictions)
    """
    dtest = xgb.DMatrix(X)
    probabilities = model.predict(dtest)
    predictions = (probabilities >= probability_threshold).astype(int)
    return probabilities, predictions

def explain_xgboost(
    model: xgb.Booster,
    X: pd.DataFrame,
    feature_names: Optional[list] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate SHAP values and feature importance for model explanation.
    
    Args:
        model: Trained XGBoost model
        X: Features to explain
        feature_names: Optional list of feature names
        
    Returns:
        Tuple of (SHAP values, feature importance dictionary)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Get feature importance
    importance_dict = dict(zip(
        feature_names if feature_names else X.columns,
        model.get_score(importance_type='gain')
    ))
    
    return shap_values, importance_dict

def save_xgboost(model: xgb.Booster, path: str) -> None:
    """Save XGBoost model to file."""
    model.save_model(path)

def load_xgboost(path: str) -> xgb.Booster:
    """Load XGBoost model from file."""
    model = xgb.Booster()
    model.load_model(path)
    return model

def get_feature_importance(model, feature_names=None):
    # Returns importance as a dict {feature: score}
    return model.get_score(importance_type='gain')

def explain_with_shap(model, X, feature_names=None):
    """
    Generate comprehensive SHAP explanations for XGBoost model predictions.
    
    Args:
        model: Trained XGBoost model
        X: Features to explain (should be numpy array or pandas DataFrame)
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary containing comprehensive SHAP explanation including:
        - shap_values: Raw SHAP values
        - feature_contributions: Dictionary of feature contributions for each class
        - prediction_breakdown: Detailed breakdown of prediction factors
        - summary_plot_data: Data for creating summary plots
        - force_plot_data: Data for creating force plots
    """
    if not SHAP_AVAILABLE:
        raise ImportError('SHAP is not installed. Please install shap to use this feature.')
    
    # Ensure X is in the right format
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_array = X
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_array)
    
    # Handle different output formats based on model type
    if isinstance(shap_values, list):
        # Multi-class model
        num_classes = len(shap_values)
        base_values = explainer.expected_value
        if isinstance(base_values, list):
            base_values = np.array(base_values)
    else:
        # Binary or single output model
        shap_values = [shap_values]
        num_classes = 1
        base_values = explainer.expected_value
        if isinstance(base_values, list):
            base_values = np.array(base_values)
    
    # Get predictions for context
    dtest = xgb.DMatrix(X_array)
    predictions = model.predict(dtest)
    
    # Ensure predictions are in the right format
    if len(predictions.shape) == 1:
        if num_classes > 1:
            predictions = predictions.reshape(-1, num_classes)
        else:
            predictions = predictions.reshape(-1, 1)
    
    # Create comprehensive explanation dictionary
    explanation = {
        'shap_values': shap_values,
        'base_values': base_values,
        'predictions': predictions,
        'feature_names': feature_names,
        'feature_contributions': {},
        'prediction_breakdown': [],
        'summary_plot_data': {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'X': X_array
        },
        'force_plot_data': {
            'shap_values': shap_values,
            'base_values': base_values,
            'feature_names': feature_names,
            'X': X_array
        }
    }
    
    # Calculate feature contributions for each sample and class
    for sample_idx in range(len(X_array)):
        sample_contributions = {}
        sample_breakdown = {
            'sample_index': sample_idx,
            'predicted_class': np.argmax(predictions[sample_idx]) if num_classes > 1 else (1 if predictions[sample_idx] > 0.5 else 0),
            'prediction_probabilities': predictions[sample_idx].tolist(),
            'feature_effects': {}
        }
        
        for class_idx in range(num_classes):
            class_contributions = {}
            for feat_idx, feat_name in enumerate(feature_names):
                contribution = shap_values[class_idx][sample_idx, feat_idx]
                class_contributions[feat_name] = contribution
                
                # Store in breakdown for the predicted class
                if class_idx == sample_breakdown['predicted_class']:
                    sample_breakdown['feature_effects'][feat_name] = contribution
            
            sample_contributions[f'class_{class_idx}'] = class_contributions
        
        explanation['feature_contributions'][f'sample_{sample_idx}'] = sample_contributions
        explanation['prediction_breakdown'].append(sample_breakdown)
    
    # Add summary statistics
    explanation['summary_stats'] = {
        'num_samples': len(X_array),
        'num_features': len(feature_names),
        'num_classes': num_classes,
        'feature_importance': {}
    }
    
    # Calculate overall feature importance across all classes
    for feat_idx, feat_name in enumerate(feature_names):
        total_importance = 0
        for class_idx in range(num_classes):
            total_importance += np.mean(np.abs(shap_values[class_idx][:, feat_idx]))
        explanation['summary_stats']['feature_importance'][feat_name] = total_importance / num_classes
    
    return explanation

def predict_xgboost_multi(
    model: xgb.Booster,
    X: pd.DataFrame,
    num_classes: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make multi-class predictions using a trained XGBoost model.
    For 'multi:softprob', returns class probabilities for each sample.
    
    Args:
        model: Trained XGBoost model
        X: Features to predict on
        num_classes: Number of classes (default: 3 for short/hold/long)
        
    Returns:
        Tuple of (class_probabilities, class_predictions)
        - class_probabilities: Array of shape (n_samples, num_classes) with probabilities for each class
        - class_predictions: Array of shape (n_samples,) with predicted class labels
    """
    dtest = xgb.DMatrix(X)
    
    # Get class probabilities (for multi:softprob, this is a flat array)
    class_probabilities = model.predict(dtest)
    
    # If output is 1D, reshape to (n_samples, num_classes)
    if len(class_probabilities.shape) == 1:
        class_probabilities = class_probabilities.reshape(-1, num_classes)
    
    # If output is 2D but second dimension is not num_classes, reshape
    elif class_probabilities.shape[1] != num_classes:
        class_probabilities = class_probabilities.reshape(-1, num_classes)
    
    # Get predicted class labels
    class_predictions = np.argmax(class_probabilities, axis=1)
    
    return class_probabilities, class_predictions 

def create_shap_plots(explanation, save_path=None):
    """
    Create SHAP visualization plots from explanation data.
    
    Args:
        explanation: SHAP explanation dictionary from explain_with_shap
        save_path: Optional path to save plots (if None, uses default location with timestamp)
        
    Returns:
        Dictionary containing plot objects and file paths
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available for plotting")
        return None
    
    try:
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        
        # Create default save path if none provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/shap_explanation_{timestamp}"
        else:
            # Ensure save_path uses the plots directory
            if not save_path.startswith("plots/"):
                save_path = f"plots/{save_path}"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plots = {}
        
        # Summary plot
        if 'summary_plot_data' in explanation:
            summary_data = explanation['summary_plot_data']
            shap_values = summary_data['shap_values']
            feature_names = summary_data['feature_names']
            X = summary_data['X']
            
            # Create summary plot
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                # Multi-class: use first class for summary
                shap.summary_plot(shap_values[0], X, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            
            plt.title("SHAP Summary Plot")
            summary_plot_path = f"{save_path}_summary.png"
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plots['summary_plot'] = summary_plot_path
            plt.close()
        
        # Force plot for latest prediction
        if 'force_plot_data' in explanation and len(explanation['prediction_breakdown']) > 0:
            force_data = explanation['force_plot_data']
            shap_values = force_data['shap_values']
            base_values = force_data['base_values']
            feature_names = force_data['feature_names']
            X = force_data['X']
            
            # Use the latest sample
            sample_idx = -1
            
            plt.figure(figsize=(12, 6))
            if isinstance(shap_values, list):
                # Multi-class: use predicted class
                predicted_class = explanation['prediction_breakdown'][sample_idx]['predicted_class']
                shap.force_plot(
                    base_values[predicted_class], 
                    shap_values[predicted_class][sample_idx], 
                    X[sample_idx],
                    feature_names=feature_names,
                    show=False
                )
            else:
                shap.force_plot(
                    base_values, 
                    shap_values[sample_idx], 
                    X[sample_idx],
                    feature_names=feature_names,
                    show=False
                )
            
            plt.title("SHAP Force Plot - Latest Prediction")
            force_plot_path = f"{save_path}_force.png"
            plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
            plots['force_plot'] = force_plot_path
            plt.close()
        
        # Waterfall plot for latest prediction
        if len(explanation['prediction_breakdown']) > 0:
            sample_idx = -1
            prediction_breakdown = explanation['prediction_breakdown'][sample_idx]
            feature_effects = prediction_breakdown['feature_effects']
            
            # Sort features by absolute contribution
            sorted_features = sorted(feature_effects.items(), key=lambda x: abs(x[1]), reverse=True)
            
            plt.figure(figsize=(10, 8))
            features = [f[0] for f in sorted_features[:10]]  # Top 10 features
            contributions = [f[1] for f in sorted_features[:10]]
            colors = ['red' if c < 0 else 'blue' for c in contributions]
            
            bars = plt.barh(features, contributions, color=colors)
            plt.xlabel('SHAP Contribution')
            plt.title('Feature Contributions to Latest Prediction')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, contribution) in enumerate(zip(bars, contributions)):
                plt.text(contribution, i, f'{contribution:.3f}', 
                        ha='left' if contribution > 0 else 'right', va='center')
            
            waterfall_plot_path = f"{save_path}_waterfall.png"
            plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
            plots['waterfall_plot'] = waterfall_plot_path
            plt.close()
        
        print(f"[SHAP] Plots saved to: {save_path}_*.png")
        return plots
        
    except Exception as e:
        print(f"Error creating SHAP plots: {e}")
        return None 