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
    if not SHAP_AVAILABLE:
        raise ImportError('SHAP is not installed. Please install shap to use this feature.')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

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