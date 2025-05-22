import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from optuna.trial import Trial
import joblib
import mlflow
from mlflow.models import infer_signature

class ModelTrainer:
    """
    Handles model training, hyperparameter optimization, and training progress tracking
    for the momentum strategy.
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        config: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        version: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ("xgboost" or "lstm")
            config: Configuration dictionary for training
            cache_dir: Directory to cache trained models
            version: Version identifier for the trainer
            mlflow_tracking_uri: URI for MLflow tracking server
        """
        self.model_type = model_type
        self.config = config or {}
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/models")
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize training state
        self.model = None
        self.best_params = None
        self.training_history = []
        self.feature_importance = {}
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup MLflow if URI provided
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(f"momentum_strategy_{self.version}")
    
    def _filter_numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only columns with types int, float, bool, or category.
        """
        allowed_kinds = {'i', 'u', 'f', 'b'}  # int, uint, float, bool
        cols = [
            col for col in X.columns
            if (
                pd.api.types.is_numeric_dtype(X[col]) or
                pd.api.types.is_bool_dtype(X[col]) or
                pd.api.types.is_categorical_dtype(X[col])
            )
        ]
        return X[cols]
    
    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        cv_splits: int = 5,
        metric: str = "f1"
    ) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using Optuna.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_trials: Number of optimization trials
            cv_splits: Number of cross-validation splits
            metric: Metric to optimize ("accuracy", "precision", "recall", "f1")
            
        Returns:
            Dictionary of best hyperparameters
        """
        def objective(trial: Trial) -> float:
            if self.model_type == "xgboost":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5)
                }
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Filter features
            X_filtered = self._filter_numeric_features(X)
            
            # Perform time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_filtered):
                X_train, X_val = X_filtered.iloc[train_idx], X_filtered.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                if self.model_type == "xgboost":
                    model = xgb.XGBClassifier(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                    y_pred = model.predict(X_val)
                
                # Calculate metric
                if metric == "accuracy":
                    score = accuracy_score(y_val, y_pred)
                elif metric == "precision":
                    score = precision_score(y_val, y_pred)
                elif metric == "recall":
                    score = recall_score(y_val, y_pred)
                elif metric == "f1":
                    score = f1_score(y_val, y_pred)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                scores.append(score)
            
            return np.mean(scores)
        
        # Create and run study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        return study.best_params
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 50,
        use_mlflow: bool = False
    ) -> Dict[str, Any]:
        """
        Train the model with the best hyperparameters.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            validation_split: Fraction of data to use for validation
            early_stopping_rounds: Number of rounds for early stopping
            use_mlflow: Whether to log training to MLflow
            
        Returns:
            Dictionary of training metrics
        """
        # Filter features
        X_filtered = self._filter_numeric_features(X)
        
        # Split data
        split_idx = int(len(X_filtered) * (1 - validation_split))
        X_train, X_val = X_filtered.iloc[:split_idx], X_filtered.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Initialize model with best parameters
        if self.model_type == "xgboost":
            if not self.best_params:
                self.best_params = {
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 500,
                    "min_child_weight": 1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "gamma": 0
                }
            
            self.model = xgb.XGBClassifier(**self.best_params)
            
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            
            # Log to MLflow if requested
            if use_mlflow:
                with mlflow.start_run():
                    # Log parameters
                    mlflow.log_params(self.best_params)
                    
                    # Log metrics
                    y_pred = self.model.predict(X_val)
                    metrics = {
                        "accuracy": accuracy_score(y_val, y_pred),
                        "precision": precision_score(y_val, y_pred),
                        "recall": recall_score(y_val, y_pred),
                        "f1": f1_score(y_val, y_pred)
                    }
                    mlflow.log_metrics(metrics)
                    
                    # Log model
                    signature = infer_signature(X_val, y_pred)
                    mlflow.sklearn.log_model(
                        self.model,
                        "model",
                        signature=signature,
                        input_example=X_val.iloc[:5]
                    )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Store feature importance
        if self.model_type == "xgboost":
            self.feature_importance = dict(zip(
                X_filtered.columns,
                [float(imp) for imp in self.model.feature_importances_]
            ))
        
        # Calculate and store metrics
        y_pred = self.model.predict(X_val)
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred)
        }
        
        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "params": self.best_params
        })
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Filter features
        X_filtered = self._filter_numeric_features(X)
        return self.model.predict(X_filtered)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Filter features
        X_filtered = self._filter_numeric_features(X)
        return self.model.predict_proba(X_filtered)
    
    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save the trained model and training history.
        
        Args:
            path: Path to save model (defaults to cache directory)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        path = Path(path) if path else self.cache_dir / f"model_{self.version}"
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model_type == "xgboost":
            joblib.dump(self.model, path / "model.joblib")
        
        # Save training history
        with open(path / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save feature importance
        with open(path / "feature_importance.json", "w") as f:
            json.dump(self.feature_importance, f, indent=2)
        
        # Save best parameters
        with open(path / "best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)
        
        self.logger.info(f"Saved model and training history to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model and its training history.
        
        Args:
            path: Path to load model from
        """
        path = Path(path)
        
        # Load model
        if self.model_type == "xgboost":
            self.model = joblib.load(path / "model.joblib")
        
        # Load training history
        with open(path / "training_history.json", "r") as f:
            self.training_history = json.load(f)
        
        # Load feature importance
        with open(path / "feature_importance.json", "r") as f:
            self.feature_importance = json.load(f)
        
        # Load best parameters
        with open(path / "best_params.json", "r") as f:
            self.best_params = json.load(f)
        
        self.logger.info(f"Loaded model and training history from {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        return self.feature_importance
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get training history.
        
        Returns:
            List of training history entries
        """
        return self.training_history
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the model trainer.
        This includes closing any open file handles, database connections, etc.
        """
        try:
            # Clear model to free memory
            self.model = None
            
            # Clear training history
            self.training_history = []
            
            # Clear feature importance
            self.feature_importance = {}
            
            # Clear best parameters
            self.best_params = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error during model trainer cleanup: {str(e)}")
            raise 