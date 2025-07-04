#!/usr/bin/env python3
"""
Walk-Forward Optimization for Three-Class XGBoost Momentum Strategy

This script implements walk-forward optimization to find the best hyperparameters
for the three-class XGBoost momentum strategy, handling class imbalance through
precision and recall optimization.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import ParameterGrid
import warnings
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

from strategies.xgBoost_momentum_strategy import MomentumStrategy, MomentumConfig
from utils.feature_generator import FeatureGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization"""
    # Data paths
    main_data_path: str = 'data/historical/ETH-USDT-SWAP_ohlcv_15m.csv'
    test_data_path: str = 'data/historical/ETH-USDT-SWAP_ohlcv_15m.csv'  # Same file, will use subset
    
    # Main optimization parameters (for 5 years of data)
    main_train_window_months: int = 6
    main_test_window_months: int = 1
    main_step_size_months: int = 1
    
    # Test optimization parameters (for 1 month of data)
    test_train_window_weeks: int = 2
    test_test_window_weeks: int = 1
    test_step_size_weeks: int = 1
    
    # Model parameters
    num_classes: int = 3  # 0=short, 1=hold, 2=long
    
    # XGBoost hyperparameter search space
    param_grid: Dict = None
    
    # Model saving
    best_model_path: str = 'models/best_xgboost_walkforward.json'
    results_path: str = 'walk_forward_results.json'
    
    def __post_init__(self):
        if self.param_grid is None:
            self.param_grid = {
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'num_boost_round': [50, 100, 150]
            }

class WalkForwardOptimizer:
    """Walk-forward optimization for XGBoost momentum strategy"""
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.results = []
        self.best_model = None
        self.best_score = 0.0
        self.best_params = None
        
    def load_data(self, use_test_mode: bool = False) -> pd.DataFrame:
        """Load and prepare data for walk-forward optimization"""
        logger.info(f"Loading data for {'test' if use_test_mode else 'main'} mode...")
        
        # Load data
        data = pd.read_csv(self.config.main_data_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data.sort_index(inplace=True)
        
        if use_test_mode:
            # Use only the last month for testing
            data = data.tail(2880)  # 1 month of 15-minute data
            logger.info(f"Using {len(data)} samples for test mode")
        else:
            logger.info(f"Using {len(data)} samples for main optimization")
        
        return data
    
    def create_walk_forward_windows(self, data: pd.DataFrame, use_test_mode: bool = False) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create training and testing windows for walk-forward optimization"""
        windows = []
        
        if use_test_mode:
            # Test mode: use weeks
            train_window = self.config.test_train_window_weeks * 7 * 24 * 4  # weeks to 15-min intervals
            test_window = self.config.test_test_window_weeks * 7 * 24 * 4
            step_size = self.config.test_step_size_weeks * 7 * 24 * 4
        else:
            # Main mode: use months
            train_window = self.config.main_train_window_months * 30 * 24 * 4  # months to 15-min intervals
            test_window = self.config.main_test_window_months * 30 * 24 * 4
            step_size = self.config.main_step_size_months * 30 * 24 * 4
        
        start_idx = train_window
        end_idx = len(data) - test_window
        
        while start_idx <= end_idx:
            train_start = start_idx - train_window
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + test_window
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            windows.append((train_data, test_data))
            start_idx += step_size
        
        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows
    
    def prepare_features_and_labels(self, data: pd.DataFrame, strategy: MomentumStrategy) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for a given dataset"""
        # Prepare features
        features = strategy.prepare_features(data)
        
        # Get available features
        available_features = [f for f in strategy.config.feature_list if f in features.columns]
        strategy.config.feature_list = available_features
        
        # Clean features
        features_clean = features[available_features].dropna()
        
        # Create labels
        labels = strategy.create_three_class_labels(
            data, 
            lookforward_periods=strategy.config.lookforward_periods,
            threshold_pct=strategy.config.price_threshold_pct
        )
        labels_clean = labels.loc[features_clean.index].dropna()
        
        # Align features and labels
        common_index = features_clean.index.intersection(labels_clean.index)
        features_final = features_clean.loc[common_index]
        labels_final = labels_clean.loc[common_index]
        
        return features_final, labels_final
    
    def evaluate_model(self, model: xgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance using precision and recall"""
        # Make predictions
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = model.predict(dtest)
        
        # Handle different prediction formats
        if len(y_pred_proba.shape) == 1:
            # If 1D, reshape to (n_samples, num_classes)
            y_pred_proba = y_pred_proba.reshape(-1, self.config.num_classes)
        
        # Get predicted class labels
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics for each class
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Calculate metrics for individual classes
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        
        # Calculate average precision and recall (focus on SHORT and LONG classes)
        # Class 0: SHORT, Class 1: HOLD, Class 2: LONG
        short_long_precision = np.mean([precision_per_class[0], precision_per_class[2]])
        short_long_recall = np.mean([recall_per_class[0], recall_per_class[2]])
        
        return {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'short_long_precision': short_long_precision,
            'short_long_recall': short_long_recall,
            'avg_short_long_score': (short_long_precision + short_long_recall) / 2
        }
    
    def train_and_evaluate(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                          params: Dict, strategy: MomentumStrategy) -> Dict[str, float]:
        """Train model and evaluate on test data"""
        try:
            # Prepare training data
            X_train, y_train = self.prepare_features_and_labels(train_data, strategy)
            
            if len(X_train) < 100:
                logger.warning(f"Insufficient training data: {len(X_train)} samples")
                return None
            
            # Prepare test data
            X_test, y_test = self.prepare_features_and_labels(test_data, strategy)
            
            if len(X_test) < 10:
                logger.warning(f"Insufficient test data: {len(X_test)} samples")
                return None
            
            # Create XGBoost parameters with multi:softprob objective
            xgb_params = {
                'objective': 'multi:softprob',  # Use softprob for probability output
                'eval_metric': 'mlogloss',
                'num_class': self.config.num_classes,
                'max_depth': params['max_depth'],
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'min_child_weight': 1,
                'gamma': 0
            }
            
            # Train model
            strategy.train(X_train, y_train, params=xgb_params, num_boost_round=params['num_boost_round'])
            
            # Evaluate model
            metrics = self.evaluate_model(strategy.model, X_test, y_test)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {e}")
            return None
    
    def optimize_hyperparameters(self, use_test_mode: bool = False) -> Dict:
        """Run walk-forward optimization to find best hyperparameters"""
        logger.info(f"Starting walk-forward optimization in {'test' if use_test_mode else 'main'} mode...")
        
        # Load data
        data = self.load_data(use_test_mode)
        
        # Create walk-forward windows
        windows = self.create_walk_forward_windows(data, use_test_mode)
        
        # Initialize strategy
        strategy = MomentumStrategy()
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(self.config.param_grid))
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Track results for each parameter combination
        param_results = {}
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing parameter combination {i+1}/{len(param_combinations)}: {params}")
            
            window_scores = []
            
            for j, (train_data, test_data) in enumerate(windows):
                logger.info(f"  Window {j+1}/{len(windows)}")
                
                # Train and evaluate
                metrics = self.train_and_evaluate(train_data, test_data, params, strategy)
                
                if metrics is not None:
                    window_scores.append(metrics)
            
            if window_scores:
                # Calculate average scores across all windows
                avg_scores = {}
                for metric in window_scores[0].keys():
                    avg_scores[metric] = np.mean([score[metric] for score in window_scores])
                
                param_results[str(params)] = {
                    'params': params,
                    'avg_scores': avg_scores,
                    'window_scores': window_scores
                }
                
                # Update best model if this is better
                current_score = avg_scores['avg_short_long_score']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_params = params
                    self.best_model = strategy.model
                    logger.info(f"New best model found! Score: {current_score:.4f}")
        
        return param_results
    
    def save_results(self, param_results: Dict, use_test_mode: bool = False):
        """Save optimization results and best model"""
        # Save results
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'test' if use_test_mode else 'main',
            'best_score': self.best_score,
            'best_params': self.best_params,
            'all_results': param_results
        }
        
        with open(self.config.results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save best model
        if self.best_model is not None:
            self.best_model.save_model(self.config.best_model_path)
            logger.info(f"Best model saved to {self.config.best_model_path}")
        
        logger.info(f"Results saved to {self.config.results_path}")
    
    def print_summary(self, param_results: Dict):
        """Print optimization summary"""
        print("\n" + "="*60)
        print("WALK-FORWARD OPTIMIZATION SUMMARY")
        print("="*60)
        
        if self.best_params:
            print(f"Best Score: {self.best_score:.4f}")
            print(f"Best Parameters: {self.best_params}")
        
        # Show top 5 parameter combinations
        sorted_results = sorted(
            param_results.items(),
            key=lambda x: x[1]['avg_scores']['avg_short_long_score'],
            reverse=True
        )
        
        print(f"\nTop 5 Parameter Combinations:")
        for i, (param_str, result) in enumerate(sorted_results[:5]):
            score = result['avg_scores']['avg_short_long_score']
            print(f"{i+1}. Score: {score:.4f}, Params: {result['params']}")
        
        print("="*60)

def main():
    """Main function to run walk-forward optimization"""
    config = WalkForwardConfig()
    optimizer = WalkForwardOptimizer(config)
    
    # Run test mode first (with 1 month data)
    print("Running test mode with 1 month of data...")
    test_results = optimizer.optimize_hyperparameters(use_test_mode=True)
    optimizer.save_results(test_results, use_test_mode=True)
    optimizer.print_summary(test_results)
    
    # Ask user if they want to proceed with main optimization
    response = input("\nDo you want to proceed with main optimization (5 years of data)? (y/n): ")
    
    if response.lower() == 'y':
        print("Running main optimization with 5 years of data...")
        main_results = optimizer.optimize_hyperparameters(use_test_mode=False)
        optimizer.save_results(main_results, use_test_mode=False)
        optimizer.print_summary(main_results)
    else:
        print("Main optimization skipped.")

if __name__ == "__main__":
    main() 