import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from typing import Dict, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from collections import Counter
from sklearn.metrics import classification_report, precision_recall_fscore_support

from utils.stgnn_utils import STGNNModel, train_stgnn, predict_stgnn
from strategies.stgnn_strategy import STGNNStrategy
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

# Import the classification model and trainer
from scripts.train_stgnn_improved import STGNNClassificationModel, ClassificationSTGNNTrainer
from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_stgnn_data(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and prepare data for STGNN training and validation
    
    Args:
        config: Configuration dictionary containing data parameters
        
    Returns:
        Tuple of (X_train, adj_train, y_train, X_val, adj_val, y_val)
    """
    # Initialize strategy components
    market_data = MarketData()
    technical_indicators = TechnicalIndicators()
    
    # Create strategy instance
    strategy = STGNNStrategy(config, market_data, technical_indicators)
    
    # Prepare data
    X, adj, y = strategy.prepare_data()
    
    # Split into train and validation sets (80/20 split)
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    adj_train = torch.FloatTensor(adj)
    adj_val = torch.FloatTensor(adj)
    
    return X_train, adj_train, y_train, X_val, adj_val, y_val

def create_classification_data_processor(config: STGNNConfig, start_time=None, end_time=None):
    """
    Create data processor for classification training
    
    Args:
        config: STGNN configuration
        start_time: Optional start time for data range
        end_time: Optional end time for data range
        
    Returns:
        Data processor instance
    """
    from market_analysis.market_data import MarketData
    from market_analysis.technical_indicators import TechnicalIndicators
    
    market_data = MarketData()
    technical_indicators = TechnicalIndicators()
    
    # Create data processor
    data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
    
    return data_processor

def objective(trial: optuna.Trial) -> float:
    """
    Enhanced objective function for hyperparameter optimization with feature engineering parameters
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Validation loss (weighted focal loss or combined metric)
    """
    # Define enhanced hyperparameter search space including feature engineering parameters
    config_dict = {
        'assets': ['ETH/USD'],  # Focus on single asset for optimization
        'features': ['price', 'volume', 'rsi', 'macd', 'bollinger', 'atr', 'adx', 'stoch', 'williams_r', 'cci', 'mfi', 'obv', 'vwap', 'support', 'resistance'],
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'hidden_dim': trial.suggest_int('hidden_dim', 16, 128, step=16),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'kernel_size': trial.suggest_int('kernel_size', 2, 5),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_int('batch_size', 16, 64, step=16),
        'seq_len': trial.suggest_int('seq_len', 50, 200, step=25),  # Much longer sequences for better feature capture
        'prediction_horizon': 15,  # Fixed as per current requirement
        'early_stopping_patience': 5,
        
        # Focal Loss parameters
        'focal_alpha': trial.suggest_float('focal_alpha', 0.25, 1.0),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 5.0),
        
        # Class multipliers for individual class weighting
        'class_multiplier_0': trial.suggest_float('class_multiplier_0', 0.5, 5.0),  # Down class
        'class_multiplier_1': trial.suggest_float('class_multiplier_1', 0.5, 5.0),  # No Direction class
        'class_multiplier_2': trial.suggest_float('class_multiplier_2', 0.5, 5.0),  # Up class
        
        # NEW: Feature Engineering Hyperparameters for 0.5% event-based prediction
        'price_threshold': 0.005,  # Fixed 0.5% threshold for classification
        
        # RSI parameters
        'rsi_period': trial.suggest_int('rsi_period', 7, 28),
        
        # MACD parameters
        'macd_fast_period': trial.suggest_int('macd_fast_period', 8, 20),
        'macd_slow_period': trial.suggest_int('macd_slow_period', 20, 40),
        'macd_signal_period': trial.suggest_int('macd_signal_period', 5, 15),
        
        # Bollinger Bands parameters
        'bb_period': trial.suggest_int('bb_period', 10, 40),
        'bb_num_std_dev': trial.suggest_float('bb_num_std_dev', 1.5, 3.0),
        
        # ATR parameters
        'atr_period': trial.suggest_int('atr_period', 7, 21),
        
        # ADX parameters
        'adx_period': trial.suggest_int('adx_period', 7, 21),
        
        # Volume parameters
        'volume_ma_period': trial.suggest_int('volume_ma_period', 10, 40),
        
        # Momentum parameters
        'price_momentum_lookback': trial.suggest_int('price_momentum_lookback', 3, 15),
    }
    
    # Create STGNNConfig with feature engineering parameters
    config = STGNNConfig(
        num_nodes=len(config_dict['assets']),
        input_dim=len(config_dict['features']),
        hidden_dim=config_dict['hidden_dim'],
        output_dim=3,  # 3 classes for classification
        num_layers=config_dict['num_layers'],
        dropout=config_dict['dropout'],
        kernel_size=config_dict['kernel_size'],
        learning_rate=config_dict['learning_rate'],
        batch_size=config_dict['batch_size'],
        seq_len=config_dict['seq_len'],
        prediction_horizon=config_dict['prediction_horizon'],
        early_stopping_patience=config_dict['early_stopping_patience'],
        features=config_dict['features'],
        assets=config_dict['assets'],
        focal_alpha=config_dict['focal_alpha'],
        focal_gamma=config_dict['focal_gamma'],
        # Feature engineering parameters
        rsi_period=config_dict['rsi_period'],
        macd_fast_period=config_dict['macd_fast_period'],
        macd_slow_period=config_dict['macd_slow_period'],
        macd_signal_period=config_dict['macd_signal_period'],
        bb_period=config_dict['bb_period'],
        bb_num_std_dev=config_dict['bb_num_std_dev'],
        atr_period=config_dict['atr_period'],
        adx_period=config_dict['adx_period'],
        volume_ma_period=config_dict['volume_ma_period'],
        price_momentum_lookback=config_dict['price_momentum_lookback'],
        price_threshold=config_dict['price_threshold']
    )
    
    try:
        # Create data processor with feature engineering config
        data_processor = create_classification_data_processor(config)
        
        # Prepare data for a recent time window (to avoid memory issues)
        # Use a 90-day window for hyperparameter optimization
        end_time = datetime.now()
        start_time = end_time - timedelta(days=90)
        
        # Create trainer with optimized parameters
        trainer = ClassificationSTGNNTrainer(
            config=config,
            data_processor=data_processor,
            price_threshold=config_dict['price_threshold'],  # Use optimized threshold
            focal_alpha=config_dict['focal_alpha'],
            focal_gamma=config_dict['focal_gamma'],
            class_weights=None,  # Will be calculated with multipliers
            start_time=start_time,
            end_time=end_time
        )
        
        # Override class weights calculation with multipliers
        def calculate_weighted_class_weights():
            """Calculate class weights with trial multipliers"""
            # Get class distribution from training data
            X, adj, y = data_processor.prepare_data(start_time, end_time)
            y_flat = y.flatten().numpy()
            
            # Convert to classes using optimized price threshold
            classes = np.ones(len(y_flat), dtype=int)  # Default to no direction
            classes[y_flat > config_dict['price_threshold']] = 2   # Up
            classes[y_flat < -config_dict['price_threshold']] = 0  # Down
            
            # Calculate base weights
            class_counts = Counter(classes)
            total_samples = len(classes)
            
            # Apply multipliers
            class_weights = []
            for i in range(3):
                if class_counts[i] > 0:
                    base_weight = total_samples / (len(class_counts) * class_counts[i])
                    multiplier = config_dict[f'class_multiplier_{i}']
                    final_weight = base_weight * multiplier
                    class_weights.append(final_weight)
                else:
                    class_weights.append(0.0)
            
            return torch.FloatTensor(class_weights)
        
        # Set the custom class weights calculation
        trainer._calculate_class_weights = calculate_weighted_class_weights
        
        # Train model (reduced epochs for hyperparameter optimization)
        original_epochs = trainer.config.num_epochs
        trainer.config.num_epochs = 5  # Reduced for faster optimization
        
        training_history = trainer.train()
        
        # Restore original epochs
        trainer.config.num_epochs = original_epochs
        
        # Evaluate on validation data
        X_val, adj_val, y_val = data_processor.prepare_data(start_time, end_time)
        
        # Convert to classification targets using optimized threshold
        y_val_flat = y_val.flatten().numpy()
        classes_val = np.ones(len(y_val_flat), dtype=int)  # Default to no direction
        classes_val[y_val_flat > config_dict['price_threshold']] = 2   # Up
        classes_val[y_val_flat < -config_dict['price_threshold']] = 0  # Down
        y_val_classes = torch.LongTensor(classes_val.reshape(y_val.shape))
        
        # Evaluate model
        evaluation_results = trainer.evaluate(X_val, y_val_classes)
        
        # Extract metrics for objective calculation
        f1_scores = evaluation_results['f1']  # [down_f1, no_dir_f1, up_f1]
        precision_scores = evaluation_results['precision']  # [down_prec, no_dir_prec, up_prec]
        recall_scores = evaluation_results['recall']  # [down_rec, no_dir_rec, up_rec]
        
        # Calculate log loss for confidence measurement
        probabilities = evaluation_results['probabilities']
        true_labels = evaluation_results['true_labels']
        
        # Calculate multi-class log loss
        def calculate_multiclass_log_loss(y_true, y_pred_proba):
            """Calculate multi-class log loss"""
            epsilon = 1e-15  # Small value to avoid log(0)
            y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
            n_samples = y_true.shape[0]
            n_classes = y_pred_proba.shape[1]
            
            # One-hot encode true labels
            y_true_onehot = np.zeros((n_samples, n_classes))
            y_true_onehot[np.arange(n_samples), y_true] = 1
            
            # Calculate log loss
            log_loss = -np.sum(y_true_onehot * np.log(y_pred_proba)) / n_samples
            return log_loss
        
        log_loss_val = calculate_multiclass_log_loss(true_labels, probabilities)
        
        # Define objective to MINIMIZE: Lower is better
        # Prioritize directional F1s more heavily for 0.5% event-based prediction
        directional_f1_avg_penalty = (1 - ((f1_scores[0] + f1_scores[2]) / 2)) * 3.0  # Weighted more heavily
        no_direction_f1_penalty = (1 - f1_scores[1]) * 1.5  # Less heavily weighted, but still important
        
        # Add confidence penalty (log loss)
        confidence_penalty = log_loss_val * 0.2  # Adjust weight as needed
        
        # Add precision penalty for directional classes (important for trading)
        directional_precision_penalty = (1 - ((precision_scores[0] + precision_scores[2]) / 2)) * 2.0
        
        # Combined objective
        combined_objective = (
            directional_f1_avg_penalty + 
            no_direction_f1_penalty + 
            confidence_penalty + 
            directional_precision_penalty
        )
        
        # Handle cases where F1 scores might be NaN/inf
        if np.any(np.isnan(f1_scores)) or np.any(np.isinf(f1_scores)):
            logger.warning(f"NaN/Inf F1 scores detected: {f1_scores}")
            return float('inf')
        
        # Log trial results for monitoring
        logger.info(f"Trial Results:")
        logger.info(f"  F1 Scores: Down={f1_scores[0]:.4f}, NoDir={f1_scores[1]:.4f}, Up={f1_scores[2]:.4f}")
        logger.info(f"  Precision: Down={precision_scores[0]:.4f}, NoDir={precision_scores[1]:.4f}, Up={precision_scores[2]:.4f}")
        logger.info(f"  Log Loss: {log_loss_val:.4f}")
        logger.info(f"  Combined Objective: {combined_objective:.4f}")
        
        return combined_objective
        
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        return float('inf')  # Return high penalty for failed trials

def main():
    """
    Main function to run enhanced hyperparameter optimization for HPC execution
    """
    # Adjust study creation for distributed/parallel execution (already correctly configured for SQLite)
    study = optuna.create_study(
        direction='minimize',
        study_name='stgnn_enhanced_hyperopt_full_search', # A new name to distinguish studies
        storage='sqlite:///stgnn_enhanced_hyperopt_full_search.db', # New database for full search
        load_if_exists=True
    )

    # Run optimization with parameters designed for comprehensive HPC utilization
    logger.info(f"Starting Optuna optimization with {study.best_trial.params if study.best_trial else 'no'} previous best parameters.")
    study.optimize(
        objective,
        n_trials=2000,  # Significantly increased trials for expanded search space.
                         # This allows more thorough exploration across architectural,
                         # loss function, and feature engineering hyperparameters.
        timeout=None,    # Remove timeout to allow the study to run to completion or n_trials.
                         # Alternatively, set to a very large value (e.g., 24*3600*7 for a week in seconds).
        gc_after_trial=True, # Enable aggressive garbage collection after each trial.
                             # Crucial for long-running studies and managing memory on HPC nodes.
        show_progress_bar=True # Keep for local monitoring, but typically handled by HPC batch system for distributed runs.
    )
    
    # Print results
    print('Best trial:')
    print(f'  Value: {study.best_trial.value}')
    print('  Params:')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
    
    # Save best parameters
    best_params = study.best_trial.params
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'config/stgnn_enhanced_best_params_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Print optimization summary
    print(f'\nOptimization Summary:')
    print(f'  Total trials: {len(study.trials)}')
    print(f'  Best validation loss: {study.best_trial.value:.4f}')
    print(f'  Best focal_alpha: {best_params.get("focal_alpha", "N/A")}')
    print(f'  Best focal_gamma: {best_params.get("focal_gamma", "N/A")}')
    print(f'  Best class multipliers: [{best_params.get("class_multiplier_0", "N/A")}, '
          f'{best_params.get("class_multiplier_1", "N/A")}, {best_params.get("class_multiplier_2", "N/A")}]')

if __name__ == '__main__':
    main() 