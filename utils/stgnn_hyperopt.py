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
import gc  # For garbage collection
import psutil  # For memory monitoring
import os
import sys

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

def manage_memory():
    """Force garbage collection and log memory usage"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Log memory usage
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {memory_mb:.1f} MB")
    
    # Force more aggressive cleanup
    if hasattr(sys, 'exc_clear'):
        sys.exc_clear()
    
    # Additional memory optimization for HPC
    if memory_mb > 1000:  # Warning if memory usage > 1GB
        logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
        # Force more aggressive cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
    Memory-optimized objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Validation loss (weighted focal loss or combined metric)
    """
    # Force memory cleanup at start of trial
    manage_memory()
    
    try:
        # Define REDUCED hyperparameter search space to minimize memory usage
        config_dict = {
            'assets': ['ETH/USD'],  # Focus on single asset for optimization
            'features': ['price', 'volume', 'rsi', 'macd', 'bollinger', 'atr', 'adx', 'stoch', 'williams_r', 'cci', 'mfi', 'obv', 'vwap', 'support', 'resistance'],
            
            # REDUCED parameter ranges to minimize memory footprint
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),  # Reduced upper bound
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 96, step=16),  # Reduced range
            'num_layers': trial.suggest_int('num_layers', 1, 3),  # Reduced max layers
            'kernel_size': trial.suggest_int('kernel_size', 2, 4),  # Reduced range
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),  # Reduced range
            'batch_size': trial.suggest_int('batch_size', 8, 32, step=8),  # Reduced batch sizes
            'seq_len': trial.suggest_int('seq_len', 20, 50, step=10),  # Reduced range for memory efficiency
            'prediction_horizon': 15,  # Fixed as per current requirement
            'early_stopping_patience': 3,  # Reduced for faster convergence
            
            # Focal Loss parameters (reduced ranges)
            'focal_alpha': trial.suggest_float('focal_alpha', 0.5, 1.0),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.5, 3.0),
            
            # Class multipliers (reduced ranges)
            'class_multiplier_0': trial.suggest_float('class_multiplier_0', 1.0, 4.0),  # Down class
            'class_multiplier_1': trial.suggest_float('class_multiplier_1', 1.0, 3.0),  # No Direction class
            'class_multiplier_2': trial.suggest_float('class_multiplier_2', 1.0, 4.0),  # Up class
            
            # Price threshold (fixed)
            'price_threshold': 0.005,  # Fixed 0.5% threshold for classification
            
            # REDUCED feature engineering parameters to minimize memory usage
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),  # Reduced range
            'macd_fast_period': trial.suggest_int('macd_fast_period', 10, 16),  # Reduced range
            'macd_slow_period': trial.suggest_int('macd_slow_period', 20, 30),  # Reduced range
            'macd_signal_period': trial.suggest_int('macd_signal_period', 7, 12),  # Reduced range
            'bb_period': trial.suggest_int('bb_period', 15, 25),  # Reduced range
            'bb_num_std_dev': trial.suggest_float('bb_num_std_dev', 1.8, 2.5),  # Reduced range
            'atr_period': trial.suggest_int('atr_period', 10, 16),  # Reduced range
            'adx_period': trial.suggest_int('adx_period', 10, 16),  # Reduced range
            'volume_ma_period': trial.suggest_int('volume_ma_period', 15, 25),  # Reduced range
            'price_momentum_lookback': trial.suggest_int('price_momentum_lookback', 5, 10),  # Reduced range
        }
        
        # Create STGNNConfig with REDUCED parameters
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
        
        # Use SMALLER time window for hyperparameter optimization to reduce memory usage
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # Reduced from 60 to 30 days for memory efficiency
        
        # Create data processor with memory-efficient approach
        data_processor = create_classification_data_processor(config)
        
        # Create trainer with optimized parameters
        trainer = ClassificationSTGNNTrainer(
            config=config,
            data_processor=data_processor,
            price_threshold=config_dict['price_threshold'],
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
        
        # Train model with REDUCED epochs for faster optimization and less memory usage
        original_epochs = trainer.config.num_epochs
        trainer.config.num_epochs = 3  # Reduced from 5 to 3 for faster optimization
        
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
        
        # Force memory cleanup after trial
        manage_memory()
        
        # Additional cleanup for HPC environment
        del trainer, data_processor, config
        if 'X_val' in locals():
            del X_val, adj_val, y_val, y_val_classes
        if 'evaluation_results' in locals():
            del evaluation_results, probabilities, true_labels
        if 'f1_scores' in locals():
            del f1_scores, precision_scores, recall_scores
        if 'log_loss_val' in locals():
            del log_loss_val
        
        return combined_objective
        
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        # Force memory cleanup on error
        manage_memory()
        return float('inf')  # Return high penalty for failed trials

def main():
    """
    Main function to run memory-optimized hyperparameter optimization for HPC execution
    """
    # Log initial memory usage
    manage_memory()
    
    # Monitor memory usage throughout optimization
    def memory_monitor():
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"Current memory usage: {memory_gb:.2f} GB")
        if memory_gb > 100:  # Warning at 100GB
            logger.warning(f"Memory usage is high: {memory_gb:.2f} GB")
        return memory_gb
    
    # Log initial memory
    memory_monitor()
    
    # Adjust study creation for distributed/parallel execution with memory optimization
    study = optuna.create_study(
        direction='minimize',
        study_name='stgnn_memory_optimized_hyperopt',  # New name for memory-optimized version
        storage='sqlite:///stgnn_memory_optimized_hyperopt.db',  # New database for memory-optimized version
        load_if_exists=True
    )

    # Run optimization with REDUCED parameters for memory efficiency
    # Check if there are any completed trials before trying to access best_trial
    best_params_info = "no"
    if study.trials:  # Checks if the list of trials is not empty
        try:
            best_params_info = study.best_trial.params
        except ValueError:
            # This catch handles cases where trials exist but none are in a 'COMPLETE' state yet
            best_params_info = "no (no completed trials yet)"
    logger.info(f"Starting memory-optimized Optuna optimization with {best_params_info} previous best parameters.")
    
    # REDUCED number of trials to minimize memory usage and prevent OOM
    study.optimize(
        objective,
        n_trials=200,  # Reduced from 500 to 200 for memory efficiency
        timeout=None,    # Remove timeout to allow the study to run to completion or n_trials.
                         # Alternatively, set to a very large value (e.g., 24*3600*7 for a week in seconds).
        gc_after_trial=True, # Enable aggressive garbage collection after each trial.
                             # Crucial for long-running studies and managing memory on HPC nodes.
        show_progress_bar=True, # Keep for local monitoring, but typically handled by HPC batch system for distributed runs.
        callbacks=[lambda study, trial: memory_monitor()]  # Monitor memory after each trial
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
    with open(f'config/stgnn_memory_optimized_best_params_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Print optimization summary
    print(f'\nMemory-Optimized Optimization Summary:')
    print(f'  Total trials: {len(study.trials)}')
    print(f'  Best validation loss: {study.best_trial.value:.4f}')
    print(f'  Best focal_alpha: {best_params.get("focal_alpha", "N/A")}')
    print(f'  Best focal_gamma: {best_params.get("focal_gamma", "N/A")}')
    print(f'  Best class multipliers: [{best_params.get("class_multiplier_0", "N/A")}, '
          f'{best_params.get("class_multiplier_1", "N/A")}, {best_params.get("class_multiplier_2", "N/A")}]')
    
    # Final memory cleanup
    manage_memory()

if __name__ == '__main__':
    main() 