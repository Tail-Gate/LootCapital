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

def get_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")
        return device
    else:
        device = torch.device('cpu')
        logger.info("GPU not available, using CPU")
        return device

def manage_memory():
    """Force garbage collection and log memory usage for GPU training"""
    gc.collect()
    
    # Log CPU memory usage
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"CPU Memory usage: {memory_mb:.1f} MB")
    
    # Log GPU memory usage if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory_mb = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            gpu_memory_cached = torch.cuda.memory_reserved(i) / 1024 / 1024
            logger.info(f"GPU {i} Memory: {gpu_memory_allocated:.1f}MB allocated, {gpu_memory_cached:.1f}MB cached, {gpu_memory_mb:.1f}MB total")
    
    # Force more aggressive cleanup
    if hasattr(sys, 'exc_clear'):
        sys.exc_clear()
    
    # Additional memory optimization for HPC
    if memory_mb > 1000:  # Warning if memory usage > 1GB
        logger.warning(f"High CPU memory usage detected: {memory_mb:.1f} MB")
        # Force more aggressive cleanup
        gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    Ultra-minimal memory-optimized objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Validation loss (weighted focal loss or combined metric)
    """
    # Force memory cleanup at start of trial
    manage_memory()
    
    try:
        # Define ULTRA-MINIMAL hyperparameter search space to prevent OOM
        config_dict = {
            'assets': ['ETH/USD'],  # Focus on single asset for optimization
            'features': [
                'returns', 'log_returns',
                'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'volume', 'volume_ma', 'volume_std', 'volume_surge', 'volume_ratio',
                'ma_crossover', 'price_momentum', 'volatility_regime',
                'support', 'resistance', 'breakout_intensity',
                'vwap_ratio', 'cumulative_delta'
            ],  # Full feature set (23 features, removed problematic adx)
            
            # ULTRA-MINIMAL parameter ranges to prevent OOM
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),  # Very small range
            'hidden_dim': trial.suggest_int('hidden_dim', 4, 6, step=2),  # Ultra-small range
            'num_layers': trial.suggest_int('num_layers', 1, 1),  # Single layer only
            'kernel_size': trial.suggest_int('kernel_size', 2, 2),  # Fixed small value
            'dropout': trial.suggest_float('dropout', 0.1, 0.1),  # Fixed small value
            'batch_size': trial.suggest_int('batch_size', 1, 2, step=1),  # Ultra-small batches
            'seq_len': trial.suggest_int('seq_len', 5, 10, step=5),  # Ultra-short sequences
            'prediction_horizon': 15,  # Fixed as per current requirement
            'early_stopping_patience': 2,  # Very short patience
            
            # Focal Loss parameters (minimal ranges)
            'focal_alpha': trial.suggest_float('focal_alpha', 0.8, 1.0),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.5, 2.0),
            
            # Class multipliers (minimal ranges)
            'class_multiplier_0': trial.suggest_float('class_multiplier_0', 1.0, 2.0),  # Down class
            'class_multiplier_1': trial.suggest_float('class_multiplier_1', 1.0, 1.5),
            'class_multiplier_2': trial.suggest_float('class_multiplier_2', 1.0, 2.0),  # Up class
            
            # Price threshold (fixed)
            'price_threshold': 0.005,  # Fixed 0.5% threshold for classification
            
            # ULTRA-MINIMAL feature engineering parameters (fixed values)
            'rsi_period': 10,  # Fixed minimal value
            'macd_fast_period': 10,  # Fixed minimal value
            'macd_slow_period': 20,  # Fixed minimal value
            'macd_signal_period': 7,  # Fixed minimal value
            'bb_period': 15,  # Fixed minimal value
            'bb_num_std_dev': 2.0,  # Fixed value
            'atr_period': 10,  # Fixed minimal value
            'adx_period': 10,  # Fixed minimal value
            'volume_ma_period': 15,  # Fixed minimal value
            'price_momentum_lookback': 3,  # Fixed minimal value
        }
        
        # Create STGNNConfig with ULTRA-MINIMAL parameters
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
        
        # Dynamically determine the full data range (5 years)
        data_file_path = 'data/historical/ETH-USDT-SWAP_ohlcv_15m.csv'
        try:
            # Read the data file to get the full date range
            data_df = pd.read_csv(data_file_path)
            data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
            latest_date = data_df['timestamp'].max()
            earliest_date = data_df['timestamp'].min()
            
            # Use the full data range (5 years)
            end_time = latest_date
            start_time = earliest_date
            
            logger.info(f"Data file date range: {earliest_date} to {latest_date}")
            logger.info(f"Using full 5-year data range: {start_time} to {end_time}")
            logger.info(f"Total data span: {(end_time - start_time).days} days")
            
        except Exception as e:
            logger.warning(f"Could not read data file to determine date range: {e}")
            # Fallback to a known good date range
            end_time = datetime(2025, 3, 15, 12, 0, 0)
            start_time = end_time - timedelta(days=1)
            logger.info(f"Using fallback date range: {start_time} to {end_time}")
        
        # Create data processor with memory-efficient approach
        data_processor = create_classification_data_processor(config)
        
        # Get device for training
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Create trainer with ultra-minimal parameters
        trainer = ClassificationSTGNNTrainer(
            config=config,
            data_processor=data_processor,
            price_threshold=config_dict['price_threshold'],
            focal_alpha=config_dict['focal_alpha'],
            focal_gamma=config_dict['focal_gamma'],
            class_weights=None,  # Will be calculated with multipliers
            start_time=start_time,
            end_time=end_time,
            device=device  # Pass device to trainer
        )
        
        # Validate trainer creation
        if trainer is None:
            logger.error("Trainer creation failed")
            return float('inf')
            
        # Ensure model is on correct device
        if hasattr(trainer, 'model') and trainer.model is not None:
            trainer.model = trainer.model.to(device)
            logger.info(f"Model moved to device: {device}")
        else:
            logger.error("Trainer model is None")
            return float('inf')
        
        # DISABLE SMOTE for hyperparameter optimization to prevent memory explosion
        # Override the train method to skip SMOTE processing
        original_train = trainer.train
        
        def train_without_smote():
            """Train without SMOTE to save memory during hyperparameter optimization"""
            # Use memory-efficient data loading
            logger.info("Using memory-efficient data loading for hyperparameter optimization")
            
            # Load data in chunks using the data processor's memory-efficient methods
            X, adj, y = data_processor.prepare_data(start_time, end_time)
            
            # Validate data shapes
            if X is None or adj is None or y is None:
                logger.error("Data preparation returned None values")
                return float('inf')
                
            logger.info(f"Data shapes - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
            
            # Convert to classification targets using optimized price threshold
            y_flat = y.flatten().numpy()
            classes = np.ones(len(y_flat), dtype=int)  # Default to no direction
            classes[y_flat > config_dict['price_threshold']] = 2   # Up
            classes[y_flat < -config_dict['price_threshold']] = 0  # Down
            y_classes = torch.LongTensor(classes.reshape(y.shape))
            
            logger.info(f"Classification targets shape: {y_classes.shape}")
            logger.info(f"Class distribution: {np.bincount(classes)}")
            
            # Split data using memory-efficient approach
            X_train, y_train, X_val, y_val = data_processor.split_data(X, y_classes)
            
            # Skip SMOTE - use original data directly
            logger.info("Skipping SMOTE for memory efficiency during hyperparameter optimization")
            
            # Create dataloaders with original (unbalanced) training data
            train_loader = data_processor.create_dataloader(X_train, y_train, drop_last=True)
            val_loader = data_processor.create_dataloader(X_val, y_val, drop_last=False)
            
            # Training loop
            patience_counter = 0
            
            for epoch in range(trainer.config.num_epochs):
                # Train epoch
                train_loss, train_acc = trainer.train_epoch(train_loader)
                trainer.train_losses.append(train_loss)
                trainer.train_accuracies.append(train_acc)
                
                # Validate
                val_loss, val_acc = trainer.validate(val_loader)
                trainer.val_losses.append(val_loss)
                trainer.val_accuracies.append(val_acc)
                
                # Early stopping
                if val_loss < trainer.best_val_loss:
                    trainer.best_val_loss = val_loss
                    trainer.best_model_state = trainer.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= trainer.config.early_stopping_patience:
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break
                    
                if (epoch + 1) % 5 == 0:  # Log every 5 epochs for ultra-minimal training
                    logger.info(f'Epoch {epoch + 1}/{trainer.config.num_epochs}:')
                    logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                    logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                    
            # Load best model
            if trainer.best_model_state is not None:
                trainer.model.load_state_dict(trainer.best_model_state)
                
            return {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'train_accuracies': trainer.train_accuracies,
                'val_accuracies': trainer.val_accuracies,
                'smote_info': {
                    'class_distribution_before': 'Skipped for memory efficiency',
                    'class_distribution_after': 'Skipped for memory efficiency',
                    'original_batch_size': X_train.shape[0],
                    'balanced_batch_size': X_train.shape[0]  # No change since SMOTE skipped
                }
            }
        
        # Replace the train method
        trainer.train = train_without_smote
        
        # Override class weights calculation with multipliers
        def calculate_weighted_class_weights():
            """Calculate class weights with trial multipliers using memory-efficient loading"""
            logger.info("Calculating class weights using memory-efficient data loading...")
            # Get class distribution from training data using memory-efficient loading
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
        
        # Train model with ULTRA-FEW epochs to prevent OOM
        original_epochs = trainer.config.num_epochs
        trainer.config.num_epochs = 1  # Ultra-minimal: only 1 epoch
        
        # Force memory cleanup before training
        manage_memory()
        
        # Add validation to ensure training completes successfully
        try:
            training_history = trainer.train()
            
            # Validate training history
            if training_history is None:
                logger.error("Training returned None - this indicates a critical error")
                return float('inf')
                
            # Check if training actually happened
            if 'train_losses' not in training_history or len(training_history['train_losses']) == 0:
                logger.error("No training losses recorded - training may have failed")
                return float('inf')
                
            logger.info(f"Training completed successfully with {len(training_history['train_losses'])} epochs")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            return float('inf')
        
        # Force memory cleanup after training
        manage_memory()
        
        # Restore original epochs
        trainer.config.num_epochs = original_epochs
        
        # Evaluate on validation data using memory-efficient loading
        logger.info("Evaluating on validation data using memory-efficient loading...")
        X_val, adj_val, y_val = data_processor.prepare_data(start_time, end_time)
        
        # Convert to classification targets using optimized threshold
        y_val_flat = y_val.flatten().numpy()
        classes_val = np.ones(len(y_val_flat), dtype=int)  # Default to no direction
        classes_val[y_val_flat > config_dict['price_threshold']] = 2   # Up
        classes_val[y_val_flat < -config_dict['price_threshold']] = 0  # Down
        y_val_classes = torch.LongTensor(classes_val.reshape(y_val.shape))
        
        # Evaluate model with validation
        try:
            evaluation_results = trainer.evaluate(X_val, y_val_classes)
            
            # Validate evaluation results
            if evaluation_results is None:
                logger.error("Evaluation returned None - this indicates a critical error")
                return float('inf')
                
            # Check required keys exist
            required_keys = ['f1', 'precision', 'recall', 'probabilities', 'true_labels']
            for key in required_keys:
                if key not in evaluation_results:
                    logger.error(f"Evaluation results missing key: {key}")
                    return float('inf')
                    
            logger.info("Evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation failed with error: {e}")
            return float('inf')
        
        # Force memory cleanup after evaluation
        manage_memory()
        
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
        logger.info(f"Ultra-Minimal Trial Results:")
        logger.info(f"  Hidden dim: {config_dict['hidden_dim']}, Layers: {config_dict['num_layers']}")
        logger.info(f"  Batch size: {config_dict['batch_size']}, Seq len: {config_dict['seq_len']}")
        logger.info(f"  Data window: 3 days, Epochs: 1")
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
        logger.error(f"Error in ultra-minimal objective function: {e}")
        # Force memory cleanup on error
        manage_memory()
        return float('inf')  # Return high penalty for failed trials

def main():
    """
    Main function to run memory-optimized hyperparameter optimization for HPC execution
    """
    # Log initial device and memory information
    device = get_device()
    logger.info(f"Starting hyperparameter optimization with device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU Information:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}")
            logger.info(f"  Memory: {props.total_memory / 1024 / 1024 / 1024:.1f} GB")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Log initial memory usage
    manage_memory()
    
    # Monitor memory usage throughout optimization
    def memory_monitor():
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"Current CPU memory usage: {memory_gb:.2f} GB")
        
        # Monitor GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory_gb = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024
                gpu_memory_total_gb = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                logger.info(f"GPU {i} memory usage: {gpu_memory_gb:.2f} GB / {gpu_memory_total_gb:.2f} GB")
                
                if gpu_memory_gb > gpu_memory_total_gb * 0.8:  # Warning at 80% GPU memory
                    logger.warning(f"GPU {i} memory usage is high: {gpu_memory_gb:.2f} GB / {gpu_memory_total_gb:.2f} GB")
        
        if memory_gb > 100:  # Warning at 100GB
            logger.warning(f"CPU memory usage is high: {memory_gb:.2f} GB")
        return memory_gb
    
    # Log initial memory
    memory_monitor()
    
    # Adjust study creation for distributed/parallel execution with ultra-minimal memory optimization
    device_suffix = "gpu" if device.type == 'cuda' else "cpu"
    study = optuna.create_study(
        direction='minimize',
        study_name=f'stgnn_ultra_minimal_hyperopt_{device_suffix}',  # Include device in study name
        storage=f'sqlite:///stgnn_ultra_minimal_hyperopt_{device_suffix}.db',  # Include device in database name
        load_if_exists=True
    )

    # Run optimization with ULTRA-MINIMAL parameters for memory efficiency
    # Check if there are any completed trials before trying to access best_trial
    best_params_info = "no"
    if study.trials:  # Checks if the list of trials is not empty
        try:
            best_params_info = study.best_trial.params
        except ValueError:
            # This catch handles cases where trials exist but none are in a 'COMPLETE' state yet
            best_params_info = "no (no completed trials yet)"
    logger.info(f"Starting ultra-minimal Optuna optimization with {best_params_info} previous best parameters.")
    
    # ULTRA-MINIMAL number of trials to prevent OOM
    study.optimize(
        objective,
        n_trials=10,  # Ultra-minimal: only 10 trials to prevent memory issues
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
    device_suffix = "gpu" if device.type == 'cuda' else "cpu"
    with open(f'config/stgnn_ultra_minimal_best_params_{device_suffix}_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Print optimization summary
    device_suffix = "GPU" if device.type == 'cuda' else "CPU"
    print(f'\nUltra-Minimal Optimization Summary ({device_suffix}):')
    print(f'  Device: {device}')
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