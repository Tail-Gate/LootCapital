import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import time
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import itertools
from threading import Thread
import queue
import logging
from logging.handlers import RotatingFileHandler
import copy

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"transformer_strategy_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_file

# Setup logging at the start
log_file = setup_logging()
logging.info(f"Logging to file: {log_file}")

from strategies.transformer_strategy import TransformerStrategy, TransformerConfig
from utils.feature_generator import FeatureGenerator
from utils.model_validation import (
    validate_model,
    perform_walk_forward_analysis,
    visualize_walk_forward_results,
    visualize_parameter_importance,
    visualize_performance_decomposition
)
from utils.technical_indicators import TechnicalIndicators
from utils.transformer_utils import train_transformer

def load_eth_data():
    """Load ETH OHLCV data"""
    data_dir = project_root / "data" / "historical"
    # Load 15-minute ETH-USDT data
    data = pd.read_csv(data_dir / "ETH-USDT-SWAP_ohlcv_15m.csv", index_col=0, parse_dates=True)
    return data

def check_feature_quality(features: pd.DataFrame) -> None:
    """Check quality of generated features"""
    logging.info("\nFeature Quality Check:")
    logging.info(f"Total features: {len(features.columns)}")
    logging.info(f"Missing values: {features.isnull().sum().sum()}")
    logging.info(f"Features with missing values: {features.columns[features.isnull().any()].tolist()}")
    logging.info(f"Features with infinite values: {features.columns[np.isinf(features).any()].tolist()}")

def clean_features(features: pd.DataFrame) -> pd.DataFrame:
    """Clean features by removing NaN and infinite values"""
    logging.info("\nCleaning features...")
    # Replace infinite values with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill NaN values
    features = features.fillna(method='ffill')
    
    # If any NaN values remain, backward fill them
    features = features.fillna(method='bfill')
    
    # If any NaN values still remain, fill with 0
    features = features.fillna(0)
    
    return features

def scale_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler"""
    logging.info("\nScaling features...")
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )
    return scaled_features, scaler

def create_target(data: pd.DataFrame, prediction_horizon: int = 15, min_price_change: float = 0.02) -> pd.Series:
    """Create target variable for prediction with discrete classes
    
    Args:
        data: DataFrame with OHLCV data
        prediction_horizon: Number of periods ahead to predict
        min_price_change: Minimum price change threshold for classification
        
    Returns:
        pd.Series with discrete classes:
        -1: Downward movement (return < -min_price_change)
         0: Neutral movement (|return| <= min_price_change)
         1: Upward movement (return > min_price_change)
    """
    # Calculate future returns
    future_returns = data['close'].shift(-prediction_horizon) / data['close'] - 1
    
    # Convert to discrete classes
    target = pd.Series(0, index=future_returns.index)  # Default to neutral (0)
    target[future_returns > min_price_change] = 1      # Upward movement (1)
    target[future_returns < -min_price_change] = -1    # Downward movement (-1)
    
    return target

def analyze_target_distribution(target: pd.Series) -> None:
    """Analyze distribution of target variable"""
    logging.info("\nTarget Distribution Analysis:")
    value_counts = target.value_counts()
    total = len(target)
    
    logging.info(f"\nTotal samples: {total}")
    logging.info("\nClass Distribution:")
    for value, count in value_counts.items():
        percentage = (count / total) * 100
        logging.info(f"Class {value}: {count} samples ({percentage:.2f}%)")

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string"""
    return str(timedelta(seconds=int(seconds)))

def log_progress(start_time: float, current_step: str, total_steps: int, current_step_num: int) -> None:
    """Log progress with timing information"""
    elapsed = time.time() - start_time
    avg_time_per_step = elapsed / current_step_num if current_step_num > 0 else 0
    remaining_steps = total_steps - current_step_num
    estimated_remaining = avg_time_per_step * remaining_steps
    
    logging.info(f"\nProgress Update - {current_step}")
    logging.info(f"Step {current_step_num}/{total_steps}")
    logging.info(f"Elapsed time: {format_time(elapsed)}")
    logging.info(f"Average time per step: {format_time(avg_time_per_step)}")
    logging.info(f"Estimated remaining time: {format_time(estimated_remaining)}")
    logging.info(f"Estimated completion: {datetime.now() + timedelta(seconds=estimated_remaining)}")

def perform_walk_forward_analysis(
    model_class,
    model_config_base,
    features,
    targets,
    initial_train_size,
    step_size,
    n_splits_outer,
    hyperparameter_grid,
    n_splits_inner
):
    """Perform walk-forward analysis with progress tracking"""
    start_time = time.time()
    total_steps = n_splits_outer * n_splits_inner * len(list(itertools.product(*hyperparameter_grid.values())))
    current_step = 0
    
    logging.info(f"\nStarting walk-forward analysis at {datetime.now()}")
    logging.info(f"Total steps to complete: {total_steps}")
    
    results = {
        'accuracy': [],
        'balanced_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'best_params': []
    }
    
    # Calculate split points
    total_data_points = len(features)
    split_points = [initial_train_size + i * step_size for i in range(n_splits_outer)]
    
    logging.info("\nWalk-forward analysis configuration:")
    logging.info(f"Total data points: {total_data_points}")
    logging.info(f"Initial train size: {initial_train_size}")
    logging.info(f"Step size: {step_size}")
    logging.info(f"Split points: {split_points}")
    
    for outer_fold in range(n_splits_outer):
        logging.info(f"\n--- Outer Fold {outer_fold + 1}/{n_splits_outer} ---")
        current_step += 1
        log_progress(start_time, f"Outer Fold {outer_fold + 1}", total_steps, current_step)
        
        # Calculate train and test indices
        train_end = split_points[outer_fold]
        test_end = train_end + step_size
        
        # Get train and test data
        train_features = features.iloc[:train_end]
        train_targets = targets.iloc[:train_end]
        test_features = features.iloc[train_end:test_end]
        test_targets = targets.iloc[train_end:test_end]
        
        logging.info(f"  Outer Train: {train_features.index[0]} to {train_features.index[-1]} ({len(train_features)} samples)")
        logging.info(f"  Outer Test: {test_features.index[0]} to {test_features.index[-1]} ({len(test_features)} samples)")
        
        # Perform inner loop for hyperparameter optimization
        logging.info("  Performing inner loop for hyperparameter optimization...")
        best_params = None
        best_score = float('-inf')
        
        for inner_fold in range(n_splits_inner):
            current_step += 1
            log_progress(start_time, f"Inner Fold {inner_fold + 1}", total_steps, current_step)
            
            # Test each hyperparameter combination
            for params in itertools.product(*hyperparameter_grid.values()):
                param_dict = dict(zip(hyperparameter_grid.keys(), params))
                logging.info(f"  Testing params: {param_dict}")
                
                # Create model with current parameters
                model_config = copy.deepcopy(model_config_base)
                for key, value in param_dict.items():
                    setattr(model_config, key, value)
                model = model_class(model_config)
                
                # Train and evaluate
                fold_start_time = time.time()
                model.train(train_features, train_targets)
                fold_time = time.time() - fold_start_time
                
                logging.info(f"  Fold training time: {format_time(fold_time)}")
                
                # Evaluate and update best parameters
                score = model.evaluate(test_features, test_targets)
                if score > best_score:
                    best_score = score
                    best_params = param_dict
        
        # Train final model with best parameters
        logging.info(f"\nTraining final model with best parameters: {best_params}")
        final_model_config = copy.deepcopy(model_config_base)
        for key, value in best_params.items():
            setattr(final_model_config, key, value)
        final_model = model_class(final_model_config)
        final_model.train(train_features, train_targets)
        
        # Evaluate on test set
        metrics = final_model.evaluate(test_features, test_targets, return_metrics=True)
        for metric, value in metrics.items():
            results[metric].append(value)
        results['best_params'].append(best_params)
        
        # Log fold results
        logging.info(f"\nFold {outer_fold + 1} Results:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
    
    total_time = time.time() - start_time
    logging.info(f"\nWalk-forward analysis completed in {format_time(total_time)}")
    return results

def main():
    start_time = time.time()
    
    # Load data
    logging.info("Loading ETH-USDT 15m data...")
    data = load_eth_data()
    logging.info(f"Loaded data from {data.index[0]} to {data.index[-1]}")
    logging.info(f"Total number of 15-minute periods: {len(data)}")
    
    # Initialize feature generator
    logging.info("\nInitializing feature generator...")
    feature_generator = FeatureGenerator()
    
    # Generate features
    logging.info("Generating features...")
    features = feature_generator.generate_features(data)
    logging.info("Features generated successfully")
    
    # Print available features
    logging.info("\nAvailable features:")
    logging.info(features.columns.tolist())
    
    # Check feature quality
    check_feature_quality(features)
    logging.info("Feature quality check completed")
    
    # Clean features
    features = clean_features(features)
    logging.info("Features cleaned")
    
    # Scale features
    features, scaler = scale_features(features)
    logging.info("Features scaled")
    
    # Create target variable
    logging.info("\nCreating target variable (predicting 15 periods ahead)...")
    target = create_target(data, prediction_horizon=15)
    logging.info("Target variable created")
    
    # Align target with cleaned features
    target = target.loc[features.index]
    logging.info("Target aligned with features")
    
    # Analyze target distribution
    analyze_target_distribution(target)
    logging.info("Target distribution analyzed")
    
    # Initialize transformer strategy
    logging.info("\nInitializing Transformer Strategy...")
    base_config = TransformerConfig(
        input_size=len(features.columns),
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        sequence_length=20,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=1e-5,
        early_stopping_patience=10,
        num_epochs=50,
        validation_split=0.2,
        focal_loss_gamma=2.0
    )
    logging.info("Base config created")
    
    # Define hyperparameter grid for optimization
    hyperparameter_grid = {
        'd_model': [32, 64],  # Two distinct values
        'num_layers': [1, 2],  # Two distinct values
        'nhead': [4],     # Fixed value
        'dim_feedforward': [128], # Fixed value
        'dropout': [0.1],  # Fixed value
        'sequence_length': [20]  # Fixed value
    }
    logging.info("Hyperparameter grid defined")
    
    # Configure walk-forward parameters
    total_data_points = len(features)
    initial_train_size = int(0.6 * total_data_points)
    step_size = int(0.1 * total_data_points)
    n_splits_outer = 2  # Reduced from 3 to 2
    n_splits_inner = 2  # Reduced from 3 to 2
    
    # Calculate total combinations and training sessions
    total_combinations = 1  # Since we're using single values
    total_training_sessions = n_splits_outer * n_splits_inner * total_combinations
    
    logging.info(f"\nStarting walk-forward analysis with {n_splits_outer} outer folds and {n_splits_inner} inner folds")
    logging.info(f"Total hyperparameter combinations to test: {total_combinations}")
    logging.info(f"Total training sessions: {total_training_sessions}")
    logging.info("About to start walk-forward analysis")
    
    try:
        logging.info("Entering walk-forward analysis")
        walk_forward_results = perform_walk_forward_analysis(
            model_class=TransformerStrategy,
            model_config_base=base_config,
            features=features,
            targets=target,
            initial_train_size=initial_train_size,
            step_size=step_size,
            n_splits_outer=n_splits_outer,
            hyperparameter_grid=hyperparameter_grid,
            n_splits_inner=n_splits_inner
        )
        logging.info("Walk-forward analysis completed")
    except Exception as e:
        logging.error(f"\nError during walk-forward analysis: {str(e)}")
        logging.error(f"Error occurred at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        raise
    
    # Print walk-forward results
    logging.info("\nWalk-Forward Analysis Results:")
    for metric, values in walk_forward_results.items():
        if metric != 'best_params':
            logging.info(f"\n{metric}:")
            logging.info(f"  Mean: {np.mean(values):.4f}")
            logging.info(f"  Std: {np.std(values):.4f}")
            logging.info(f"  Values: {[f'{v:.4f}' for v in values]}")
    
    logging.info("\nBest Parameters for Each Fold:")
    for i, params in enumerate(walk_forward_results['best_params']):
        logging.info(f"\nFold {i+1}:")
        for param, value in params.items():
            logging.info(f"  {param}: {value}")
    
    # Create visualization directory
    viz_dir = project_root / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate and save visualizations with progress bar
    logging.info("\nGenerating visualizations...")
    with tqdm(total=3, desc="Generating visualizations") as pbar:
        # 1. Main walk-forward results
        logging.info("Generating walk-forward results plot...")
        visualize_walk_forward_results(
            walk_forward_results,
            save_path=str(viz_dir / "transformer_walk_forward_results.png")
        )
        pbar.update(1)
        
        # 2. Parameter importance
        logging.info("Generating parameter importance plot...")
        visualize_parameter_importance(
            walk_forward_results,
            save_path=str(viz_dir / "transformer_parameter_importance.png")
        )
        pbar.update(1)
        
        # 3. Performance decomposition
        logging.info("Generating performance decomposition plot...")
        visualize_performance_decomposition(
            walk_forward_results,
            save_path=str(viz_dir / "transformer_performance_decomposition.png")
        )
        pbar.update(1)
    
    # Save the scaler for future use
    scaler_path = project_root / "models" / "transformer_feature_scaler.pkl"
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"\nScaler saved to: {scaler_path}")
    
    # Save features and target
    logging.info("\nSaving features and target...")
    features_path = project_root / "models" / "transformer_features.parquet"
    target_path = project_root / "models" / "transformer_target.parquet"
    features.to_parquet(features_path)
    # Convert target Series to DataFrame before saving
    target_df = pd.DataFrame(target, columns=['target'])
    target_df.to_parquet(target_path)
    logging.info(f"Features saved to: {features_path}")
    logging.info(f"Target saved to: {target_path}")
    
    # Print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main() 