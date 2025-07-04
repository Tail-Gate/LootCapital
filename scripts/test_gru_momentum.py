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

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategies.gru_momentum_strategy import GRUMomentumStrategy, GRUMomentumConfig
from utils.feature_generator import FeatureGenerator
from utils.model_validation import validate_model, perform_walk_forward_analysis, visualize_walk_forward_results, visualize_parameter_importance, visualize_performance_decomposition
from utils.technical_indicators import TechnicalIndicators

def load_eth_data():
    """Load ETH OHLCV data"""
    data_dir = project_root / "data" / "historical"
    # Load 15-minute ETH-USDT data
    data = pd.read_csv(data_dir / "ETH-USDT-SWAP_ohlcv_15m.csv", index_col=0, parse_dates=True)
    return data

def create_target(data: pd.DataFrame, 
                 prediction_horizon: int = 15,  # Number of 15m periods to look ahead (3.75 hours)
                 min_price_change: float = 0.02) -> pd.Series:  # 2% minimum price change for longer horizon
    """
    Create target variable for momentum prediction
    
    Args:
        data: DataFrame with OHLCV data
        prediction_horizon: Number of periods to look ahead (default: 15 = 3.75 hour)
        min_price_change: Minimum price change threshold (default: 2%)
    
    Returns:
        Series with target values:
        1: Significant upward movement (>2%)
        0: No significant movement (<2% in either direction)
        -1: Significant downward movement (<-2%)
    """
    # Calculate future returns over the prediction horizon
    future_returns = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
    
    # Create target based on price movement thresholds
    target = pd.Series(0, index=data.index)  # Initialize with 0 (no significant movement)
    target[future_returns > min_price_change] = 1  # Significant upward movement
    target[future_returns < -min_price_change] = -1  # Significant downward movement
    
    return target

def analyze_target_distribution(target: pd.Series) -> None:
    """Analyze the distribution of target values"""
    total_samples = len(target)
    up_moves = (target == 1).sum()
    down_moves = (target == -1).sum()
    no_moves = (target == 0).sum()
    
    print("\nTarget Distribution Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Significant upward movements (>2%): {up_moves} ({up_moves/total_samples*100:.2f}%)")
    print(f"Significant downward movements (<-2%): {down_moves} ({down_moves/total_samples*100:.2f}%)")
    print(f"No significant movements: {no_moves} ({no_moves/total_samples*100:.2f}%)")

def check_feature_quality(features: pd.DataFrame) -> None:
    """Check for NaN and infinite values in features"""
    print("\nFeature Quality Analysis:")
    
    # Check for NaN values
    nan_counts = features.isna().sum()
    if nan_counts.any():
        print("\nFeatures with NaN values:")
        for feature, count in nan_counts[nan_counts > 0].items():
            print(f"{feature}: {count} NaN values ({count/len(features)*100:.2f}%)")
    else:
        print("No NaN values found in features.")
    
    # Check for infinite values
    inf_counts = pd.Series({
        col: np.isinf(features[col]).sum() 
        for col in features.select_dtypes(include=[np.number]).columns
    })
    if inf_counts.any():
        print("\nFeatures with infinite values:")
        for feature, count in inf_counts[inf_counts > 0].items():
            print(f"{feature}: {count} infinite values ({count/len(features)*100:.2f}%)")
    else:
        print("No infinite values found in features.")
    
    # Check value ranges
    print("\nFeature value ranges:")
    for col in features.select_dtypes(include=[np.number]).columns:
        min_val = features[col].min()
        max_val = features[col].max()
        mean_val = features[col].mean()
        std_val = features[col].std()
        print(f"{col}:")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std: {std_val:.4f}")

def clean_features(features: pd.DataFrame) -> pd.DataFrame:
    """Clean features by removing rows with NaN values"""
    original_len = len(features)
    cleaned_features = features.dropna()
    removed_rows = original_len - len(cleaned_features)
    
    print(f"\nCleaning features:")
    print(f"Original dataset size: {original_len}")
    print(f"Rows removed due to NaN values: {removed_rows} ({removed_rows/original_len*100:.2f}%)")
    print(f"Final dataset size: {len(cleaned_features)}")
    
    return cleaned_features

def scale_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler"""
    print("\nScaling features...")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale features
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    # Print scaling statistics
    print("\nFeature scaling statistics:")
    for col in features.columns:
        print(f"\n{col}:")
        print(f"  Original mean: {features[col].mean():.4f}")
        print(f"  Original std: {features[col].std():.4f}")
        print(f"  Scaled mean: {scaled_features[col].mean():.4f}")
        print(f"  Scaled std: {scaled_features[col].std():.4f}")
    
    return scaled_features, scaler

def main():
    start_time = time.time()
    
    # Load data
    print("Loading ETH-USDT 15m data...")
    data = load_eth_data()
    print(f"Loaded data from {data.index[0]} to {data.index[-1]}")
    print(f"Total number of 15-minute periods: {len(data)}")
    
    # Initialize feature generator
    print("\nInitializing feature generator...")
    feature_generator = FeatureGenerator()  # Initialize without data
    
    # Generate features with progress bar
    print("Generating features...")
    features = feature_generator.generate_features(data)  # Pass data to generate_features method
    
    # Print available features
    print("\nAvailable features:")
    print(features.columns.tolist())
    
    # Check feature quality
    check_feature_quality(features)
    
    # Clean features by removing NaN values
    features = clean_features(features)
    
    # Scale features
    features, scaler = scale_features(features)
    
    # Create target variable
    print("\nCreating target variable (predicting 15 periods ahead, min change: 2.0%)...")
    target = create_target(data, prediction_horizon=15, min_price_change=0.02)
    
    # Align target with cleaned features
    target = target.loc[features.index]
    
    # Analyze target distribution
    analyze_target_distribution(target)
    
    # Initialize base configuration
    print("\nInitializing GRU Momentum Strategy...")
    config = GRUMomentumConfig(
        model_path=str(project_root / "models" / "gru_momentum.pt"),
    )
    
    # Define hyperparameter grid for optimization
    hyperparameter_grid = {
        'gru_hidden_dim': [64, 128],        # Hidden layer dimensions
        'learning_rate': [0.0001, 0.0005],   # Adam optimizer learning rates
        'batch_size': [32, 64],             # Training batch sizes
        'early_stopping_patience': [5],      # Early stopping patience
        'gru_sequence_length': [20, 30, 45]  # GRU sequence length
    }
    
    # Configure walk-forward parameters
    total_data_points = len(features)
    initial_train_size = int(0.6 * total_data_points)  # 60% initial training
    step_size = int(0.1 * total_data_points)          # 10% step size
    n_splits_outer = 3                                # Outer splits
    n_splits_inner = 3                                # Inner splits
    
    # Calculate total number of combinations for progress tracking
    total_combinations = len(hyperparameter_grid['gru_hidden_dim']) * \
                        len(hyperparameter_grid['learning_rate']) * \
                        len(hyperparameter_grid['batch_size']) * \
                        len(hyperparameter_grid['early_stopping_patience']) * \
                        len(hyperparameter_grid['gru_sequence_length'])
    
    print(f"\nStarting walk-forward analysis with {n_splits_outer} outer folds and {n_splits_inner} inner folds")
    print(f"Total hyperparameter combinations to test: {total_combinations}")
    print(f"Total training iterations: {n_splits_outer * n_splits_inner * total_combinations}")
    
    # Perform nested walk-forward analysis
    print("\nPerforming nested walk-forward analysis...")
    try:
        walk_forward_results = perform_walk_forward_analysis(
            model_class=GRUMomentumStrategy,
            model_config_base=config,
            features=features,
            targets=target,
            initial_train_size=initial_train_size,
            step_size=step_size,
            n_splits_outer=n_splits_outer,
            hyperparameter_grid=hyperparameter_grid,
            n_splits_inner=n_splits_inner
        )
        print("\nWalk-forward analysis completed successfully!")
    except Exception as e:
        print(f"\nError during walk-forward analysis: {str(e)}")
        raise
    
    # Print walk-forward results
    print("\nWalk-Forward Analysis Results:")
    for metric, values in walk_forward_results.items():
        if metric != 'best_params':
            print(f"\n{metric}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")
            print(f"  Values: {[f'{v:.4f}' for v in values]}")
    
    print("\nBest Parameters for Each Fold:")
    for i, params in enumerate(walk_forward_results['best_params']):
        print(f"\nFold {i+1}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    # Create visualization directory
    viz_dir = project_root / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate and save visualizations with progress bar
    print("\nGenerating visualizations...")
    with tqdm(total=3, desc="Generating visualizations") as pbar:
        # 1. Main walk-forward results
        print("Generating walk-forward results plot...")
        visualize_walk_forward_results(
            walk_forward_results,
            save_path=str(viz_dir / "gru_walk_forward_results.png")
        )
        pbar.update(1)
        
        # 2. Parameter importance
        print("Generating parameter importance plot...")
        visualize_parameter_importance(
            walk_forward_results,
            save_path=str(viz_dir / "gru_parameter_importance.png")
        )
        pbar.update(1)
        
        # 3. Performance decomposition
        print("Generating performance decomposition plot...")
        visualize_performance_decomposition(
            walk_forward_results,
            save_path=str(viz_dir / "gru_performance_decomposition.png")
        )
        pbar.update(1)
    
    # Save the scaler for future use
    scaler_path = project_root / "models" / "gru_momentum_feature_scaler.pkl"
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved to: {scaler_path}")
    
    # Save features and target
    print("\nSaving features and target...")
    features.to_parquet(config.features_path)
    target.to_parquet(config.target_path)
    print(f"Features saved to: {config.features_path}")
    print(f"Target saved to: {config.target_path}")
    
    # Print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main() 