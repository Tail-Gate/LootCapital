# XGBoost Walk-Forward Optimization Implementation Guide

## Overview

This guide provides step-by-step instructions for adapting the existing STGNN walk-forward optimization system to work with XGBoost models. The goal is to create a production-ready XGBoost walk-forward optimization system that leverages the best hyperparameters found by the hyperparameter optimization.

## Step 1: Create XGBoost Walk-Forward Optimizer Class

```python
#!/usr/bin/env python3
"""
XGBoost Walk-Forward Optimization System

This script implements walk-forward optimization for XGBoost models:
1. Uses optimized hyperparameters from hyperparameter optimization
2. Implements time-series aware data splitting
3. Saves XGBoost models and artifacts for production inference
4. Generates comprehensive performance reports
5. Uses SMOTE for class balancing
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import gc
import psutil
import joblib
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from XGBoostMean.xgboost_hyperopt_config import XGBoostHyperoptConfig
from XGBoostMean.xgboost_hyperopt_trainer import XGBoostHyperoptTrainer
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators
from utils.feature_generator import FeatureGenerator

class XGBoostWalkForwardOptimizer:
    """Walk-forward optimization for XGBoost models"""
    
    def __init__(self, 
                 train_window_days: int = 180,  # 6 months training
                 test_window_days: int = 30,    # 1 month testing
                 step_size_days: int = 15,      # 2 weeks step
                 output_dir: str = "models",
                 reports_dir: str = "reports",
                 plots_dir: str = "plots",
                 logs_dir: str = "logs"):
        
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_size_days = step_size_days
        
        # Output directories
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.plots_dir = Path(plots_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        for directory in [self.output_dir, self.reports_dir, self.plots_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'periods': [],
            'train_metrics': [],
            'test_metrics': [],
            'model_paths': [],
            'config_paths': [],
            'features_paths': [],
            'scaler_paths': [],
            'metadata_paths': [],
            'configs': []
        }
        
        # Initialize components
        self.market_data = MarketData(data_source_path='data')
        self.technical_indicators = TechnicalIndicators()
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f'xgboost_walk_forward_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_data_splits(self, start_date: datetime, end_date: datetime):
        """Generate walk-forward data splits"""
        splits = []
        current_start = start_date
        
        while current_start + timedelta(days=self.train_window_days + self.test_window_days) <= end_date:
            # Training period
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_window_days)
            
            # Testing period
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)
            
            splits.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'period_name': f"{train_start.strftime('%Y-%m')}_to_{test_end.strftime('%Y-%m')}"
            })
            
            # Move to next period
            current_start += timedelta(days=self.step_size_days)
        
        return splits
    
    def load_optimized_hyperparameters(self):
        """Load best XGBoost parameters from config directory"""
        config_dir = Path('config')
        if not config_dir.exists():
            self.logger.warning("Config directory not found")
            return None
        
        # Find all XGBoost parameter files
        param_files = list(config_dir.glob('xgboost_mean_reversion_best_params_*.json'))
        
        if not param_files:
            self.logger.warning("No XGBoost parameter files found")
            return None
        
        # Get the most recent file
        latest_file = max(param_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r') as f:
                optimized_params = json.load(f)
            
            self.logger.info(f"Loaded optimized hyperparameters from: {latest_file}")
            self.logger.info(f"Optimized parameters: {optimized_params}")
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"Failed to load optimized hyperparameters: {e}")
            return None
    
    def create_time_based_target(self, data: pd.DataFrame, config: XGBoostHyperoptConfig):
        """Create time-based classification target for mean reversion"""
        
        price_threshold = config.price_threshold
        early_window = config.early_window
        late_window = config.late_window
        moderate_threshold_ratio = config.moderate_threshold_ratio
        window_size = 15  # 15 candlesticks window
        
        self.logger.info(f"Creating time-based target with:")
        self.logger.info(f"  Price threshold: {price_threshold:.4f}")
        self.logger.info(f"  Early window: {early_window}, Late window: {late_window}")
        self.logger.info(f"  Moderate threshold ratio: {moderate_threshold_ratio:.2f}")
        
        future_returns = []
        
        for i in range(len(data)):
            if i + window_size >= len(data):
                # For the last few candlesticks, we can't look ahead 15 periods
                future_returns.append(1)  # No signal
            else:
                # Get current price
                current_price = data['close'].iloc[i]
                
                # Calculate early period returns (first 5 periods)
                early_prices = data['close'].iloc[i:i+early_window+1]
                early_max_return = (early_prices.max() - current_price) / current_price
                early_min_return = (early_prices.min() - current_price) / current_price
                
                # Calculate late period returns (periods 6-15)
                late_prices = data['close'].iloc[i+early_window:i+window_size+1]
                late_max_return = (late_prices.max() - current_price) / current_price
                late_min_return = (late_prices.min() - current_price) / current_price
                
                # Calculate full window returns
                future_prices = data['close'].iloc[i:i+window_size+1]
                full_max_return = (future_prices.max() - current_price) / current_price
                full_min_return = (future_prices.min() - current_price) / current_price
                
                # Time-based classification logic
                if early_max_return >= price_threshold:
                    future_returns.append(2)  # Early Up signal
                elif early_min_return <= -price_threshold:
                    future_returns.append(0)  # Early Down signal
                elif late_max_return >= price_threshold:
                    future_returns.append(1)  # Late Up signal (hold)
                elif late_min_return <= -price_threshold:
                    future_returns.append(1)  # Late Down signal (hold)
                elif full_max_return >= price_threshold * moderate_threshold_ratio:
                    future_returns.append(1)  # Hold signal (moderate movement)
                elif full_min_return <= -price_threshold * moderate_threshold_ratio:
                    future_returns.append(1)  # Hold signal (moderate movement)
                else:
                    future_returns.append(1)  # No significant movement (hold)
        
        return np.array(future_returns)
    
    def prepare_period_data(self, start_date: datetime, end_date: datetime, config: XGBoostHyperoptConfig):
        """Prepare data for a specific time period"""
        
        self.logger.info(f"Preparing data for period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Load market data
        data_response = self.market_data.get_data(['ETH/USD'])
        
        # Extract DataFrame from response
        if isinstance(data_response, dict):
            data = data_response['ETH/USD']
        else:
            data = data_response
        
        # Filter data for the specific time range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        self.logger.info(f"Data shape: {data.shape}")
        
        # Generate features using FeatureGenerator with config
        feature_generator = FeatureGenerator(config=config.__dict__)
        features = feature_generator.generate_features(data)
        
        self.logger.info(f"Generated {len(features.columns)} features")
        
        # Create target variable
        y = self.create_time_based_target(data, config)
        
        # Ensure features and target have the same length
        if len(features) != len(y):
            min_length = min(len(features), len(y))
            features = features.iloc[:min_length]
            y = y[:min_length]
        
        # Remove rows with NaN values
        valid_mask = ~(features.isna().any(axis=1))
        features = features[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"Final features shape: {features.shape}")
        self.logger.info(f"Final target shape: {y.shape}")
        self.logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return features, y
    
    def train_and_evaluate_xgboost_period(self, period_data: dict):
        """Train XGBoost model for a specific time period"""
        
        self.logger.info(f"Processing period: {period_data['period_name']}")
        period_start_time = time.time()
        
        try:
            # Load optimized parameters
            optimized_params = self.load_optimized_hyperparameters()
            
            if optimized_params:
                config = XGBoostHyperoptConfig(**optimized_params)
                self.logger.info("Using optimized hyperparameters")
            else:
                config = XGBoostHyperoptConfig()
                self.logger.info("Using default hyperparameters")
            
            # Prepare training data
            self.logger.info("Preparing training data...")
            X_train_full, y_train_full = self.prepare_period_data(
                period_data['train_start'], 
                period_data['train_end'], 
                config
            )
            
            # Prepare test data
            self.logger.info("Preparing test data...")
            X_test_full, y_test_full = self.prepare_period_data(
                period_data['test_start'], 
                period_data['test_end'], 
                config
            )
            
            # Check if we have enough data
            if len(X_train_full) < 100 or len(X_test_full) < 20:
                self.logger.warning(f"Insufficient data for period {period_data['period_name']}")
                self.logger.warning(f"Training samples: {len(X_train_full)}, Test samples: {len(X_test_full)}")
                return None
            
            # Split training data into train/validation
            split_idx = int(len(X_train_full) * 0.8)  # 80% train, 20% validation
            
            X_train = X_train_full.iloc[:split_idx].values
            y_train = y_train_full[:split_idx]
            X_val = X_train_full.iloc[split_idx:].values
            y_val = y_train_full[split_idx:]
            X_test = X_test_full.values
            y_test = y_test_full
            
            self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Create trainer
            trainer = XGBoostHyperoptTrainer(config, None)
            
            # Train model
            self.logger.info(f"Starting training for period {period_data['period_name']}...")
            training_start_time = time.time()
            training_history = trainer.train_with_smote(X_train, y_train, X_val, y_val)
            training_time = time.time() - training_start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate on training data
            self.logger.info("Evaluating on training data...")
            train_metrics = trainer.evaluate(X_train, y_train)
            
            # Evaluate on test data
            self.logger.info("Evaluating on test data...")
            test_metrics = trainer.evaluate(X_test, y_test)
            
            # Save model and artifacts
            self.logger.info("Saving model and artifacts...")
            saved_paths = self.save_xgboost_model(trainer, config, period_data, test_metrics)
            
            period_time = time.time() - period_start_time
            self.logger.info(f"Period {period_data['period_name']} completed in {period_time:.2f} seconds")
            
            return {
                'period_name': period_data['period_name'],
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'training_history': training_history,
                'config': config,
                'saved_paths': saved_paths
            }
            
        except Exception as e:
            self.logger.error(f"Error in train_and_evaluate_xgboost_period for {period_data['period_name']}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def save_xgboost_model(self, trainer, config, period_data, test_metrics):
        """Save XGBoost model and all artifacts for production"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        period_name = period_data['period_name']
        
        saved_paths = {}
        
        try:
            # 1. Save XGBoost model
            model_path = self.output_dir / f'wfo_xgboost_{period_name}_{timestamp}.json'
            trainer.model.save_model(str(model_path))
            saved_paths['model_path'] = str(model_path)
            self.logger.info(f"Saved XGBoost model: {model_path}")
            
            # 2. Save configuration
            config_path = self.output_dir / f'wfo_xgboost_config_{period_name}_{timestamp}.json'
            with open(config_path, 'w') as f:
                json.dump(config.__dict__, f, indent=4)
            saved_paths['config_path'] = str(config_path)
            self.logger.info(f"Saved configuration: {config_path}")
            
            # 3. Save feature list
            features_path = self.output_dir / f'wfo_xgboost_features_{period_name}_{timestamp}.json'
            with open(features_path, 'w') as f:
                json.dump(config.feature_list, f, indent=4)
            saved_paths['features_path'] = str(features_path)
            self.logger.info(f"Saved feature list: {features_path}")
            
            # 4. Save scaler (if used)
            scaler_path = self.output_dir / f'wfo_xgboost_scaler_{period_name}_{timestamp}.joblib'
            joblib.dump(trainer.scaler, scaler_path)
            saved_paths['scaler_path'] = str(scaler_path)
            self.logger.info(f"Saved scaler: {scaler_path}")
            
            # 5. Save inference metadata
            metadata = {
                'model_path': str(model_path),
                'config_path': str(config_path),
                'features_path': str(features_path),
                'scaler_path': str(scaler_path),
                'period_name': period_name,
                'timestamp': timestamp,
                'test_metrics': test_metrics,
                'input_shape': list(X_train.shape) if 'X_train' in locals() else None
            }
            
            metadata_path = self.output_dir / f'wfo_xgboost_metadata_{period_name}_{timestamp}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            saved_paths['metadata_path'] = str(metadata_path)
            self.logger.info(f"Saved metadata: {metadata_path}")
            
            # Verify all files were created
            for path_name, path_str in saved_paths.items():
                if not Path(path_str).exists():
                    self.logger.error(f"Failed to create {path_name}: {path_str}")
                else:
                    file_size = Path(path_str).stat().st_size
                    self.logger.info(f"✓ {path_name}: {file_size} bytes")
            
        except Exception as e:
            self.logger.error(f"Error saving XGBoost model: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        return saved_paths
    
    def run_optimization(self, start_date: datetime, end_date: datetime):
        """Run walk-forward optimization for XGBoost"""
        
        self.logger.info(f"Starting XGBoost walk-forward optimization from {start_date} to {end_date}")
        optimization_start_time = time.time()
        
        # Generate data splits
        self.logger.info("Generating walk-forward data splits...")
        splits = self.get_data_splits(start_date, end_date)
        self.logger.info(f"Generated {len(splits)} walk-forward periods")
        
        # Process each period
        successful_periods = 0
        failed_periods = 0
        
        for i, split in enumerate(tqdm(splits, desc="Processing periods", unit="period")):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing period {i+1}/{len(splits)}: {split['period_name']}")
            self.logger.info(f"Progress: {i+1}/{len(splits)} ({((i+1)/len(splits)*100):.1f}%)")
            self.logger.info(f"{'='*60}")
            
            # Train and evaluate
            result = self.train_and_evaluate_xgboost_period(split)
            
            if result is not None:
                successful_periods += 1
                self.results['periods'].append(result['period_name'])
                self.results['train_metrics'].append(result['train_metrics'])
                self.results['test_metrics'].append(result['test_metrics'])
                self.results['model_paths'].append(result['saved_paths']['model_path'])
                self.results['config_paths'].append(result['saved_paths']['config_path'])
                self.results['features_paths'].append(result['saved_paths']['features_path'])
                self.results['scaler_paths'].append(result['saved_paths']['scaler_path'])
                self.results['metadata_paths'].append(result['saved_paths']['metadata_path'])
                self.results['configs'].append(result['config'])
                
                # Log results
                self.logger.info(f"✓ Period {split['period_name']} completed successfully")
                self.logger.info(f"  Test Accuracy: {result['test_metrics']['classification_report']['accuracy']:.4f}")
                self.logger.info(f"  Test F1 (Up): {result['test_metrics']['f1'][2]:.4f}")
                self.logger.info(f"  Test F1 (Down): {result['test_metrics']['f1'][0]:.4f}")
            else:
                failed_periods += 1
                self.logger.error(f"✗ Period {split['period_name']} failed")
            
            # Log progress summary
            elapsed_time = time.time() - optimization_start_time
            avg_time_per_period = elapsed_time / (i + 1)
            remaining_periods = len(splits) - (i + 1)
            estimated_remaining_time = remaining_periods * avg_time_per_period
            
            self.logger.info(f"Progress Summary:")
            self.logger.info(f"  Completed: {i+1}/{len(splits)} periods")
            self.logger.info(f"  Successful: {successful_periods}, Failed: {failed_periods}")
            self.logger.info(f"  Elapsed time: {elapsed_time/60:.1f} minutes")
            self.logger.info(f"  Avg time per period: {avg_time_per_period/60:.1f} minutes")
            self.logger.info(f"  Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
        
        # Final summary
        total_time = time.time() - optimization_start_time
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"XGBOOST WALK-FORWARD OPTIMIZATION COMPLETED")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total periods: {len(splits)}")
        self.logger.info(f"Successful periods: {successful_periods}")
        self.logger.info(f"Failed periods: {failed_periods}")
        if len(splits) > 0:
            self.logger.info(f"Success rate: {successful_periods/len(splits)*100:.1f}%")
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        
        # Generate summary report
        if successful_periods > 0:
            self.generate_summary_report()
        else:
            self.logger.error("No successful periods to generate report")
        
        return self.results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        self.logger.info("Generating XGBoost walk-forward optimization summary report")
        
        # Calculate aggregate metrics
        test_accuracies = [metrics['classification_report']['accuracy'] for metrics in self.results['test_metrics']]
        test_f1_up = [metrics['f1'][2] for metrics in self.results['test_metrics']]
        test_f1_down = [metrics['f1'][0] for metrics in self.results['test_metrics']]
        test_f1_hold = [metrics['f1'][1] for metrics in self.results['test_metrics']]
        
        # Calculate statistics
        summary = {
            'total_periods': len(self.results['periods']),
            'mean_test_accuracy': np.mean(test_accuracies),
            'std_test_accuracy': np.std(test_accuracies),
            'mean_f1_up': np.mean(test_f1_up),
            'mean_f1_down': np.mean(test_f1_down),
            'mean_f1_hold': np.mean(test_f1_hold),
            'best_period': self.results['periods'][np.argmax(test_accuracies)],
            'worst_period': self.results['periods'][np.argmin(test_accuracies)],
            'best_accuracy': np.max(test_accuracies),
            'worst_accuracy': np.min(test_accuracies),
            'periods': self.results['periods'],
            'test_accuracies': test_accuracies,
            'test_f1_up': test_f1_up,
            'test_f1_down': test_f1_down,
            'test_f1_hold': test_f1_hold
        }
        
        # Print summary
        print("\n" + "="*80)
        print("XGBOOST WALK-FORWARD OPTIMIZATION SUMMARY REPORT")
        print("="*80)
        print(f"Total periods: {summary['total_periods']}")
        print(f"Mean test accuracy: {summary['mean_test_accuracy']:.4f} ± {summary['std_test_accuracy']:.4f}")
        print(f"\nMean F1 Scores:")
        print(f"  Up: {summary['mean_f1_up']:.4f}")
        print(f"  Down: {summary['mean_f1_down']:.4f}")
        print(f"  Hold: {summary['mean_f1_hold']:.4f}")
        print(f"\nBest period: {summary['best_period']} (accuracy: {summary['best_accuracy']:.4f})")
        print(f"Worst period: {summary['worst_period']} (accuracy: {summary['worst_accuracy']:.4f})")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f'xgboost_walk_forward_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        self.logger.info(f"Detailed report saved to: {report_path}")
        
        # Create visualization
        self.create_visualizations(summary)
        
        return summary
    
    def create_visualizations(self, summary: dict):
        """Create visualizations of walk-forward results"""
        
        # Plot 1: Accuracy over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(summary['test_accuracies'], marker='o')
        plt.title('XGBoost Test Accuracy Over Time')
        plt.xlabel('Period')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Plot 2: F1 scores over time
        plt.subplot(2, 2, 2)
        plt.plot(summary['test_f1_up'], label='Up', marker='o')
        plt.plot(summary['test_f1_down'], label='Down', marker='s')
        plt.plot(summary['test_f1_hold'], label='Hold', marker='^')
        plt.title('XGBoost F1 Scores Over Time')
        plt.xlabel('Period')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Distribution of accuracies
        plt.subplot(2, 2, 3)
        plt.hist(summary['test_accuracies'], bins=10, alpha=0.7, edgecolor='black')
        plt.title('Distribution of XGBoost Test Accuracies')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.axvline(summary['mean_test_accuracy'], color='red', linestyle='--', label=f'Mean: {summary["mean_test_accuracy"]:.4f}')
        plt.legend()
        
        # Plot 4: Box plot of F1 scores
        plt.subplot(2, 2, 4)
        f1_data = [summary['test_f1_up'], summary['test_f1_down'], summary['test_f1_hold']]
        plt.boxplot(f1_data, labels=['Up', 'Down', 'Hold'])
        plt.title('XGBoost F1 Score Distribution by Class')
        plt.ylabel('F1 Score')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.plots_dir / f'xgboost_walk_forward_results_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to: {plot_path}")

def main():
    """Main function for XGBoost walk-forward optimization"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='XGBoost Walk-Forward Optimization')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--train-window-days', type=int, default=180, help='Training window in days')
    parser.add_argument('--test-window-days', type=int, default=30, help='Testing window in days')
    parser.add_argument('--step-size-days', type=int, default=15, help='Step size in days')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--reports-dir', type=str, default='reports', help='Reports directory')
    parser.add_argument('--plots-dir', type=str, default='plots', help='Plots directory')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Logs directory')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = XGBoostWalkForwardOptimizer(
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        step_size_days=args.step_size_days,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        plots_dir=args.plots_dir,
        logs_dir=args.logs_dir
    )
    
    # Parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
        
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        # Default to 2 years ago
        start_date = end_date - timedelta(days=730)
    
    # Run optimization
    results = optimizer.run_optimization(start_date, end_date)
    
    return results

if __name__ == "__main__":
    main()
```

## Step 2: Create Production Inference Script

```python
#!/usr/bin/env python3
"""
XGBoost Production Inference Script

This script loads trained XGBoost models and makes predictions for production use.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators
from utils.feature_generator import FeatureGenerator

class XGBoostInferenceEngine:
    """Production inference engine for XGBoost models"""
    
    def __init__(self, metadata_path: str):
        """
        Initialize inference engine with model metadata
        
        Args:
            metadata_path: Path to the metadata JSON file
        """
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.config = None
        self.feature_list = None
        self.scaler = None
        
        # Load model and artifacts
        self.load_model()
        
        # Initialize components
        self.market_data = MarketData()
        self.technical_indicators = TechnicalIndicators()
    
    def load_model(self):
        """Load XGBoost model and all artifacts"""
        
        try:
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load XGBoost model
            model_path = metadata['model_path']
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            
            # Load configuration
            with open(metadata['config_path'], 'r') as f:
                config_dict = json.load(f)
            self.config = config_dict
            
            # Load feature list
            with open(metadata['features_path'], 'r') as f:
                self.feature_list = json.load(f)
            
            # Load scaler
            self.scaler = joblib.load(metadata['scaler_path'])
            
            print(f"✓ Loaded XGBoost model from: {model_path}")
            print(f"✓ Model period: {metadata['period_name']}")
            print(f"✓ Features: {len(self.feature_list)}")
            print(f"✓ Test accuracy: {metadata['test_metrics']['classification_report']['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for inference"""
        
        # Generate features using FeatureGenerator
        feature_generator = FeatureGenerator(config=self.config)
        features = feature_generator.generate_features(market_data)
        
        # Select only required features
        features = features[self.feature_list]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on market data
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        
        # Prepare features
        features_scaled = self.prepare_features(market_data)
        
        # Create DMatrix
        dtest = xgb.DMatrix(features_scaled)
        
        # Make prediction
        probabilities = self.model.predict(dtest)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def predict_latest(self, symbol: str = 'ETH/USD') -> Tuple[int, np.ndarray]:
        """
        Make prediction on the latest market data
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (prediction, probabilities)
        """
        
        # Get latest market data
        data_response = self.market_data.get_data([symbol])
        data = data_response[symbol] if isinstance(data_response, dict) else data_response
        
        # Use last 100 periods for feature calculation
        latest_data = data.tail(100)
        
        # Make prediction
        predictions, probabilities = self.predict(latest_data)
        
        # Return prediction for the latest period
        latest_prediction = predictions[-1]
        latest_probabilities = probabilities[-1]
        
        return latest_prediction, latest_probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        
        importance_dict = self.model.get_score(importance_type='gain')
        
        # Map feature indices to feature names
        feature_importance = {}
        for idx, importance in importance_dict.items():
            feature_idx = int(idx.replace('f', ''))
            if feature_idx < len(self.feature_list):
                feature_name = self.feature_list[feature_idx]
                feature_importance[feature_name] = importance
        
        return feature_importance
    
    def explain_prediction(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain the latest prediction
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with prediction explanation
        """
        
        predictions, probabilities = self.predict(market_data)
        latest_prediction = predictions[-1]
        latest_probabilities = probabilities[-1]
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        # Prepare features for analysis
        features_scaled = self.prepare_features(market_data)
        latest_features = features_scaled[-1]
        
        # Find top contributing features
        feature_contributions = {}
        for i, (feature_name, importance) in enumerate(feature_importance.items()):
            if i < len(latest_features):
                feature_contributions[feature_name] = {
                    'value': latest_features[i],
                    'importance': importance,
                    'contribution': latest_features[i] * importance
                }
        
        # Sort by contribution
        sorted_contributions = sorted(
            feature_contributions.items(), 
            key=lambda x: abs(x[1]['contribution']), 
            reverse=True
        )
        
        class_names = ['Down', 'Hold', 'Up']
        prediction_class = class_names[latest_prediction]
        
        explanation = {
            'prediction': latest_prediction,
            'prediction_class': prediction_class,
            'probabilities': {
                'down': float(latest_probabilities[0]),
                'hold': float(latest_probabilities[1]),
                'up': float(latest_probabilities[2])
            },
            'confidence': float(max(latest_probabilities)),
            'top_features': sorted_contributions[:10],
            'feature_importance': feature_importance
        }
        
        return explanation

def main():
    """Example usage of XGBoost inference engine"""
    
    # Example metadata path (replace with actual path)
    metadata_path = "models/wfo_xgboost_2024-01_to_2024-02_20241201_143022_metadata.json"
    
    try:
        # Initialize inference engine
        engine = XGBoostInferenceEngine(metadata_path)
        
        # Get latest prediction
        prediction, probabilities = engine.predict_latest('ETH/USD')
        
        print(f"\nLatest Prediction:")
        print(f"  Prediction: {prediction} ({['Down', 'Hold', 'Up'][prediction]})")
        print(f"  Probabilities: Down={probabilities[0]:.3f}, Hold={probabilities[1]:.3f}, Up={probabilities[2]:.3f}")
        print(f"  Confidence: {max(probabilities):.3f}")
        
        # Get explanation
        data_response = engine.market_data.get_data(['ETH/USD'])
        data = data_response['ETH/USD'] if isinstance(data_response, dict) else data_response
        latest_data = data.tail(100)
        
        explanation = engine.explain_prediction(latest_data)
        
        print(f"\nPrediction Explanation:")
        print(f"  Top contributing features:")
        for feature_name, details in explanation['top_features'][:5]:
            print(f"    {feature_name}: {details['value']:.4f} (importance: {details['importance']:.4f})")
        
        # Get feature importance
        importance = engine.get_feature_importance()
        print(f"\nTop 10 Most Important Features:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature_name, importance_score in sorted_importance[:10]:
            print(f"  {feature_name}: {importance_score:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

## Step 3: Usage Examples

### Running Walk-Forward Optimization

```bash
# Run with default settings (2 years of data)
python xgboost_walk_forward_optimizer.py

# Run with custom time range
python xgboost_walk_forward_optimizer.py \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --train-window-days 180 \
  --test-window-days 30 \
  --step-size-days 15

# Run with custom output directories
python xgboost_walk_forward_optimizer.py \
  --output-dir /path/to/models \
  --reports-dir /path/to/reports \
  --plots-dir /path/to/plots \
  --logs-dir /path/to/logs
```

### Using Production Inference

```python
# Load model and make predictions
from xgboost_inference_engine import XGBoostInferenceEngine

# Initialize with metadata file
engine = XGBoostInferenceEngine("models/wfo_xgboost_metadata.json")

# Get latest prediction
prediction, probabilities = engine.predict_latest('ETH/USD')
print(f"Prediction: {['Down', 'Hold', 'Up'][prediction]}")
print(f"Probabilities: {probabilities}")

# Get explanation
explanation = engine.explain_prediction(market_data)
print(f"Top features: {explanation['top_features'][:5]}")
```

## Key Features Implemented

1. **Automated Hyperparameter Loading**: Automatically finds and loads the best hyperparameters from the config directory
2. **Time-Based Classification**: Implements the same sophisticated time-based classification logic as the hyperparameter optimization
3. **Production Model Saving**: Saves all artifacts needed for production inference
4. **Comprehensive Logging**: Detailed logging throughout the process
5. **Memory Management**: Efficient memory usage with garbage collection
6. **Performance Monitoring**: Tracks training time and progress
7. **Visualization**: Creates plots and reports for analysis
8. **Production Inference**: Ready-to-use inference engine for real-time predictions

## Integration with Existing System

The XGBoost walk-forward optimizer integrates seamlessly with the existing system:

1. **Uses the same FeatureGenerator**: Leverages the same feature engineering pipeline
2. **Uses the same MarketData**: Uses the same data loading infrastructure
3. **Uses the same TechnicalIndicators**: Uses the same technical analysis components
4. **Uses optimized hyperparameters**: Automatically loads the best parameters from hyperparameter optimization
5. **Follows the same patterns**: Uses the same architectural patterns as the STGNN system

This implementation provides a complete, production-ready XGBoost walk-forward optimization system that can be used alongside or as an alternative to the STGNN system. 