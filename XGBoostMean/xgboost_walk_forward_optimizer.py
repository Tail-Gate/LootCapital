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
            self.logger.info(f"Aligned features and target to length: {min_length}")
        
        # Remove rows with NaN values
        valid_mask = ~(features.isna().any(axis=1))
        features = features[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"Final features shape: {features.shape}")
        self.logger.info(f"Final target shape: {y.shape}")
        
        # Report class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        self.logger.info(f"Class distribution: {class_distribution}")
        
        return features, y, data
    
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
            
            # Also get the raw data for trading simulation
            data_test_response = self.market_data.get_data(['ETH/USD'])
            if isinstance(data_test_response, dict):
                data_test_full = data_test_response['ETH/USD']
            else:
                data_test_full = data_test_response
            data_test_full = data_test_full[(data_test_full.index >= period_data['test_start']) & 
                                         (data_test_full.index <= period_data['test_end'])]
            
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
            
            # SHARPE RATIO TRADING SIMULATION (same as hyperopt runner)
            self.logger.info("Starting Sharpe ratio calculation for live trading simulation...")
            
            # Get predictions and probabilities
            dtest = xgb.DMatrix(X_test)
            y_pred_proba = trainer.model.predict(dtest, output_margin=False)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate trading metrics through simulation
            sharpe_ratio, win_rate, profit_factor, max_drawdown = self.calculate_sharpe_ratio_trading_simulation(
                X_test, y_test, y_pred, y_pred_proba, data_test_full
            )
            
            self.logger.info(f"Trading Simulation Results:")
            self.logger.info(f"  Sharpe Ratio (daily): {sharpe_ratio:.4f}")
            self.logger.info(f"  Win Rate: {win_rate:.4f}")
            self.logger.info(f"  Profit Factor: {profit_factor:.4f}")
            self.logger.info(f"  Max Drawdown: {max_drawdown:.4f}")
            
            # Add trading metrics to test_metrics
            test_metrics['trading_metrics'] = {
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown
            }
            
            # Save the model and all artifacts
            self.logger.info("Saving XGBoost model and artifacts...")
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
            
            # 5. Save inference metadata with trading performance focus
            metadata = {
                'model_path': str(model_path),
                'config_path': str(config_path),
                'features_path': str(features_path),
                'scaler_path': str(scaler_path),
                'period_name': period_name,
                'timestamp': timestamp,
                'test_metrics': test_metrics,
                # PRIORITY: Trading Performance Metrics (same as hyperopt runner)
                'trading_metrics': test_metrics.get('trading_metrics', {}),
                # Secondary: Trading Signal Metrics
                'trading_signal_metrics': {
                    'up_f1': test_metrics['f1'][2],
                    'down_f1': test_metrics['f1'][0],
                    'hold_f1': test_metrics['f1'][1],
                    'combined_trading_f1': (test_metrics['f1'][2] + test_metrics['f1'][0]) / 2,
                    'up_precision': test_metrics['precision'][2],
                    'down_precision': test_metrics['precision'][0],
                    'up_recall': test_metrics['recall'][2],
                    'down_recall': test_metrics['recall'][0]
                },
                'overall_accuracy': test_metrics['classification_report']['accuracy']
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
                self.results['configs'].append(result['config'])
                
                # Log results - PRIORITIZE TRADING PERFORMANCE METRICS
                self.logger.info(f"✓ Period {split['period_name']} completed successfully")
                self.logger.info(f"  🎯 TRADING PERFORMANCE METRICS:")
                self.logger.info(f"    Sharpe Ratio: {result['test_metrics']['trading_metrics']['sharpe_ratio']:.4f}")
                self.logger.info(f"    Win Rate: {result['test_metrics']['trading_metrics']['win_rate']:.4f}")
                self.logger.info(f"    Profit Factor: {result['test_metrics']['trading_metrics']['profit_factor']:.4f}")
                self.logger.info(f"    Max Drawdown: {result['test_metrics']['trading_metrics']['max_drawdown']:.4f}")
                self.logger.info(f"  📊 Traditional ML Metrics:")
                self.logger.info(f"    UP (Buy Signal): {result['test_metrics']['f1'][2]:.4f}")
                self.logger.info(f"    DOWN (Sell Signal): {result['test_metrics']['f1'][0]:.4f}")
                self.logger.info(f"    Hold (No Trade): {result['test_metrics']['f1'][1]:.4f}")
                self.logger.info(f"    Overall Accuracy: {result['test_metrics']['classification_report']['accuracy']:.4f}")
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
        
        return self.results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        self.logger.info("Generating XGBoost walk-forward optimization summary report")
        
        # Calculate aggregate metrics - PRIORITY: Trading Performance Metrics
        test_accuracies = [metrics['classification_report']['accuracy'] for metrics in self.results['test_metrics']]
        test_f1_up = [metrics['f1'][2] for metrics in self.results['test_metrics']]
        test_f1_down = [metrics['f1'][0] for metrics in self.results['test_metrics']]
        test_f1_hold = [metrics['f1'][1] for metrics in self.results['test_metrics']]
        
        # Trading performance metrics (same as hyperopt runner)
        test_sharpe_ratios = [metrics.get('trading_metrics', {}).get('sharpe_ratio', 0.0) for metrics in self.results['test_metrics']]
        test_win_rates = [metrics.get('trading_metrics', {}).get('win_rate', 0.0) for metrics in self.results['test_metrics']]
        test_profit_factors = [metrics.get('trading_metrics', {}).get('profit_factor', 0.0) for metrics in self.results['test_metrics']]
        test_max_drawdowns = [metrics.get('trading_metrics', {}).get('max_drawdown', 0.0) for metrics in self.results['test_metrics']]
        
        # Calculate statistics
        summary = {
            'total_periods': len(self.results['periods']),
            # PRIORITY: Trading Performance Metrics
            'mean_sharpe_ratio': np.mean(test_sharpe_ratios),
            'std_sharpe_ratio': np.std(test_sharpe_ratios),
            'mean_win_rate': np.mean(test_win_rates),
            'std_win_rate': np.std(test_win_rates),
            'mean_profit_factor': np.mean(test_profit_factors),
            'std_profit_factor': np.std(test_profit_factors),
            'mean_max_drawdown': np.mean(test_max_drawdowns),
            'std_max_drawdown': np.std(test_max_drawdowns),
            # Secondary: Traditional ML Metrics
            'mean_test_accuracy': np.mean(test_accuracies),
            'std_test_accuracy': np.std(test_accuracies),
            'mean_f1_up': np.mean(test_f1_up),
            'mean_f1_down': np.mean(test_f1_down),
            'mean_f1_hold': np.mean(test_f1_hold),
            # Best/Worst periods
            'best_sharpe_period': self.results['periods'][np.argmax(test_sharpe_ratios)] if test_sharpe_ratios else None,
            'best_win_rate_period': self.results['periods'][np.argmax(test_win_rates)] if test_win_rates else None,
            'best_profit_factor_period': self.results['periods'][np.argmax(test_profit_factors)] if test_profit_factors else None,
            'best_accuracy_period': self.results['periods'][np.argmax(test_accuracies)],
            'worst_accuracy_period': self.results['periods'][np.argmin(test_accuracies)],
            # Best/Worst values
            'best_sharpe_ratio': np.max(test_sharpe_ratios) if test_sharpe_ratios else 0.0,
            'best_win_rate': np.max(test_win_rates) if test_win_rates else 0.0,
            'best_profit_factor': np.max(test_profit_factors) if test_profit_factors else 0.0,
            'worst_max_drawdown': np.min(test_max_drawdowns) if test_max_drawdowns else 0.0,
            'best_accuracy': np.max(test_accuracies),
            'worst_accuracy': np.min(test_accuracies),
            # All data
            'periods': self.results['periods'],
            'test_accuracies': test_accuracies,
            'test_f1_up': test_f1_up,
            'test_f1_down': test_f1_down,
            'test_f1_hold': test_f1_hold,
            'test_sharpe_ratios': test_sharpe_ratios,
            'test_win_rates': test_win_rates,
            'test_profit_factors': test_profit_factors,
            'test_max_drawdowns': test_max_drawdowns
        }
        
        # Print summary - PRIORITIZE TRADING PERFORMANCE METRICS
        print("\n" + "="*80)
        print("🎯 XGBOOST WALK-FORWARD OPTIMIZATION - TRADING PERFORMANCE FOCUS")
        print("="*80)
        print(f"Total periods: {summary['total_periods']}")
        
        # PRIORITY: Trading Performance Metrics (same as hyperopt runner)
        print(f"\n🎯 TRADING PERFORMANCE METRICS (PRIORITY):")
        print(f"  📈 Sharpe Ratio: {summary['mean_sharpe_ratio']:.4f} ± {summary['std_sharpe_ratio']:.4f}")
        print(f"  🎯 Win Rate: {summary['mean_win_rate']:.4f} ± {summary['std_win_rate']:.4f}")
        print(f"  💰 Profit Factor: {summary['mean_profit_factor']:.4f} ± {summary['std_profit_factor']:.4f}")
        print(f"  📉 Max Drawdown: {summary['mean_max_drawdown']:.4f} ± {summary['std_max_drawdown']:.4f}")
        
        # Secondary: Traditional ML Metrics
        print(f"\n📊 TRADITIONAL ML METRICS (SECONDARY):")
        print(f"  📈 UP (Buy Signal): {summary['mean_f1_up']:.4f} ± {np.std(summary['test_f1_up']):.4f}")
        print(f"  📉 DOWN (Sell Signal): {summary['mean_f1_down']:.4f} ± {np.std(summary['test_f1_down']):.4f}")
        print(f"  ⏸️  Hold (No Trade): {summary['mean_f1_hold']:.4f}")
        print(f"  📋 Overall Accuracy: {summary['mean_test_accuracy']:.4f} ± {summary['std_test_accuracy']:.4f}")
        
        # Best/Worst periods for trading performance
        best_sharpe_period = summary['best_sharpe_period']
        best_win_rate_period = summary['best_win_rate_period']
        best_profit_factor_period = summary['best_profit_factor_period']
        
        print(f"\n🏆 BEST TRADING PERFORMANCE PERIODS:")
        print(f"  📈 Best Sharpe Ratio: {best_sharpe_period} (Sharpe: {summary['best_sharpe_ratio']:.4f})")
        print(f"  🎯 Best Win Rate: {best_win_rate_period} (Win Rate: {summary['best_win_rate']:.4f})")
        print(f"  💰 Best Profit Factor: {best_profit_factor_period} (Profit Factor: {summary['best_profit_factor']:.4f})")
        print(f"  📉 Worst Max Drawdown: {summary['worst_max_drawdown']:.4f}")
        
        # Print focal loss information if available
        if 'focal_loss_enabled' in summary and summary['focal_loss_enabled']:
            print(f"\n🔧 FOCAL LOSS CONFIGURATION:")
            print(f"  Alpha: {summary['focal_alpha']:.4f}")
            print(f"  Gamma: {summary['focal_gamma']:.4f}")
            print(f"  Class Multipliers: [0: {summary['class_multipliers']['class_0']:.4f}, 1: {summary['class_multipliers']['class_1']:.4f}, 2: {summary['class_multipliers']['class_2']:.4f}]")
        else:
            print(f"\n🔧 Loss Function: Standard Cross-Entropy")
        
        # Print period-by-period results - PRIORITIZE TRADING PERFORMANCE
        print("\n📊 PERIOD-BY-PERIOD TRADING PERFORMANCE RESULTS:")
        print(f"{'Period':<20} {'📈Sharpe':<8} {'🎯WinRate':<10} {'💰Profit':<10} {'📉Drawdown':<10}")
        print("-" * 70)
        for i, period in enumerate(self.results['periods']):
            # Get trading performance metrics
            sharpe = test_sharpe_ratios[i]
            win_rate = test_win_rates[i]
            profit_factor = test_profit_factors[i]
            max_drawdown = test_max_drawdowns[i]
            
            # Add indicators for good trading performance
            sharpe_indicator = "🔥" if sharpe > 0.5 else "📈" if sharpe > 0.0 else "📈"
            win_rate_indicator = "🔥" if win_rate > 0.6 else "🎯" if win_rate > 0.5 else "🎯"
            profit_indicator = "🔥" if profit_factor > 2.0 else "💰" if profit_factor > 1.0 else "💰"
            
            print(f"{period:<20} {sharpe_indicator}{sharpe:<7.4f} {win_rate_indicator}{win_rate:<9.4f} {profit_indicator}{profit_factor:<9.4f} {max_drawdown:<10.4f}")
        
        # Summary of trading performance
        print(f"\n🎯 TRADING PERFORMANCE SUMMARY:")
        print(f"  📈 Sharpe Ratio - Mean: {summary['mean_sharpe_ratio']:.4f}, Best: {summary['best_sharpe_ratio']:.4f}, Std: {summary['std_sharpe_ratio']:.4f}")
        print(f"  🎯 Win Rate - Mean: {summary['mean_win_rate']:.4f}, Best: {summary['best_win_rate']:.4f}, Std: {summary['std_win_rate']:.4f}")
        print(f"  💰 Profit Factor - Mean: {summary['mean_profit_factor']:.4f}, Best: {summary['best_profit_factor']:.4f}, Std: {summary['std_profit_factor']:.4f}")
        print(f"  📉 Max Drawdown - Mean: {summary['mean_max_drawdown']:.4f}, Worst: {summary['worst_max_drawdown']:.4f}, Std: {summary['std_max_drawdown']:.4f}")
        
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
        """Create visualizations of walk-forward results - TRADING PERFORMANCE FOCUS"""
        
        # Plot 1: TRADING PERFORMANCE METRICS (PRIORITY)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(summary['test_sharpe_ratios'], label='Sharpe Ratio', marker='o', color='green', linewidth=2)
        plt.plot(summary['test_win_rates'], label='Win Rate', marker='s', color='blue', linewidth=2)
        plt.title('🎯 TRADING PERFORMANCE METRICS (PRIORITY)', fontweight='bold')
        plt.xlabel('Period')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Profit Factor and Max Drawdown
        plt.subplot(2, 2, 2)
        plt.plot(summary['test_profit_factors'], label='Profit Factor', marker='o', color='purple', linewidth=2)
        plt.plot([abs(x) for x in summary['test_max_drawdowns']], label='Max Drawdown (abs)', marker='s', color='red', linewidth=2)
        plt.title('💰 Profit Factor & Risk Metrics', fontweight='bold')
        plt.xlabel('Period')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Trading Performance Distribution
        plt.subplot(2, 2, 3)
        plt.hist(summary['test_sharpe_ratios'], bins=8, alpha=0.7, color='green', label='Sharpe Ratio', edgecolor='black')
        plt.hist(summary['test_win_rates'], bins=8, alpha=0.7, color='blue', label='Win Rate', edgecolor='black')
        plt.title('📊 Trading Performance Distribution', fontweight='bold')
        plt.xlabel('Metric Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.axvline(summary['mean_sharpe_ratio'], color='green', linestyle='--', label=f'Sharpe Mean: {summary["mean_sharpe_ratio"]:.4f}')
        plt.axvline(summary['mean_win_rate'], color='blue', linestyle='--', label=f'Win Rate Mean: {summary["mean_win_rate"]:.4f}')
        plt.legend()
        
        # Plot 4: Trading Performance Box Plot
        plt.subplot(2, 2, 4)
        trading_performance_data = [summary['test_sharpe_ratios'], summary['test_win_rates'], summary['test_profit_factors']]
        plt.boxplot(trading_performance_data, tick_labels=['Sharpe Ratio', 'Win Rate', 'Profit Factor'])
        plt.title('🎯 Trading Performance Distribution', fontweight='bold')
        plt.ylabel('Metric Value')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.plots_dir / f'xgboost_walk_forward_results_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to: {plot_path}")

    def calculate_sharpe_ratio_trading_simulation(self, features_val, y_val, y_pred, y_pred_proba, data_val):
        """Calculate Sharpe ratio through trading simulation - same as hyperopt runner"""
        
        # Trading parameters (same as hyperopt runner)
        position_size = 0.04  # 4% of portfolio
        leverage = 100  # 100x leverage
        effective_position = position_size * leverage  # 400% effective position
        stop_loss = 0.008  # 0.8% stop loss
        trading_fee = 0.0002  # 0.02% per trade
        max_hold_hours = 24  # Max 24 hours hold time
        
        # Initialize trading simulation
        portfolio_value = 10000  # Starting portfolio value
        trades = []
        current_position = None
        entry_time = None
        entry_price = None
        
        # Loss tracking variables
        consecutive_losses = 0
        trading_pause_until = None
        
        # Get price data for validation set
        prices = data_val['close'].values
        timestamps = data_val.index
        
        # Daily Sharpe calculation parameters
        daily_returns = []
        current_day_returns = []
        current_day_start = None
        
        for i in range(len(y_pred)):
            current_time = timestamps[i]
            current_price = prices[i]
            prediction = y_pred[i]
            confidence = np.max(y_pred_proba[i])
            
            # Check if we're in trading pause
            if trading_pause_until is not None and current_time < trading_pause_until:
                continue  # Skip trading during pause
            elif trading_pause_until is not None and current_time >= trading_pause_until:
                trading_pause_until = None  # Resume trading
            
            # Daily grouping for Sharpe calculation
            if current_day_start is None:
                current_day_start = current_time
            elif (current_time - current_day_start).days >= 1:
                # End of day, calculate daily return
                if current_day_returns:
                    daily_return = np.sum(current_day_returns)
                    daily_returns.append(daily_return)
                    current_day_returns = []
                current_day_start = current_time
            
            # Check for stop loss on existing position
            if current_position is not None:
                hold_hours = (current_time - entry_time).total_seconds() / 3600
                
                # Calculate current P&L
                if current_position == 'long':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # short
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Check stop loss or max hold time
                if pnl_pct <= -stop_loss or hold_hours >= max_hold_hours:
                    # Close position
                    trade_return = pnl_pct * effective_position - (2 * trading_fee)  # Entry + exit fees
                    trades.append({
                        'type': current_position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return': trade_return,
                        'hold_hours': hold_hours,
                        'reason': 'stop_loss' if pnl_pct <= -stop_loss else 'max_hold_time'
                    })
                    
                    # Track consecutive losses
                    if trade_return < 0:
                        consecutive_losses += 1
                        if consecutive_losses >= 3:
                            trading_pause_until = current_time + timedelta(hours=12)
                            consecutive_losses = 0  # Reset after pause
                    else:
                        consecutive_losses = 0  # Reset on win
                    
                    # Add to daily returns
                    current_day_returns.append(trade_return)
                    
                    # Update portfolio
                    portfolio_value *= (1 + trade_return)
                    current_position = None
                    entry_time = None
                    entry_price = None
            
            # Check for new trading signals (only take signals 0 and 2)
            if prediction in [0, 2]:  # Early Down or Early Up
                # Open new position
                current_position = 'short' if prediction == 0 else 'long'
                entry_time = current_time
                entry_price = current_price
                
                # Pay entry fee
                portfolio_value *= (1 - trading_fee)
        
        # Close any remaining position at the end
        if current_position is not None:
            final_price = prices[-1]
            if current_position == 'long':
                pnl_pct = (final_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - final_price) / entry_price
            
            trade_return = pnl_pct * effective_position - (2 * trading_fee)
            trades.append({
                'type': current_position,
                'entry_price': entry_price,
                'exit_price': final_price,
                'return': trade_return,
                'hold_hours': (timestamps[-1] - entry_time).total_seconds() / 3600,
                'reason': 'end_of_period'
            })
            current_day_returns.append(trade_return)
        
        # Add final day returns
        if current_day_returns:
            daily_return = np.sum(current_day_returns)
            daily_returns.append(daily_return)
        
        # Calculate Sharpe ratio (daily)
        if len(daily_returns) < 2:
            return -10.0, 0.0, 0.0, 0.0  # Penalty for insufficient data
        
        daily_returns = np.array(daily_returns)
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
        
        # Calculate additional metrics
        if trades:
            trade_returns = [t['return'] for t in trades]
            profitable_trades = [r for r in trade_returns if r > 0]
            win_rate = len(profitable_trades) / len(trade_returns) if trade_returns else 0
            
            # Profit factor
            gross_profit = sum([r for r in trade_returns if r > 0])
            gross_loss = abs(sum([r for r in trade_returns if r < 0]))
            profit_factor = gross_profit / (gross_loss + 1e-8)
            
            # Max drawdown
            cumulative_returns = np.cumsum(trade_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = np.min(drawdown)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            max_drawdown = 0.0
        
        return sharpe_ratio, win_rate, profit_factor, max_drawdown


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
        # Default to 2025-05-29
        end_date = datetime(2025, 5, 29)
        
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        # Default to 2024-01-02
        start_date = datetime(2024, 1, 2)
    
    # Run optimization
    results = optimizer.run_optimization(start_date, end_date)
    
    # Generate summary report if we have results
    if results['periods']:
        optimizer.generate_summary_report()
    
    return results


if __name__ == "__main__":
    main() 