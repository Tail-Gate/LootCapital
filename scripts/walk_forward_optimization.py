#!/usr/bin/env python3
"""
Walk-Forward Optimization for STGNN Model

This script implements walk-forward optimization to:
1. Avoid look-ahead bias
2. Test model robustness across different time periods
3. Validate strategy performance on out-of-sample data
4. Optimize hyperparameters over time
5. Uses SMOTE for class balancing in training
6. Saves TorchScript models for production inference
7. Saves fitted scalers and feature lists for consistent preprocessing
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from imblearn.over_sampling import SMOTE
import argparse
import gc  # For garbage collection
import psutil  # For memory monitoring
import joblib  # For saving scaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from utils.feature_generator import FeatureGenerator
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

# Import the classification model and trainer
from scripts.train_stgnn_improved import STGNNClassificationModel, ClassificationSTGNNTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/walk_forward_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def manage_memory():
    """Force garbage collection and log memory usage for CPU-only training"""
    gc.collect()
    # No CUDA operations for CPU-only training
    
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

class EnhancedSTGNNDataProcessor(STGNNDataProcessor):
    """Enhanced STGNN data processor that uses FeatureGenerator for comprehensive features"""
    
    def __init__(self, config: STGNNConfig, market_data: MarketData, technical_indicators: TechnicalIndicators):
        super().__init__(config, market_data, technical_indicators)
        self.feature_generator = FeatureGenerator()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features using FeatureGenerator for comprehensive feature engineering
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with prepared features
        """
        logger.info("Using FeatureGenerator for comprehensive feature engineering...")
        
        # Store original price data for event-based analysis (from parent class)
        self._original_prices = data['close'].copy()
        
        # Use FeatureGenerator to generate comprehensive features
        features = self.feature_generator.generate_features(data)
        
        # CRITICAL DEBUG: Add comprehensive NaN/Inf checks after feature generation
        if features.isnull().any().any() or (features == np.inf).any().any() or (features == -np.inf).any().any():
            logger.error(f"DEBUG: NaN/Inf detected in 'features' DataFrame after feature generation. Shape: {features.shape}")
            logger.error("--- Head of problematic features ---")
            logger.error(features.head())
            logger.error("--- Tail of problematic features ---")
            logger.error(features.tail())
            logger.error("--- Columns with NaN/Inf values ---")
            nan_inf_cols = []
            for col in features.columns:
                if features[col].isnull().any() or (features[col] == np.inf).any() or (features[col] == -np.inf).any():
                    nan_inf_cols.append(col)
                    logger.error(f"  Column '{col}' has NaN/Inf.")
                    # Print statistics for the problematic column
                    col_data = features[col].replace([np.inf, -np.inf], np.nan)
                    logger.error(f"    Stats for {col}: min={col_data.min()}, max={col_data.max()}, mean={col_data.mean()}, NaNs={col_data.isnull().sum()}")
            # Optionally, save the problematic DataFrame to a CSV for manual inspection
            # features.to_csv("problematic_features_after_generation.csv")
            raise ValueError("NaN/Inf detected in features after generation. Stopping to debug.")
        
        # Store returns separately for target calculation
        self._returns = features['returns'] if 'returns' in features.columns else pd.Series(0, index=data.index)
        
        # Ensure all required features from config are present
        for feat in self.config.features:
            if feat not in features.columns:
                logger.warning(f"Feature '{feat}' not found in FeatureGenerator output, using 0")
                features[feat] = 0
        
        # Select only the features specified in config
        features = features[self.config.features]
        
        # Handle missing/infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill()
        features = features.fillna(0)
        
        logger.info(f"Generated {len(features.columns)} features using FeatureGenerator")
        logger.info(f"Feature columns: {list(features.columns)}")
        
        return features
        
    def set_scaler(self, scaler_type: str = 'minmax'):
        """Set scaler type for feature normalization"""
        super().set_scaler(scaler_type)
        logger.info(f"Enhanced data processor scaler set to: {scaler_type}")
        
    def fit_scaler(self, features: pd.DataFrame):
        """Fit scaler on training data"""
        super().fit_scaler(features)
        logger.info("Enhanced data processor scaler fitted successfully")
        
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler"""
        return super().transform_features(features)

class WalkForwardOptimizer:
    """Walk-forward optimization for STGNN model"""
    
    def __init__(self, 
                 initial_config: STGNNConfig,
                 train_window_days: int = 180,  # 6 months training
                 test_window_days: int = 30,    # 1 month testing
                 step_size_days: int = 15,      # 2 weeks step
                 price_threshold: float = 0.005,
                 output_dir: str = "models",
                 reports_dir: str = "reports",
                 plots_dir: str = "plots",
                 logs_dir: str = "logs"):
        
        self.initial_config = initial_config
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_size_days = step_size_days
        self.price_threshold = price_threshold
        
        # Output directories
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.plots_dir = Path(plots_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'periods': [],
            'train_metrics': [],
            'test_metrics': [],
            'model_paths': [],
            'torchscript_paths': [],
            'scaler_paths': [],
            'features_paths': [],
            'metadata_paths': [],
            'configs': []
        }
        
        # Initialize components
        self.market_data = MarketData(data_source_path='data')  # Use local data directory
        self.technical_indicators = TechnicalIndicators()
        
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
    
    def prepare_period_data(self, start_date: datetime, end_date: datetime, config: STGNNConfig):
        """Prepare data for a specific time period using enhanced data processor"""
        
        logger.info(f"Preparing data for period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Create enhanced data processor with FeatureGenerator
        data_processor = EnhancedSTGNNDataProcessor(config, self.market_data, self.technical_indicators)
        
        # Set scaler type if specified
        if hasattr(self, 'scaler_type'):
            data_processor.set_scaler(self.scaler_type)
        
        # Override the data loading to use specific time range
        original_get_data = self.market_data.get_data
        
        def get_data_with_range(symbol, start_time=None, end_time=None):
            logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
            return original_get_data(symbol, start_date, end_date)
        
        # Temporarily override the method
        self.market_data.get_data = get_data_with_range
        
        try:
            # Prepare data using enhanced processor
            logger.info("Calling enhanced data_processor.prepare_data()...")
            
            # Manage memory before heavy operation
            manage_memory()
            
            start_time = time.time()
            X, adj, y = data_processor.prepare_data()
            data_prep_time = time.time() - start_time
            logger.info(f"Enhanced data preparation completed in {data_prep_time:.2f} seconds")
            logger.info(f"Data shapes - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
            
            # Manage memory after heavy operation
            manage_memory()
            
            # Convert to classification targets
            logger.info("Converting to classification targets...")
            y_flat = y.flatten().numpy()
            classes = np.ones(len(y_flat), dtype=int)  # Default to no direction
            classes[y_flat > config.price_threshold] = 2   # Up (use config threshold)
            classes[y_flat < -config.price_threshold] = 0  # Down (use config threshold)
            y_classes = torch.LongTensor(classes.reshape(y.shape))
            
            # Log class distribution
            unique, counts = np.unique(classes, return_counts=True)
            class_dist = dict(zip(unique, counts))
            logger.info(f"Class distribution: {class_dist}")
            
            return X, adj, y_classes, data_processor
            
        except Exception as e:
            logger.error(f"Error in prepare_period_data: {e}")
            raise
        finally:
            # Restore original method
            self.market_data.get_data = original_get_data
    
    def train_and_evaluate_period(self, period_data: dict, config: STGNNConfig):
        """Train model on training data and evaluate on test data"""
        
        logger.info(f"Processing period: {period_data['period_name']}")
        period_start_time = time.time()
        
        try:
            # Prepare training data
            logger.info("Preparing training data...")
            X_train, adj_train, y_train, data_processor = self.prepare_period_data(
                period_data['train_start'], 
                period_data['train_end'], 
                config
            )
            
            # Prepare test data
            logger.info("Preparing test data...")
            X_test, adj_test, y_test, _ = self.prepare_period_data(
                period_data['test_start'], 
                period_data['test_end'], 
                config
            )
            
            # Check if we have enough data
            if len(X_train) < 100 or len(X_test) < 20:
                logger.warning(f"Insufficient data for period {period_data['period_name']}")
                logger.warning(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
                return None
            
            logger.info(f"Data validation passed - Training: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            # Calculate class weights from training data with optimized adjustments
            logger.info("Calculating class weights from training data with optimized adjustments...")
            y_train_flat = y_train.flatten().numpy()
            
            # FIXED: Use the already-calculated class labels from prepare_period_data
            # Previously, we were recalculating classes here which caused class 0 to be lost
            # because the class assignment logic was being applied twice incorrectly
            classes = y_train_flat.astype(int)  # Use the already-calculated classes
            
            # Debug: Verify class distribution
            unique_classes, class_counts_debug = np.unique(classes, return_counts=True)
            logger.info(f"DEBUG: Class distribution from y_train: {dict(zip(unique_classes, class_counts_debug))}")
            logger.info(f"DEBUG: Unique classes found: {unique_classes}")
            logger.info(f"DEBUG: Total samples: {len(classes)}")
            
            # Calculate class weights
            from collections import Counter
            class_counts = Counter(classes)
            total_samples = len(classes)
            class_weights = []
            
            # Get optimized class multipliers if available
            class_multipliers = getattr(self, 'optimized_class_multipliers', {
                'class_multiplier_0': 3.2,
                'class_multiplier_1': 2.0,
                'class_multiplier_2': 3.2
            })
            
            for i in range(3):
                if class_counts[i] > 0:
                    weight = total_samples / (len(class_counts) * class_counts[i])
                    
                    # --- START Optimized Weight Adjustment ---
                    multiplier_key = f'class_multiplier_{i}'
                    multiplier = class_multipliers.get(multiplier_key, 1.0)
                    weight *= multiplier
                    
                    class_name = {0: 'Down', 1: 'No Direction', 2: 'Up'}[i]
                    logger.info(f"  Applied {multiplier:.2f}x multiplier to Class {i} ({class_name}) weight: {weight:.4f}")
                    # --- END Optimized Weight Adjustment ---
                    
                    class_weights.append(weight)
                else:
                    class_weights.append(0.0)  # Avoid division by zero for missing classes
            class_weights = torch.FloatTensor(class_weights)
            
            logger.info(f"Class distribution: {dict(class_counts)}")
            logger.info(f"Original class weights: {[total_samples / (len(class_counts) * class_counts[i]) if class_counts[i] > 0 else 0.0 for i in range(3)]}")
            logger.info(f"Adjusted class weights: {class_weights}")
            
            # Create trainer with pre-calculated class weights
            logger.info("Creating trainer...")
            trainer = ClassificationSTGNNTrainer(
                config, 
                data_processor, 
                config.price_threshold,  # Use optimized price threshold from config
                focal_alpha=config.focal_alpha,  # Use config parameter
                focal_gamma=config.focal_gamma,  # Use config parameter
                class_weights=class_weights,  # Pass pre-calculated weights
                start_time=period_data['train_start'],  # Pass training start time
                end_time=period_data['train_end']       # Pass training end time
            )
            
            # Train model
            logger.info(f"Starting training for period {period_data['period_name']}...")
            training_start_time = time.time()
            training_history = trainer.train()
            training_time = time.time() - training_start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate on training data
            logger.info("Evaluating on training data...")
            train_metrics = trainer.evaluate(X_train, y_train)
            
            # Evaluate on test data
            logger.info("Evaluating on test data...")
            test_metrics = trainer.evaluate(X_test, y_test)
            
            # Save probabilities and true labels for analysis
            logger.info("Saving probabilities and true labels for analysis...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create probability analysis directory
            prob_analysis_dir = self.output_dir / "probability_analysis"
            prob_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Save test set probabilities and true labels
            test_probabilities = test_metrics['probabilities']  # Shape: [n_samples, 3]
            test_true_labels = test_metrics['true_labels']      # Shape: [n_samples]
            test_predictions = test_metrics['predictions']      # Shape: [n_samples]
            
            # Save as numpy arrays
            prob_file = prob_analysis_dir / f'test_probabilities_{period_data["period_name"]}_{timestamp}.npy'
            labels_file = prob_analysis_dir / f'test_true_labels_{period_data["period_name"]}_{timestamp}.npy'
            pred_file = prob_analysis_dir / f'test_predictions_{period_data["period_name"]}_{timestamp}.npy'
            
            np.save(prob_file, test_probabilities)
            np.save(labels_file, test_true_labels)
            np.save(pred_file, test_predictions)
            
            # Also save as CSV for easier analysis
            prob_df = pd.DataFrame(
                test_probabilities, 
                columns=['prob_down', 'prob_no_direction', 'prob_up']
            )
            prob_df['true_label'] = test_true_labels
            prob_df['predicted_label'] = test_predictions
            prob_df['period_name'] = period_data['period_name']
            
            csv_file = prob_analysis_dir / f'test_probabilities_{period_data["period_name"]}_{timestamp}.csv'
            prob_df.to_csv(csv_file, index=False)
            
            logger.info(f"Saved probability analysis files:")
            logger.info(f"  Probabilities: {prob_file}")
            logger.info(f"  True labels: {labels_file}")
            logger.info(f"  Predictions: {pred_file}")
            logger.info(f"  CSV summary: {csv_file}")
            
            # Log probability statistics for quick analysis
            logger.info(f"Probability statistics for period {period_data['period_name']}:")
            logger.info(f"  Test samples: {len(test_true_labels)}")
            logger.info(f"  True Down samples: {np.sum(test_true_labels == 0)}")
            logger.info(f"  True No Direction samples: {np.sum(test_true_labels == 1)}")
            logger.info(f"  True Up samples: {np.sum(test_true_labels == 2)}")
            
            # Analyze Down class predictions
            down_mask = test_true_labels == 0
            if np.any(down_mask):
                down_probs = test_probabilities[down_mask]
                logger.info(f"  Down class (true=0) probability stats:")
                logger.info(f"    Mean prob_down: {np.mean(down_probs[:, 0]):.4f}")
                logger.info(f"    Mean prob_no_direction: {np.mean(down_probs[:, 1]):.4f}")
                logger.info(f"    Mean prob_up: {np.mean(down_probs[:, 2]):.4f}")
                logger.info(f"    Times prob_1 > prob_0: {np.sum(down_probs[:, 1] > down_probs[:, 0])}")
                logger.info(f"    Times prob_2 > prob_0: {np.sum(down_probs[:, 2] > down_probs[:, 0])}")
            
            # Analyze No Direction class predictions
            no_dir_mask = test_true_labels == 1
            if np.any(no_dir_mask):
                no_dir_probs = test_probabilities[no_dir_mask]
                logger.info(f"  No Direction class (true=1) probability stats:")
                logger.info(f"    Mean prob_down: {np.mean(no_dir_probs[:, 0]):.4f}")
                logger.info(f"    Mean prob_no_direction: {np.mean(no_dir_probs[:, 1]):.4f}")
                logger.info(f"    Mean prob_up: {np.mean(no_dir_probs[:, 2]):.4f}")
                logger.info(f"    Times prob_0 > prob_1: {np.sum(no_dir_probs[:, 0] > no_dir_probs[:, 1])}")
                logger.info(f"    Times prob_2 > prob_1: {np.sum(no_dir_probs[:, 2] > no_dir_probs[:, 1])}")
            
            # Save model
            logger.info("Saving model...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.output_dir / f'wfo_stgnn_{period_data["period_name"]}_{timestamp}.pt'
            
            # 1. Save regular PyTorch model
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'config': config,
                'training_history': training_history,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'period_data': period_data,
                'probability_files': {
                    'probabilities': str(prob_file),
                    'true_labels': str(labels_file),
                    'predictions': str(pred_file),
                    'csv_summary': str(csv_file)
                }
            }, model_path)
            
            # 2. Convert to TorchScript and save
            logger.info("Converting model to TorchScript...")
            trainer.model.eval()  # Set to evaluation mode
            
            # Create dummy inputs for tracing with exact shapes and types
            # X shape: (1, seq_len, num_nodes, input_dim)
            dummy_X = X_train[0:1].clone()  # Take first sample as dummy input
            # adj shape: (1, num_nodes, num_nodes)
            dummy_adj = adj_train[0:1].clone()  # Take first sample as dummy input
            
            logger.info(f"TorchScript tracing with dummy inputs - X shape: {dummy_X.shape}, adj shape: {dummy_adj.shape}")
            
            # Trace the model
            with torch.no_grad():
                scripted_model = torch.jit.trace(trainer.model, (dummy_X, dummy_adj))
            
            # Save TorchScript model
            torchscript_path = self.output_dir / f'wfo_stgnn_torchscript_{period_data["period_name"]}_{timestamp}.pt'
            scripted_model.save(str(torchscript_path))
            logger.info(f"TorchScript model saved to: {torchscript_path}")
            
            # 3. Save fitted scaler
            logger.info("Saving fitted scaler...")
            scaler_path = self.output_dir / f'wfo_stgnn_scaler_{period_data["period_name"]}_{timestamp}.joblib'
            joblib.dump(data_processor.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
            
            # 4. Save feature list (order is critical for inference)
            logger.info("Saving feature list...")
            features_path = self.output_dir / f'wfo_stgnn_features_{period_data["period_name"]}_{timestamp}.json'
            with open(features_path, 'w') as f:
                json.dump(config.features, f, indent=4)
            logger.info(f"Feature list saved to: {features_path}")
            
            # 5. Save inference metadata
            inference_metadata = {
                'model_path': str(model_path),
                'torchscript_path': str(torchscript_path),
                'scaler_path': str(scaler_path),
                'features_path': str(features_path),
                'input_shapes': {
                    'X_shape': list(dummy_X.shape),
                    'adj_shape': list(dummy_adj.shape)
                },
                'config': {
                    'num_nodes': config.num_nodes,
                    'input_dim': config.input_dim,
                    'seq_len': config.seq_len,
                    'output_dim': config.output_dim
                },
                'period_name': period_data['period_name'],
                'timestamp': timestamp
            }
            
            metadata_path = self.output_dir / f'wfo_stgnn_inference_metadata_{period_data["period_name"]}_{timestamp}.json'
            with open(metadata_path, 'w') as f:
                json.dump(inference_metadata, f, indent=4)
            logger.info(f"Inference metadata saved to: {metadata_path}")
            
            period_time = time.time() - period_start_time
            logger.info(f"Period {period_data['period_name']} completed in {period_time:.2f} seconds")
            
            return {
                'period_name': period_data['period_name'],
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model_path': str(model_path),
                'torchscript_path': str(torchscript_path),
                'scaler_path': str(scaler_path),
                'features_path': str(features_path),
                'metadata_path': str(metadata_path),
                'config': config,
                'training_history': training_history
            }
            
        except Exception as e:
            logger.error(f"Error in train_and_evaluate_period for {period_data['period_name']}: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def run_optimization(self, start_date: datetime, end_date: datetime):
        """Run walk-forward optimization"""
        
        logger.info(f"Starting walk-forward optimization from {start_date} to {end_date}")
        optimization_start_time = time.time()
        
        # Generate data splits
        logger.info("Generating walk-forward data splits...")
        splits = self.get_data_splits(start_date, end_date)
        logger.info(f"Generated {len(splits)} walk-forward periods")
        
        # Log split details
        for i, split in enumerate(splits[:3]):  # Log first 3 splits
            logger.info(f"Split {i+1}: {split['period_name']} "
                       f"(Train: {split['train_start'].strftime('%Y-%m-%d')} to {split['train_end'].strftime('%Y-%m-%d')}, "
                       f"Test: {split['test_start'].strftime('%Y-%m-%d')} to {split['test_end'].strftime('%Y-%m-%d')})")
        
        if len(splits) > 3:
            logger.info(f"... and {len(splits) - 3} more splits")
        
        # Process each period
        successful_periods = 0
        failed_periods = 0
        
        for i, split in enumerate(tqdm(splits, desc="Processing periods", unit="period")):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing period {i+1}/{len(splits)}: {split['period_name']}")
            logger.info(f"Progress: {i+1}/{len(splits)} ({((i+1)/len(splits)*100):.1f}%)")
            logger.info(f"{'='*60}")
            
            # Optionally optimize hyperparameters for this period
            config = self.optimize_hyperparameters(split, i)
            
            # Train and evaluate
            result = self.train_and_evaluate_period(split, config)
            
            if result is not None:
                successful_periods += 1
                self.results['periods'].append(result['period_name'])
                self.results['train_metrics'].append(result['train_metrics'])
                self.results['test_metrics'].append(result['test_metrics'])
                self.results['model_paths'].append(result['model_path'])
                self.results['torchscript_paths'].append(result['torchscript_path'])
                self.results['scaler_paths'].append(result['scaler_path'])
                self.results['features_paths'].append(result['features_path'])
                self.results['metadata_paths'].append(result['metadata_path'])
                self.results['configs'].append(result['config'])
                
                # Log results
                logger.info(f"✓ Period {split['period_name']} completed successfully")
                logger.info(f"  Test Accuracy: {result['test_metrics']['classification_report']['accuracy']:.4f}")
                logger.info(f"  Test F1 (Up): {result['test_metrics']['f1'][2]:.4f}")
                logger.info(f"  Test F1 (Down): {result['test_metrics']['f1'][0]:.4f}")
                logger.info(f"  Inference files created:")
                logger.info(f"    TorchScript: {result['torchscript_path']}")
                logger.info(f"    Scaler: {result['scaler_path']}")
                logger.info(f"    Features: {result['features_path']}")
                logger.info(f"    Metadata: {result['metadata_path']}")
            else:
                failed_periods += 1
                logger.error(f"✗ Period {split['period_name']} failed")
            
            # Log progress summary
            elapsed_time = time.time() - optimization_start_time
            avg_time_per_period = elapsed_time / (i + 1)
            remaining_periods = len(splits) - (i + 1)
            estimated_remaining_time = remaining_periods * avg_time_per_period
            
            logger.info(f"Progress Summary:")
            logger.info(f"  Completed: {i+1}/{len(splits)} periods")
            logger.info(f"  Successful: {successful_periods}, Failed: {failed_periods}")
            logger.info(f"  Elapsed time: {elapsed_time/60:.1f} minutes")
            logger.info(f"  Avg time per period: {avg_time_per_period/60:.1f} minutes")
            logger.info(f"  Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
        
        # Final summary
        total_time = time.time() - optimization_start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD OPTIMIZATION COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total periods: {len(splits)}")
        logger.info(f"Successful periods: {successful_periods}")
        logger.info(f"Failed periods: {failed_periods}")
        if len(splits) > 0:
            logger.info(f"Success rate: {successful_periods/len(splits)*100:.1f}%")
        else:
            logger.warning("No walk-forward periods generated - check your time window parameters")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Average time per period: {total_time/len(splits)/60:.1f} minutes" if len(splits) > 0 else "No periods to average")
        
        # Generate summary report
        if successful_periods > 0:
            self.generate_summary_report()
        else:
            logger.error("No successful periods to generate report")
        
        return self.results
    
    def optimize_hyperparameters(self, split: dict, period_index: int):
        """Optimize hyperparameters for the current period"""
        
        # Try to load optimized parameters from hyperparameter optimization
        optimized_params = self.load_optimized_hyperparameters()
        
        if optimized_params:
            logger.info(f"Using optimized hyperparameters for period {period_index}")
            config = STGNNConfig(
                num_nodes=self.initial_config.num_nodes,
                input_dim=self.initial_config.input_dim,
                hidden_dim=optimized_params.get('hidden_dim', 96),
                output_dim=3,  # 3 classes: down/no direction/up
                num_layers=optimized_params.get('num_layers', 2),
                dropout=optimized_params.get('dropout', 0.2),
                kernel_size=optimized_params.get('kernel_size', 3),
                learning_rate=optimized_params.get('learning_rate', 0.0005),
                batch_size=optimized_params.get('batch_size', 2),
                num_epochs=40,  # Keep fixed for walk-forward
                early_stopping_patience=8,
                seq_len=optimized_params.get('seq_len', 150),  # Increased default for better feature capture
                prediction_horizon=15,
                features=self.initial_config.features,
                assets=self.initial_config.assets,
                confidence_threshold=0.51,
                buy_threshold=0.6,
                sell_threshold=0.4,
                retrain_interval=24,
                focal_alpha=optimized_params.get('focal_alpha', 1.0),
                focal_gamma=optimized_params.get('focal_gamma', 2.0),
                # Feature engineering parameters
                rsi_period=optimized_params.get('rsi_period', 14),
                macd_fast_period=optimized_params.get('macd_fast_period', 12),
                macd_slow_period=optimized_params.get('macd_slow_period', 26),
                macd_signal_period=optimized_params.get('macd_signal_period', 9),
                bb_period=optimized_params.get('bb_period', 20),
                bb_num_std_dev=optimized_params.get('bb_num_std_dev', 2.0),
                atr_period=optimized_params.get('atr_period', 14),
                adx_period=optimized_params.get('adx_period', 14),
                volume_ma_period=optimized_params.get('volume_ma_period', 20),
                price_momentum_lookback=optimized_params.get('price_momentum_lookback', 5),
                price_threshold=optimized_params.get('price_threshold', 0.005)  # 0.5% threshold
            )
            
            # Store optimized class multipliers for use in training
            self.optimized_class_multipliers = {
                'class_multiplier_0': optimized_params.get('class_multiplier_0', 3.2),
                'class_multiplier_1': optimized_params.get('class_multiplier_1', 2.0),
                'class_multiplier_2': optimized_params.get('class_multiplier_2', 3.2)
            }
        else:
            logger.info(f"Using default hyperparameters for period {period_index}")
            config = STGNNConfig(
                num_nodes=self.initial_config.num_nodes,
                input_dim=self.initial_config.input_dim,
                hidden_dim=96,  # Restored complexity for better learning
                output_dim=3,  # 3 classes: down/no direction/up
                num_layers=2,  # Restored 2 layers for deeper learning
                dropout=0.2,  # Keep reduced regularization
                kernel_size=3,
                learning_rate=0.0005,  # Restored for stable training
                batch_size=2,  # Keep small for memory management
                num_epochs=40,  # Restored for more training time
                early_stopping_patience=8,  # Restored for more training time
                seq_len=150,  # Increased for better capture of technical indicator patterns
                prediction_horizon=15,
                features=self.initial_config.features,
                assets=self.initial_config.assets,
                confidence_threshold=0.51,
                buy_threshold=0.6,
                sell_threshold=0.4,
                retrain_interval=24,
                focal_alpha=1.0,  # Keep this as 1.0 given you have class_weights
                focal_gamma=2.0,   # Default focusing parameter
                # Default feature engineering parameters
                rsi_period=14,
                macd_fast_period=12,
                macd_slow_period=26,
                macd_signal_period=9,
                bb_period=20,
                bb_num_std_dev=2.0,
                atr_period=14,
                adx_period=14,
                volume_ma_period=20,
                price_momentum_lookback=5,
                price_threshold=0.005  # 0.5% threshold
            )
            
            # Use default class multipliers
            self.optimized_class_multipliers = {
                'class_multiplier_0': 3.2,
                'class_multiplier_1': 2.0,
                'class_multiplier_2': 3.2
            }
        
        return config
    
    def load_optimized_hyperparameters(self):
        """Load optimized hyperparameters from hyperparameter optimization results"""
        import glob
        import os
        
        # Look for the most recent optimized parameters file
        config_dir = Path('config')
        if not config_dir.exists():
            return None
        
        # Find all enhanced hyperparameter files
        pattern = config_dir / 'stgnn_enhanced_best_params_*.json'
        param_files = list(config_dir.glob('stgnn_enhanced_best_params_*.json'))
        
        if not param_files:
            logger.info("No optimized hyperparameter files found")
            return None
        
        # Get the most recent file
        latest_file = max(param_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r') as f:
                optimized_params = json.load(f)
            
            logger.info(f"Loaded optimized hyperparameters from: {latest_file}")
            logger.info(f"Optimized parameters: {optimized_params}")
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Failed to load optimized hyperparameters: {e}")
            return None
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        logger.info("Generating walk-forward optimization summary report")
        
        # Calculate aggregate metrics
        test_accuracies = [metrics['classification_report']['accuracy'] for metrics in self.results['test_metrics']]
        test_f1_up = [metrics['f1'][2] for metrics in self.results['test_metrics']]
        test_f1_down = [metrics['f1'][0] for metrics in self.results['test_metrics']]
        test_f1_no_direction = [metrics['f1'][1] for metrics in self.results['test_metrics']]
        
        # Extract precision and recall metrics
        test_precision_up = [metrics['precision'][2] for metrics in self.results['test_metrics']]
        test_precision_down = [metrics['precision'][0] for metrics in self.results['test_metrics']]
        test_precision_no_direction = [metrics['precision'][1] for metrics in self.results['test_metrics']]
        
        test_recall_up = [metrics['recall'][2] for metrics in self.results['test_metrics']]
        test_recall_down = [metrics['recall'][0] for metrics in self.results['test_metrics']]
        test_recall_no_direction = [metrics['recall'][1] for metrics in self.results['test_metrics']]
        
        # Calculate statistics
        summary = {
            'total_periods': len(self.results['periods']),
            'mean_test_accuracy': np.mean(test_accuracies),
            'std_test_accuracy': np.std(test_accuracies),
            'mean_f1_up': np.mean(test_f1_up),
            'mean_f1_down': np.mean(test_f1_down),
            'mean_f1_no_direction': np.mean(test_f1_no_direction),
            'mean_precision_up': np.mean(test_precision_up),
            'mean_precision_down': np.mean(test_precision_down),
            'mean_precision_no_direction': np.mean(test_precision_no_direction),
            'mean_recall_up': np.mean(test_recall_up),
            'mean_recall_down': np.mean(test_recall_down),
            'mean_recall_no_direction': np.mean(test_recall_no_direction),
            'best_period': self.results['periods'][np.argmax(test_accuracies)],
            'worst_period': self.results['periods'][np.argmin(test_accuracies)],
            'best_accuracy': np.max(test_accuracies),
            'worst_accuracy': np.min(test_accuracies),
            'periods': self.results['periods'],
            'test_accuracies': test_accuracies,
            'test_f1_up': test_f1_up,
            'test_f1_down': test_f1_down,
            'test_f1_no_direction': test_f1_no_direction,
            'test_precision_up': test_precision_up,
            'test_precision_down': test_precision_down,
            'test_precision_no_direction': test_precision_no_direction,
            'test_recall_up': test_recall_up,
            'test_recall_down': test_recall_down,
            'test_recall_no_direction': test_recall_no_direction
        }
        
        # Print summary
        print("\n" + "="*80)
        print("WALK-FORWARD OPTIMIZATION SUMMARY REPORT")
        print("="*80)
        print(f"Total periods: {summary['total_periods']}")
        print(f"Mean test accuracy: {summary['mean_test_accuracy']:.4f} ± {summary['std_test_accuracy']:.4f}")
        print(f"\nMean F1 Scores:")
        print(f"  Up: {summary['mean_f1_up']:.4f}")
        print(f"  Down: {summary['mean_f1_down']:.4f}")
        print(f"  No Direction: {summary['mean_f1_no_direction']:.4f}")
        print(f"\nMean Precision Scores:")
        print(f"  Up: {summary['mean_precision_up']:.4f}")
        print(f"  Down: {summary['mean_precision_down']:.4f}")
        print(f"  No Direction: {summary['mean_precision_no_direction']:.4f}")
        print(f"\nMean Recall Scores:")
        print(f"  Up: {summary['mean_recall_up']:.4f}")
        print(f"  Down: {summary['mean_recall_down']:.4f}")
        print(f"  No Direction: {summary['mean_recall_no_direction']:.4f}")
        print(f"\nBest period: {summary['best_period']} (accuracy: {summary['best_accuracy']:.4f})")
        print(f"Worst period: {summary['worst_period']} (accuracy: {summary['worst_accuracy']:.4f})")
        
        # Print period-by-period results
        print("\nPeriod-by-Period Results:")
        print(f"{'Period':<20} {'Accuracy':<10} {'F1(Up)':<8} {'F1(Down)':<10} {'F1(NoDir)':<10} {'Prec(Up)':<8} {'Rec(Up)':<8}")
        print("-" * 85)
        for i, period in enumerate(self.results['periods']):
            print(f"{period:<20} {test_accuracies[i]:<10.4f} {test_f1_up[i]:<8.4f} {test_f1_down[i]:<10.4f} {test_f1_no_direction[i]:<10.4f} {test_precision_up[i]:<8.4f} {test_recall_up[i]:<8.4f}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f'walk_forward_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        logger.info(f"Detailed report saved to: {report_path}")
        
        # Create visualization
        self.create_visualizations(summary)
        
        return summary
    
    def create_visualizations(self, summary: dict):
        """Create visualizations of walk-forward results"""
        
        # Plot 1: Accuracy over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(summary['test_accuracies'], marker='o')
        plt.title('Test Accuracy Over Time')
        plt.xlabel('Period')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Plot 2: F1 scores over time
        plt.subplot(2, 2, 2)
        plt.plot(summary['test_f1_up'], label='Up', marker='o')
        plt.plot(summary['test_f1_down'], label='Down', marker='s')
        plt.plot(summary['test_f1_no_direction'], label='No Direction', marker='^')
        plt.title('F1 Scores Over Time')
        plt.xlabel('Period')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Distribution of accuracies
        plt.subplot(2, 2, 3)
        plt.hist(summary['test_accuracies'], bins=10, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Test Accuracies')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.axvline(summary['mean_test_accuracy'], color='red', linestyle='--', label=f'Mean: {summary["mean_test_accuracy"]:.4f}')
        plt.legend()
        
        # Plot 4: Box plot of F1 scores
        plt.subplot(2, 2, 4)
        f1_data = [summary['test_f1_up'], summary['test_f1_down'], summary['test_f1_no_direction']]
        plt.boxplot(f1_data, labels=['Up', 'Down', 'No Direction'])
        plt.title('F1 Score Distribution by Class')
        plt.ylabel('F1 Score')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.plots_dir / f'walk_forward_results_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {plot_path}")

    def calculate_expected_periods(self, start_date: datetime, end_date: datetime):
        """Calculate expected number of walk-forward periods"""
        total_days = (end_date - start_date).days
        required_days_per_period = self.train_window_days + self.test_window_days
        max_periods = (total_days - required_days_per_period) // self.step_size_days + 1
        
        return max_periods

    def check_data_availability(self, start_date: datetime, end_date: datetime):
        """Check if sufficient data is available for the specified time range"""
        try:
            # Test data loading for the full range
            test_data = self.market_data.get_data('ETH/USD', start_date, end_date)
            
            if test_data is None or test_data.empty:
                logger.warning("No data available for the specified time range")
                return False
            
            # Check if we have enough data points
            min_required_points = 100  # Reduced for testing (was 1000)
            if len(test_data) < min_required_points:
                logger.warning(f"Insufficient data points: {len(test_data)} < {min_required_points}")
                return False
            
            # Check data coverage
            data_start = test_data.index.min()
            data_end = test_data.index.max()
            coverage_days = (data_end - data_start).days
            
            logger.info(f"Data available from {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")
            logger.info(f"Data coverage: {coverage_days} days")
            logger.info(f"Total data points: {len(test_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return False

    def set_scaler_type(self, scaler_type: str):
        """Set scaler type for feature normalization"""
        self.scaler_type = scaler_type
        logger.info(f"Walk-forward optimizer scaler set to: {scaler_type}")

def create_wfo_config():
    """Create configuration for walk-forward optimization with comprehensive features"""
    
    # Focus on ETH/USD
    assets = ['ETH/USD']
    
    # Engineered features only - NO raw OHLCV data
    # These features are derived from price/volume data and are suitable for ML training
    features = [
        # Price-derived features (safe for ML)
        'returns', 'log_returns',
        
        # Technical indicators (derived from price)
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        
        # Volume-derived features (safe for ML)
        'volume_ma', 'volume_std', 'volume_surge', 'volume_ratio',
        
        # Momentum and regime features
        'price_momentum', 'volatility_regime',
        
        # Support/Resistance levels (derived from price)
        'support', 'resistance',
        
        # VWAP analysis
        'vwap_ratio',
        
        # Additional technical features
        'ma_crossover', 'swing_rsi', 'breakout_intensity',
        'adx', 'cumulative_delta', 'atr'
    ]
    
    # Create configuration
    config = STGNNConfig(
        num_nodes=len(assets),
        input_dim=len(features),
        hidden_dim=96,  # Restored complexity for better learning
        output_dim=3,  # 3 classes: down/no direction/up
        num_layers=2,  # Restored 2 layers for deeper learning
        dropout=0.2,  # Keep reduced regularization
        kernel_size=3,
        learning_rate=0.0005,  # Restored for stable training
        batch_size=2,  # Keep small for memory management
        num_epochs=40,  # Restored for more training time
        early_stopping_patience=8,  # Restored for more training time
        seq_len=150,  # Increased for better capture of technical indicator patterns
        prediction_horizon=15,
        features=features,
        assets=assets,
        confidence_threshold=0.51,
        buy_threshold=0.6,
        sell_threshold=0.4,
        retrain_interval=24,
        focal_alpha=1.0,  # Keep this as 1.0 given you have class_weights
        focal_gamma=2.0   # Default focusing parameter
    )
    
    return config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Walk-Forward Optimization for STGNN Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scripts/walk_forward_optimization.py

  # Run with custom output directories
  python scripts/walk_forward_optimization.py \\
    --output-dir /path/to/models \\
    --reports-dir /path/to/reports \\
    --plots-dir /path/to/plots \\
    --logs-dir /path/to/logs

  # Run with custom time range
  python scripts/walk_forward_optimization.py \\
    --start-date 2020-01-01 \\
    --end-date 2023-12-31 \\
    --output-dir /path/to/models \\
    --reports-dir /path/to/reports

  # Run with custom parameters
  python scripts/walk_forward_optimization.py \\
    --train-window-days 180 \\
    --test-window-days 30 \\
    --step-size-days 15 \\
    --price-threshold 0.005
        """
    )
    
    # Data and time range arguments
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for walk-forward optimization (YYYY-MM-DD format, defaults to 5 years ago)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for walk-forward optimization (YYYY-MM-DD format, defaults to today)'
    )
    
    # Output directory arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--reports-dir',
        type=str,
        default='reports',
        help='Directory to save reports (default: reports)'
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        default='plots',
        help='Directory to save plots (default: plots)'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='logs',
        help='Directory to save logs (default: logs)'
    )
    
    # Walk-forward parameters
    parser.add_argument(
        '--train-window-days',
        type=int,
        default=365,
        help='Training window size in days (default: 365)'
    )
    parser.add_argument(
        '--test-window-days',
        type=int,
        default=60,
        help='Testing window size in days (default: 60)'
    )
    parser.add_argument(
        '--step-size-days',
        type=int,
        default=30,
        help='Step size between periods in days (default: 30)'
    )
    parser.add_argument(
        '--price-threshold',
        type=float,
        default=0.005,  # 0.5% threshold for meaningful price movements
        help='Price threshold for classification (default: 0.02 for 2% movements)'
    )
    
    # Model parameters
    parser.add_argument(
        '--scaler-type',
        type=str,
        choices=['minmax', 'standard'],
        default='minmax',
        help='Type of scaler to use (default: minmax)'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output (only log to file)'
    )
    
    return parser.parse_args()

def setup_logging(logs_dir: str, log_level: str, quiet: bool = False):
    """Setup logging configuration"""
    
    # Create logs directory
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level_num = getattr(logging, log_level.upper())
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    handlers = []
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(logs_path / f'walk_forward_optimization_{timestamp}.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level_num)
    handlers.append(file_handler)
    
    # Console handler (if not quiet)
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level_num)
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level_num,
        handlers=handlers,
        force=True
    )

def main():
    """Main walk-forward optimization function with command-line argument support"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.logs_dir, args.log_level, args.quiet)
    
    logger.info("="*80)
    logger.info("STARTING WALK-FORWARD OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Arguments: {vars(args)}")
    main_start_time = time.time()
    
    try:
        # Create configuration
        logger.info("Creating configuration...")
        config = create_wfo_config()
        logger.info(f"Configuration created successfully")
        logger.info(f"Assets: {config.assets}")
        logger.info(f"Features: {len(config.features)} features")
        logger.info(f"Model parameters: hidden_dim={config.hidden_dim}, layers={config.num_layers}")
        
        # Create walk-forward optimizer with command-line parameters
        logger.info("Creating walk-forward optimizer...")
        wfo = WalkForwardOptimizer(
            initial_config=config,
            train_window_days=args.train_window_days,
            test_window_days=args.test_window_days,
            step_size_days=args.step_size_days,
            price_threshold=args.price_threshold,
            output_dir=args.output_dir,
            reports_dir=args.reports_dir,
            plots_dir=args.plots_dir,
            logs_dir=args.logs_dir
        )
        logger.info("Walk-forward optimizer created successfully")
        
        # Parse date arguments
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
            
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        else:
            # Default to 5 years ago
            start_date = end_date - timedelta(days=1825)  # 5 years (365 * 5)
        
        # Check data availability
        logger.info("Checking data availability...")
        if not wfo.check_data_availability(start_date, end_date):
            logger.error("Insufficient data available for walk-forward optimization")
            return None
        
        # Calculate expected periods
        expected_periods = wfo.calculate_expected_periods(start_date, end_date)
        
        logger.info(f"Configuration Summary:")
        logger.info(f"  Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"  Total data span: {(end_date - start_date).days} days")
        logger.info(f"  Training window: {args.train_window_days} days")
        logger.info(f"  Testing window: {args.test_window_days} days")
        logger.info(f"  Step size: {args.step_size_days} days")
        logger.info(f"  Expected walk-forward periods: {expected_periods}")
        logger.info(f"  Price threshold: {args.price_threshold}")
        logger.info(f"  Features: {len(config.features)} comprehensive features")
        logger.info(f"  Feature scaling: {args.scaler_type}")
        logger.info(f"  Output directories:")
        logger.info(f"    Models: {args.output_dir}")
        logger.info(f"    Reports: {args.reports_dir}")
        logger.info(f"    Plots: {args.plots_dir}")
        logger.info(f"    Logs: {args.logs_dir}")
        
        # Estimate total runtime
        estimated_time_per_period = 10  # minutes (conservative estimate)
        estimated_total_time = expected_periods * estimated_time_per_period
        logger.info(f"Estimated runtime: {estimated_total_time} minutes ({estimated_total_time/60:.1f} hours)")
        
        # Set scaler type for the walk-forward optimizer
        wfo.set_scaler_type(args.scaler_type)
        logger.info(f"Feature scaling enabled: {args.scaler_type}")
        
        # Run walk-forward optimization
        logger.info("\nStarting walk-forward optimization...")
        results = wfo.run_optimization(start_date, end_date)
        
        # Final summary
        total_time = time.time() - main_start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"WALK-FORWARD OPTIMIZATION COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"Total execution time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        logger.info(f"Results saved to:")
        logger.info(f"  Models: {args.output_dir}")
        logger.info(f"  Reports: {args.reports_dir}")
        logger.info(f"  Plots: {args.plots_dir}")
        logger.info(f"  Logs: {args.logs_dir}")
        
        return results
        
    except Exception as e:
        total_time = time.time() - main_start_time
        logger.error(f"\n{'='*80}")
        logger.error(f"WALK-FORWARD OPTIMIZATION FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Execution time before failure: {total_time/60:.1f} minutes")
        logger.error(f"Error: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 