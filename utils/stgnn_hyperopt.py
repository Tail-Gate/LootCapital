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
from scripts.train_stgnn_improved import STGNNClassificationModel, ClassificationSTGNNTrainer, WeightedFocalLoss
from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor

# Add a stream handler for print-like output
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to debug for more verbosity

# Clear existing handlers to prevent duplicate output when script is run multiple times
if logger.hasHandlers():
    logger.handlers.clear()

# Add a stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler) # <-- Simpler, always adds

def get_device():
    """Get the best available device for training with comprehensive logging."""
    print("[DEVICE] Checking available devices...")
    logger.info("[DEVICE] Checking available devices...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        print(f"[DEVICE] CUDA available with {cuda_count} device(s)")
        logger.info(f"[DEVICE] CUDA available with {cuda_count} device(s)")
        
        # Get current device info
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"[DEVICE] Current CUDA device: {current_device} - {device_name}")
        logger.info(f"[DEVICE] Current CUDA device: {current_device} - {device_name}")
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
        cached_memory = torch.cuda.memory_reserved(current_device) / 1024**3
        
        print(f"[DEVICE] GPU Memory - Total: {gpu_memory:.1f}GB, Allocated: {allocated_memory:.1f}GB, Cached: {cached_memory:.1f}GB")
        logger.info(f"[DEVICE] GPU Memory - Total: {gpu_memory:.1f}GB, Allocated: {allocated_memory:.1f}GB, Cached: {cached_memory:.1f}GB")
        
        # Use CUDA device 0 explicitly
        device = torch.device('cuda:0')
        print(f"[DEVICE] Using GPU device: {device}")
        logger.info(f"[DEVICE] Using GPU device: {device}")
        
        # Clear GPU cache to start fresh
        torch.cuda.empty_cache()
        print("[DEVICE] GPU cache cleared")
        logger.info("[DEVICE] GPU cache cleared")
        
        return device
    else:
        print("[DEVICE] CUDA not available, using CPU")
        logger.info("[DEVICE] CUDA not available, using CPU")
        
        # Log CPU information
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"[DEVICE] CPU cores available: {cpu_count}")
        logger.info(f"[DEVICE] CPU cores available: {cpu_count}")
        
        device = torch.device('cpu')
        print(f"[DEVICE] Using CPU device: {device}")
        logger.info(f"[DEVICE] Using CPU device: {device}")
        return device

def manage_memory():
    """Force garbage collection and log memory usage for both CPU and GPU training"""
    print("[MEMORY] Starting memory management...")
    logger.info("[MEMORY] Starting memory management...")
    
    # Force garbage collection
    gc.collect()
    print("[MEMORY] Garbage collection completed")
    logger.info("[MEMORY] Garbage collection completed")
    
    # Log CPU memory usage
    process = psutil.Process(os.getpid())
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_memory_gb = cpu_memory_mb / 1024
    print(f"[MEMORY] CPU Memory usage: {cpu_memory_mb:.1f} MB ({cpu_memory_gb:.2f} GB)")
    logger.info(f"[MEMORY] CPU Memory usage: {cpu_memory_mb:.1f} MB ({cpu_memory_gb:.2f} GB)")
    
    # Log GPU memory usage if available
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_allocated_mb = torch.cuda.memory_allocated(current_device) / 1024 / 1024
        gpu_cached_mb = torch.cuda.memory_reserved(current_device) / 1024 / 1024
        gpu_total_gb = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"[MEMORY] GPU Memory - Allocated: {gpu_allocated_mb:.1f} MB, Cached: {gpu_cached_mb:.1f} MB, Total: {gpu_total_gb:.1f} GB")
        logger.info(f"[MEMORY] GPU Memory - Allocated: {gpu_allocated_mb:.1f} MB, Cached: {gpu_cached_mb:.1f} MB, Total: {gpu_total_gb:.1f} GB")
        
        # Clear GPU cache if usage is high
        if gpu_cached_mb > 1000:  # Warning if GPU cache > 1GB
            print(f"[MEMORY] High GPU cache usage detected: {gpu_cached_mb:.1f} MB, clearing cache...")
            logger.warning(f"[MEMORY] High GPU cache usage detected: {gpu_cached_mb:.1f} MB, clearing cache...")
            torch.cuda.empty_cache()
            print("[MEMORY] GPU cache cleared")
            logger.info("[MEMORY] GPU cache cleared")
    
    # Force more aggressive cleanup
    if hasattr(sys, 'exc_clear'):
        sys.exc_clear()
    
    # Additional memory optimization for HPC
    if cpu_memory_mb > 1000:  # Warning if memory usage > 1GB
        print(f"[MEMORY] High CPU memory usage detected: {cpu_memory_mb:.1f} MB")
        logger.warning(f"[MEMORY] High CPU memory usage detected: {cpu_memory_mb:.1f} MB")
        # Force more aggressive cleanup
        gc.collect()
        print("[MEMORY] Additional garbage collection completed")
        logger.info("[MEMORY] Additional garbage collection completed")
    
    print("[MEMORY] Memory management completed")
    logger.info("[MEMORY] Memory management completed")

def load_stgnn_data(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and prepare data for STGNN training and validation with comprehensive logging
    
    Args:
        config: Configuration dictionary containing data parameters
        
    Returns:
        Tuple of (X_train, adj_train, y_train, X_val, adj_val, y_val)
    """
    print("[DATA] Starting STGNN data loading...")
    logger.info("[DATA] Starting STGNN data loading...")
    
    try:
        # Initialize strategy components
        print("[DATA] Initializing market data and technical indicators...")
        logger.info("[DATA] Initializing market data and technical indicators...")
        market_data = MarketData()
        technical_indicators = TechnicalIndicators()
        print("[DATA] Market data and technical indicators initialized successfully")
        logger.info("[DATA] Market data and technical indicators initialized successfully")
        
        # Create strategy instance
        print("[DATA] Creating STGNN strategy instance...")
        logger.info("[DATA] Creating STGNN strategy instance...")
        strategy = STGNNStrategy(config, market_data, technical_indicators)
        print("[DATA] STGNN strategy instance created successfully")
        logger.info("[DATA] STGNN strategy instance created successfully")
        
        # Prepare data
        print("[DATA] Preparing data using strategy...")
        logger.info("[DATA] Preparing data using strategy...")
        X, adj, y = strategy.prepare_data()
        
        # Validate data shapes
        print(f"[DATA] Data shapes - X: {X.shape if X is not None else 'None'}, adj: {adj.shape if adj is not None else 'None'}, y: {y.shape if y is not None else 'None'}")
        logger.info(f"[DATA] Data shapes - X: {X.shape if X is not None else 'None'}, adj: {adj.shape if adj is not None else 'None'}, y: {y.shape if y is not None else 'None'}")
        
        # Check for None values
        if X is None or adj is None or y is None:
            print("[DATA] ERROR: Data preparation returned None values")
            logger.error("[DATA] ERROR: Data preparation returned None values")
            raise ValueError("Data preparation returned None values")
        
        # Check for empty data
        if len(X) == 0 or len(y) == 0:
            print("[DATA] ERROR: Data preparation returned empty arrays")
            logger.error("[DATA] ERROR: Data preparation returned empty arrays")
            raise ValueError("Data preparation returned empty arrays")
        
        # Check for NaN/Inf values
        if np.isnan(X).any() or np.isinf(X).any():
            print("[DATA] ERROR: NaN/Inf values detected in X")
            logger.error("[DATA] ERROR: NaN/Inf values detected in X")
            raise ValueError("NaN/Inf values detected in X")
        
        if np.isnan(adj).any() or np.isinf(adj).any():
            print("[DATA] ERROR: NaN/Inf values detected in adj")
            logger.error("[DATA] ERROR: NaN/Inf values detected in adj")
            raise ValueError("NaN/Inf values detected in adj")
        
        if np.isnan(y).any() or np.isinf(y).any():
            print("[DATA] ERROR: NaN/Inf values detected in y")
            logger.error("[DATA] ERROR: NaN/Inf values detected in y")
            raise ValueError("NaN/Inf values detected in y")
        
        print("[DATA] Data validation passed successfully")
        logger.info("[DATA] Data validation passed successfully")
        
        # Split into train and validation sets (80/20 split)
        print("[DATA] Splitting data into train/validation sets (80/20)...")
        logger.info("[DATA] Splitting data into train/validation sets (80/20)...")
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        print(f"[DATA] Split sizes - Train: {len(X_train)}, Validation: {len(X_val)}")
        logger.info(f"[DATA] Split sizes - Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Convert to tensors
        print("[DATA] Converting data to PyTorch tensors...")
        logger.info("[DATA] Converting data to PyTorch tensors...")
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.FloatTensor(y_train)
        y_val = torch.FloatTensor(y_val)
        adj_train = torch.FloatTensor(adj)
        adj_val = torch.FloatTensor(adj)
        
        print(f"[DATA] Tensor shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, y_train: {y_train.shape}, y_val: {y_val.shape}")
        logger.info(f"[DATA] Tensor shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, y_train: {y_train.shape}, y_val: {y_val.shape}")
        
        print("[DATA] STGNN data loading completed successfully")
        logger.info("[DATA] STGNN data loading completed successfully")
        
        return X_train, adj_train, y_train, X_val, adj_val, y_val
        
    except Exception as e:
        print(f"[DATA] ERROR: Failed to load STGNN data: {e}")
        logger.error(f"[DATA] ERROR: Failed to load STGNN data: {e}")
        import traceback
        logger.error(f"[DATA] Traceback: {traceback.format_exc()}")
        raise

def create_classification_data_processor(config: STGNNConfig, start_time=None, end_time=None):
    """
    Create data processor for classification training with comprehensive logging
    
    Args:
        config: STGNN configuration
        start_time: Optional start time for data range
        end_time: Optional end time for data range
        
    Returns:
        Data processor instance
    """
    print("[PROCESSOR] Creating classification data processor...")
    logger.info("[PROCESSOR] Creating classification data processor...")
    
    try:
        print(f"[PROCESSOR] Config parameters - Assets: {config.assets}, Features: {len(config.features)}")
        logger.info(f"[PROCESSOR] Config parameters - Assets: {config.assets}, Features: {len(config.features)}")
        
        if start_time and end_time:
            print(f"[PROCESSOR] Data range: {start_time} to {end_time}")
            logger.info(f"[PROCESSOR] Data range: {start_time} to {end_time}")
        
        from market_analysis.market_data import MarketData
        from market_analysis.technical_indicators import TechnicalIndicators
        
        print("[PROCESSOR] Initializing market data...")
        logger.info("[PROCESSOR] Initializing market data...")
        market_data = MarketData()
        print("[PROCESSOR] Market data initialized successfully")
        logger.info("[PROCESSOR] Market data initialized successfully")
        
        print("[PROCESSOR] Initializing technical indicators...")
        logger.info("[PROCESSOR] Initializing technical indicators...")
        technical_indicators = TechnicalIndicators()
        print("[PROCESSOR] Technical indicators initialized successfully")
        logger.info("[PROCESSOR] Technical indicators initialized successfully")
        
        # Create data processor
        print("[PROCESSOR] Creating STGNN data processor...")
        logger.info("[PROCESSOR] Creating STGNN data processor...")
        data_processor = STGNNDataProcessor(config, market_data, technical_indicators)
        print("[PROCESSOR] STGNN data processor created successfully")
        logger.info("[PROCESSOR] STGNN data processor created successfully")
        
        return data_processor
        
    except Exception as e:
        print(f"[PROCESSOR] ERROR: Failed to create data processor: {e}")
        logger.error(f"[PROCESSOR] ERROR: Failed to create data processor: {e}")
        import traceback
        logger.error(f"[PROCESSOR] Traceback: {traceback.format_exc()}")
        raise

def objective(trial: optuna.Trial) -> float:
    print("\n" + "="*60)
    print("🚀 NEW OPTUNA TRIAL STARTING")
    print("="*60)
    print(f"📊 Trial number: {trial.number}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    logger.info("🚀 NEW OPTUNA TRIAL STARTING")
    logger.info("="*60)
    logger.info(f"📊 Trial number: {trial.number}")
    logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Force memory cleanup at start of trial
    print("[TRIAL] Starting memory cleanup...")
    logger.info("[TRIAL] Starting memory cleanup...")
    manage_memory()
    print("[TRIAL] Memory cleanup completed")
    logger.info("[TRIAL] Memory cleanup completed")
    
    try:
        print("[OBJECTIVE] Defining hyperparameter search space...")
        logger.info("[OBJECTIVE] Defining hyperparameter search space...")
        logger.debug(f"Trial params: {trial.params}")
        
        # Define ULTRA-MINIMAL hyperparameter search space to prevent OOM
        print("[OBJECTIVE] Creating ultra-minimal hyperparameter configuration...")
        logger.info("[OBJECTIVE] Creating ultra-minimal hyperparameter configuration...")
        
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
            
            'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.0025, log=True), # Centered around 0.001578
            'hidden_dim': trial.suggest_int('hidden_dim', 224, 288, step=32), # Centered around 256
            'num_layers': trial.suggest_int('num_layers', 4, 7), # Around 4
            'kernel_size': trial.suggest_int('kernel_size', 2, 3), # Around 2
            'dropout': trial.suggest_float('dropout', 0.4, 0.5), # Around 0.4985
            'batch_size': trial.suggest_int('batch_size', 32, 64, step=16), # Around 48
            'seq_len': trial.suggest_int('seq_len', 120, 300, step=10), # Around 110
            'prediction_horizon': 15,
            'early_stopping_patience': 5,
            
            # Focal Loss parameters (minimal ranges)
            'focal_alpha': trial.suggest_float('focal_alpha', 1.0, 1.8), # Around 1.4054
            'focal_gamma': trial.suggest_float('focal_gamma', 3.0, 4.5), # Around 3.7522
            
            'class_multiplier_0': trial.suggest_float('class_multiplier_0', 3.5, 4.5),  # Around 3.9929
            'class_multiplier_1': trial.suggest_float('class_multiplier_1', 0.7, 1.0), # Around 0.8613
            'class_multiplier_2': trial.suggest_float('class_multiplier_2', 4.0, 5.0),  # Around 4.8036  # Up class
            
            # Price threshold (fixed)
            'price_threshold': 0.005,  # Fixed 0.5% threshold for classification
            
            # Feature engineering parameters (searchable)
            'rsi_period': trial.suggest_int('rsi_period', 50, 100),
            'macd_fast_period': trial.suggest_int('macd_fast_period', 20, 50),
            'macd_slow_period': trial.suggest_int('macd_slow_period', 50, 90),
            'macd_signal_period': trial.suggest_int('macd_signal_period', 9, 30),
            'bb_period': trial.suggest_int('bb_period', 10, 20),
            'bb_num_std_dev': trial.suggest_float('bb_num_std_dev', 1.0, 1.5),
            'atr_period': trial.suggest_int('atr_period', 15, 25),
            'adx_period': trial.suggest_int('adx_period', 30, 45),
            'volume_ma_period': trial.suggest_int('volume_ma_period', 30, 45),
            'price_momentum_lookback': trial.suggest_int('price_momentum_lookback', 20, 50),
        }
        
        print(f"[OBJECTIVE] Config: {config_dict}")
        logger.info(f"[OBJECTIVE] Config: {config_dict}")
        logger.debug(f"Config: {config_dict}")
        
        # Create STGNNConfig with ULTRA-MINIMAL parameters
        print("[OBJECTIVE] Creating STGNNConfig object...")
        logger.info("[OBJECTIVE] Creating STGNNConfig object...")
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
        
        print("[OBJECTIVE] STGNNConfig created successfully")
        logger.info("[OBJECTIVE] STGNNConfig created successfully")
        print(f"[OBJECTIVE] Config details - Nodes: {config.num_nodes}, Input dim: {config.input_dim}, Hidden dim: {config.hidden_dim}")
        logger.info(f"[OBJECTIVE] Config details - Nodes: {config.num_nodes}, Input dim: {config.input_dim}, Hidden dim: {config.hidden_dim}")
        
        print("[OBJECTIVE] Determining data range...")
        logger.info("[OBJECTIVE] Determining data range...")
        
        # Dynamically determine the full data range (5 years)
        data_file_path = 'data/historical/ETH-USDT-SWAP_ohlcv_15m.csv'
        print(f"[OBJECTIVE] Reading data file: {data_file_path}")
        logger.info(f"[OBJECTIVE] Reading data file: {data_file_path}")
        
        try:
            # Read the data file to get the full date range
            print("[OBJECTIVE] Loading CSV data...")
            logger.info("[OBJECTIVE] Loading CSV data...")
            data_df = pd.read_csv(data_file_path)
            print(f"[OBJECTIVE] CSV loaded successfully - Shape: {data_df.shape}")
            logger.info(f"[OBJECTIVE] CSV loaded successfully - Shape: {data_df.shape}")
            
            print("[OBJECTIVE] Converting timestamp column...")
            logger.info("[OBJECTIVE] Converting timestamp column...")
            data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
            latest_date = data_df['timestamp'].max()
            earliest_date = data_df['timestamp'].min()
            
            # Use the full data range (5 years)
            end_time = latest_date
            start_time = end_time - timedelta(days=90)
            
            print(f"[OBJECTIVE] Data file date range: {earliest_date} to {latest_date}")
            print(f"[OBJECTIVE] Using full 5-year data range: {start_time} to {end_time}")
            print(f"[OBJECTIVE] Total data span: {(end_time - start_time).days} days")
            logger.info(f"[OBJECTIVE] Data file date range: {earliest_date} to {latest_date}")
            logger.info(f"[OBJECTIVE] Using full 5-year data range: {start_time} to {end_time}")
            logger.info(f"[OBJECTIVE] Total data span: {(end_time - start_time).days} days")
            
        except Exception as e:
            print(f"[OBJECTIVE] WARNING: Could not read data file to determine date range: {e}")
            logger.warning(f"Could not read data file to determine date range: {e}")
            # Fallback to a known good date range
            end_time = datetime(2025, 3, 15, 12, 0, 0)
            start_time = end_time - timedelta(days=1)
            print(f"[OBJECTIVE] Using fallback date range: {start_time} to {end_time}")
            logger.info(f"[OBJECTIVE] Using fallback date range: {start_time} to {end_time}")
        
        # Create data processor with memory-efficient approach
        print("[OBJECTIVE] Creating data processor...")
        logger.info("[OBJECTIVE] Creating data processor...")
        data_processor = create_classification_data_processor(config)
        print("[OBJECTIVE] Data processor created successfully")
        logger.info("[OBJECTIVE] Data processor created successfully")
        
        # Get device for training
        print("[OBJECTIVE] Getting training device...")
        logger.info("[OBJECTIVE] Getting training device...")
        device = get_device()
        print(f"[OBJECTIVE] Using device: {device}")
        logger.info(f"Using device: {device}")
        
        # Create trainer with ultra-minimal parameters
        print("[OBJECTIVE] Creating ClassificationSTGNNTrainer...")
        logger.info("[OBJECTIVE] Creating ClassificationSTGNNTrainer...")
        print(f"[OBJECTIVE] Trainer parameters - Price threshold: {config_dict['price_threshold']}, Focal alpha: {config_dict['focal_alpha']}, Focal gamma: {config_dict['focal_gamma']}")
        logger.info(f"[OBJECTIVE] Trainer parameters - Price threshold: {config_dict['price_threshold']}, Focal alpha: {config_dict['focal_alpha']}, Focal gamma: {config_dict['focal_gamma']}")
        
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
        
        print("[OBJECTIVE] ClassificationSTGNNTrainer created successfully")
        logger.info("[OBJECTIVE] ClassificationSTGNNTrainer created successfully")
        
        # Validate trainer creation
        print("[OBJECTIVE] Validating trainer creation...")
        logger.info("[OBJECTIVE] Validating trainer creation...")
        
        if trainer is None:
            print("[OBJECTIVE] ERROR: Trainer creation failed - trainer is None")
            logger.error("Trainer creation failed")
            return float('inf')
        
        print("[OBJECTIVE] Trainer creation validation passed")
        logger.info("[OBJECTIVE] Trainer creation validation passed")
            
        # Ensure model is on correct device immediately
        print("[OBJECTIVE] Checking trainer model...")
        logger.info("[OBJECTIVE] Checking trainer model...")
        
        if hasattr(trainer, 'model') and trainer.model is not None:
            print("[OBJECTIVE] Trainer model exists, moving to correct device...")
            logger.info("[OBJECTIVE] Trainer model exists, moving to correct device...")
            
            # Move trainer's model to the determined device
            print(f"[OBJECTIVE] Moving model to device: {device}")
            logger.info(f"Model moved to device: {device}")
            trainer.model.to(device)  # This ensures the model is on cuda:0

            # Re-check device to confirm (the previous check was causing the error)
            # We can simplify this check because we explicitly moved it.
            # The previous error "Failed to fix device placement: cuda:0 vs cuda"
            # was because you were checking str(model_device) != str(device)
            # where device was `torch.device('cuda')` and model_device was `torch.device('cuda:0')`.
            # By making device also `cuda:0`, this exact string comparison issue should go away.
            model_device_actual = next(trainer.model.parameters()).device
            print(f"[OBJECTIVE] Model device after move: {model_device_actual}")
            logger.info(f"[OBJECTIVE] Model device after move: {model_device_actual}")
            
            if model_device_actual != device:  # Simple object comparison is best now
                print(f"[OBJECTIVE] FATAL: Model is not on the correct device after move: {model_device_actual} vs {device}")
                logger.error(f"FATAL: Model is not on the correct device after move: {model_device_actual} vs {device}")
                return float('inf')
            
            print(f"[OBJECTIVE] Model successfully verified on device: {device}")
            logger.info(f"Model successfully verified on device: {device}")
            
            # Update the logging to use the actual model device
            print(f"[OBJECTIVE] Trainer created. Model device: {next(trainer.model.parameters()).device}")
            logger.debug(f"Trainer created. Model device: {next(trainer.model.parameters()).device}")
        else:
            print("[OBJECTIVE] ERROR: Trainer model is None")
            logger.error("Trainer model is None")
            return float('inf')
        
        # DISABLE SMOTE for hyperparameter optimization to prevent memory explosion
        # Override the train method to skip SMOTE processing
        print("[OBJECTIVE] Overriding train method to skip SMOTE...")
        logger.info("[OBJECTIVE] Overriding train method to skip SMOTE...")
        original_train = trainer.train
        print("[OBJECTIVE] Original train method saved for override")
        logger.info("[OBJECTIVE] Original train method saved for override")
        
        def train_without_smote():
            print("\n" + "-"*50)
            print("🎯 TRAINING WITHOUT SMOTE STARTING")
            print("-"*50)
            print("[TRAIN] Starting training WITHOUT SMOTE...")
            logger.info("[TRAIN] Starting training WITHOUT SMOTE...")
            
            # Load data in chunks using the data processor's memory-efficient methods
            print("[TRAIN] Loading data using data processor...")
            logger.info("[TRAIN] Loading data using data processor...")
            X, adj, y_orig_returns = data_processor.prepare_data(start_time, end_time)  # Let's rename y to y_orig_returns for clarity

            print(f"[TRAIN] Data loaded. X shape: {X.shape}, adj shape: {adj.shape}, y shape: {y_orig_returns.shape}")
            logger.info(f"[TRAIN] Data loaded. X shape: {X.shape}, adj shape: {adj.shape}, y shape: {y_orig_returns.shape}")
            logger.debug(f"[TRAIN] Data loaded. X: {X.shape}, adj: {adj.shape}, y: {y_orig_returns.shape}")

            # CRITICAL FIX: Validate data shapes and content
            print("[TRAIN] Validating data shapes and content...")
            logger.info("[TRAIN] Validating data shapes and content...")
            
            if X is None or adj is None or y_orig_returns is None:
                print("[TRAIN] ERROR: Data preparation returned None values")
                logger.error("Data preparation returned None values")
                return float('inf')
            
            # CRITICAL FIX: Validate that we have actual data
            if len(X) == 0:
                print("[TRAIN] ERROR: Data preparation returned empty X tensor")
                logger.error("Data preparation returned empty X tensor")
                return float('inf')
            if len(y_orig_returns) == 0:
                print("[TRAIN] ERROR: Data preparation returned empty y tensor")
                logger.error("Data preparation returned empty y tensor")
                return float('inf')
            
            # CRITICAL DEBUG: Check for NaN/Inf in X before any processing
            print("[TRAIN] Checking for NaN/Inf values in X tensor...")
            logger.info("[TRAIN] Checking for NaN/Inf values in X tensor...")
            
            if torch.isnan(X).any() or torch.isinf(X).any():
                print("[TRAIN] CRITICAL: NaN/Inf detected in X tensor BEFORE any processing!")
                logger.error(f"CRITICAL: NaN/Inf detected in X tensor BEFORE any processing!")
                logger.error(f"X shape: {X.shape}")
                logger.error(f"X stats: min={X.min().item()}, max={X.max().item()}, mean={X.mean().item()}")
                logger.error(f"NaN count: {torch.isnan(X).sum().item()}")
                logger.error(f"Inf count: {torch.isinf(X).sum().item()}")
                return float('inf')
            
            print("[TRAIN] X tensor validation passed - no NaN/Inf values")
            logger.info("[TRAIN] X tensor validation passed - no NaN/Inf values")
            
            # CRITICAL DEBUG: Check for NaN/Inf in adj before any processing
            print("[TRAIN] Checking for NaN/Inf values in adj tensor...")
            logger.info("[TRAIN] Checking for NaN/Inf values in adj tensor...")
            
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                print("[TRAIN] CRITICAL: NaN/Inf detected in adj tensor BEFORE any processing!")
                logger.error(f"CRITICAL: NaN/Inf detected in adj tensor BEFORE any processing!")
                logger.error(f"adj shape: {adj.shape}")
                logger.error(f"adj stats: min={adj.min().item()}, max={adj.max().item()}, mean={adj.mean().item()}")
                logger.error(f"NaN count: {torch.isnan(adj).sum().item()}")
                logger.error(f"Inf count: {torch.isinf(adj).sum().item()}")
                return float('inf')
            
            print("[TRAIN] Adj tensor validation passed - no NaN/Inf values")
            logger.info("[TRAIN] Adj tensor validation passed - no NaN/Inf values")
            
            print(f"[TRAIN] Data validation passed - X shape: {X.shape}, y shape: {y_orig_returns.shape}")
            logger.debug(f"Data validation passed - X: {X.shape}, y: {y_orig_returns.shape}")

            # Convert to classification targets using optimized price threshold
            print("[TRAIN] Converting to classification targets...")
            logger.info("[TRAIN] Converting to classification targets...")
            print(f"[TRAIN] Price threshold: {config_dict['price_threshold']}")
            logger.info(f"[TRAIN] Price threshold: {config_dict['price_threshold']}")
            
            y_flat = y_orig_returns.flatten().cpu().numpy()
            print(f"[TRAIN] Flattened y shape: {y_flat.shape}")
            logger.info(f"[TRAIN] Flattened y shape: {y_flat.shape}")
            
            classes = np.ones(len(y_flat), dtype=int)  # Default to no direction
            classes[y_flat > config_dict['price_threshold']] = 2   # Up
            classes[y_flat < -config_dict['price_threshold']] = 0  # Down

            # Log class distribution
            unique_classes, class_counts = np.unique(classes, return_counts=True)
            print(f"[TRAIN] Class distribution: {dict(zip(unique_classes, class_counts))}")
            logger.info(f"[TRAIN] Class distribution: {dict(zip(unique_classes, class_counts))}")

            y_classes = torch.LongTensor(classes.reshape(y_orig_returns.shape))
            print(f"[TRAIN] Classification targets y_classes prepared: {y_classes.shape}")
            logger.info(f"[TRAIN] Classification targets y_classes prepared: {y_classes.shape}")
            logger.debug(f"Classification targets y_classes prepared: {y_classes.shape}")

            # Split data. X_train, y_train, X_val, y_val should remain on CPU here.
            print("[TRAIN] Splitting data into train/validation sets...")
            logger.info("[TRAIN] Splitting data into train/validation sets...")
            
            X_train, y_train, X_val, y_val = data_processor.split_data(X.cpu(), y_classes.cpu())

            print(f"[TRAIN] Splitting data: X_train {X_train.shape}, X_val {X_val.shape}")
            print(f"[TRAIN] Splitting data: y_train {y_train.shape}, y_val {y_val.shape}")
            logger.info(f"[TRAIN] Splitting data: X_train {X_train.shape}, X_val {X_val.shape}")
            logger.info(f"[TRAIN] Splitting data: y_train {y_train.shape}, y_val {y_val.shape}")
            logger.debug(f"Splitting data: X_train: {X_train.shape}, X_val: {X_val.shape}")

            if len(X_val) == 0:
                print(f"[TRAIN] ERROR: Validation set is empty! X_val shape: {X_val.shape}, total X samples: {len(X)}")
                logger.error(f"Validation set is empty! X_val shape: {X_val.shape}, total X samples: {len(X)}")
                return float('inf')
            if len(X_train) == 0:
                print(f"[TRAIN] ERROR: Training set is empty! X_train shape: {X_train.shape}, total X samples: {len(X)}")
                logger.error(f"Training set is empty! X_train shape: {X_train.shape}, total X samples: {len(X)}")
                return float('inf')
            
            # Log data statistics for debugging
            logger.info(f"Data statistics - X: {X.shape}, X_train: {X_train.shape}, X_val: {X_val.shape}")
            print(f"[TRAIN] Data statistics - X: {X.shape}, X_train: {X_train.shape}, X_val: {X_val.shape}")
            logger.debug(f"Data statistics - X: {X.shape}, X_train: {X_train.shape}, X_val: {X_val.shape}")
            logger.info(f"Batch size: {config.batch_size}, Train samples: {len(X_train)}, Val samples: {len(X_val)}")
            print(f"[TRAIN] Batch size: {config.batch_size}, Train samples: {len(X_train)}, Val samples: {len(X_val)}")

            # !!! Crucial Fix: Update trainer.adj attribute !!!
            # The 'adj' here is the local variable from data_processor.prepare_data()
            # It must be moved to the correct device and assigned to the trainer's attribute.
            print("[TRAIN] Moving adjacency matrix to correct device...")
            logger.info("[TRAIN] Moving adjacency matrix to correct device...")
            trainer.adj = adj.to(device)
            logger.info(f"Adjacency matrix moved to device: {trainer.adj.device}")
            print(f"[TRAIN] Adjacency matrix moved to device: {trainer.adj.device}")

            # Calculate class weights using the prepared data to ensure consistency
            print(f"[TRAIN] Calculating class weights...")
            logger.info(f"[TRAIN] Calculating class weights...")
            class_weights = calculate_weighted_class_weights(y_orig_returns)
            print(f"[TRAIN] Class weights calculated: {class_weights}")
            logger.info(f"[TRAIN] Class weights calculated: {class_weights}")
            
            class_weights = class_weights.to(device)
            print(f"[TRAIN] Class weights moved to device: {class_weights.device}")
            logger.info(f"[TRAIN] Class weights moved to device: {class_weights.device}")
            
            # Update the trainer's criterion with the calculated weights
            print("[TRAIN] Updating trainer criterion with WeightedFocalLoss...")
            logger.info("[TRAIN] Updating trainer criterion with WeightedFocalLoss...")
            trainer.criterion = WeightedFocalLoss(
                class_weights=class_weights,
                alpha=config_dict['focal_alpha'],
                gamma=config_dict['focal_gamma']
            )
            logger.info(f"Updated criterion with class weights: {class_weights}")
            print(f"[TRAIN] Updated criterion with class weights: {class_weights}")

            # Skipping SMOTE for memory efficiency during hyperparameter optimization
            print("[TRAIN] Skipping SMOTE for memory efficiency during hyperparameter optimization")
            logger.info("Skipping SMOTE for memory efficiency during hyperparameter optimization")
            print("[TRAIN] Using original (unbalanced) training data")
            logger.info("[TRAIN] Using original (unbalanced) training data")

            # CRITICAL FIX: Validate data before creating DataLoaders
            print("[TRAIN] Validating data before creating DataLoaders...")
            logger.info("[TRAIN] Validating data before creating DataLoaders...")
            
            if len(X_train) == 0:
                print("[TRAIN] ERROR: Training set is empty! Cannot create DataLoader")
                logger.error(f"Training set is empty! Cannot create DataLoader")
                return float('inf')
            if len(X_val) == 0:
                print("[TRAIN] ERROR: Validation set is empty! Cannot create DataLoader")
                logger.error(f"Validation set is empty! Cannot create DataLoader")
                return float('inf')
            
            # Validate minimum data requirements
            min_train_samples = config.batch_size * 2  # At least 2 batches
            min_val_samples = config.batch_size  # At least 1 batch
            
            print(f"[TRAIN] Minimum requirements - Train: {min_train_samples}, Val: {min_val_samples}")
            logger.info(f"[TRAIN] Minimum requirements - Train: {min_train_samples}, Val: {min_val_samples}")
            
            if len(X_train) < min_train_samples:
                print(f"[TRAIN] ERROR: Insufficient training samples: {len(X_train)} < {min_train_samples}")
                logger.error(f"Insufficient training samples: {len(X_train)} < {min_train_samples}")
                return float('inf')
            if len(X_val) < min_val_samples:
                print(f"[TRAIN] ERROR: Insufficient validation samples: {len(X_val)} < {min_val_samples}")
                logger.error(f"Insufficient validation samples: {len(X_val)} < {min_val_samples}")
                return float('inf')
            
            # Create dataloaders with original (unbalanced) training data
            print("[TRAIN] Creating DataLoaders...")
            logger.info("[TRAIN] Creating DataLoaders...")
            
            train_loader = data_processor.create_dataloader(X_train, y_train, drop_last=True)
            val_loader = data_processor.create_dataloader(X_val, y_val, drop_last=False)

            print(f"[TRAIN] DataLoaders created successfully")
            logger.info(f"[TRAIN] DataLoaders created successfully")
            logger.debug(f"Creating DataLoaders: X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

            # CRITICAL FIX: Validate DataLoaders after creation
            print("[TRAIN] Validating DataLoaders after creation...")
            logger.info("[TRAIN] Validating DataLoaders after creation...")
            
            if len(train_loader) == 0:
                print(f"[TRAIN] ERROR: Train DataLoader is empty! Number of samples: {len(X_train)}, Batch size: {config.batch_size}")
                logger.error(f"Train DataLoader is empty! Number of samples: {len(X_train)}, Batch size: {config.batch_size}")
                logger.error(f"Train samples: {len(X_train)}, Batch size: {config.batch_size}, Drop last: True")
                return float('inf')
            if len(val_loader) == 0:
                print(f"[TRAIN] ERROR: Validation DataLoader is empty! Number of samples: {len(X_val)}, Batch size: {config.batch_size}")
                logger.error(f"Validation DataLoader is empty! Number of samples: {len(X_val)}, Batch size: {config.batch_size}")
                logger.error(f"Val samples: {len(X_val)}, Batch size: {config.batch_size}, Drop last: False")
                return float('inf')
            
            # CRITICAL DEBUG: Check first batch for NaN/Inf values
            print("[TRAIN] Checking first train batch for NaN/Inf values...")
            logger.info("[TRAIN] Checking first train batch for NaN/Inf values...")
            
            try:
                first_train_batch = next(iter(train_loader))
                X_batch, y_batch = first_train_batch
                
                print(f"[TRAIN] First batch shapes - X_batch: {X_batch.shape}, y_batch: {y_batch.shape}")
                logger.info(f"[TRAIN] First batch shapes - X_batch: {X_batch.shape}, y_batch: {y_batch.shape}")
                
                if torch.isnan(X_batch).any() or torch.isinf(X_batch).any():
                    print("[TRAIN] CRITICAL: NaN/Inf detected in first train batch!")
                    logger.error(f"CRITICAL: NaN/Inf detected in first train batch!")
                    logger.error(f"X_batch shape: {X_batch.shape}")
                    logger.error(f"X_batch stats: min={X_batch.min().item()}, max={X_batch.max().item()}, mean={X_batch.mean().item()}")
                    logger.error(f"NaN count: {torch.isnan(X_batch).sum().item()}")
                    logger.error(f"Inf count: {torch.isinf(X_batch).sum().item()}")
                    return float('inf')
                
                print(f"[TRAIN] First train batch validation passed - X_batch shape: {X_batch.shape}")
                logger.info(f"First train batch validation passed - X_batch shape: {X_batch.shape}")
                
            except Exception as e:
                print(f"[TRAIN] ERROR: Error checking first train batch: {e}")
                logger.error(f"Error checking first train batch: {e}")
                return float('inf')
            
            logger.info(f"DataLoader validation passed - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            print(f"[TRAIN] DataLoader validation passed - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            # DEBUG: Double-check DataLoader state immediately after validation
            logger.info(f"DEBUG: Immediately after validation check - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            print(f"[TRAIN] DEBUG: Immediately after validation check - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            # Training loop
            patience_counter = 0
            
            print(f"[TRAIN] Starting training loop for {trainer.config.num_epochs} epochs...")
            logger.info(f"[TRAIN] Starting training loop for {trainer.config.num_epochs} epochs...")
            logger.debug(f"Starting training loop for {trainer.config.num_epochs} epochs...")
            
            print(f"[TRAIN] Training configuration - Batch size: {config.batch_size}, Learning rate: {config.learning_rate}")
            logger.info(f"[TRAIN] Training configuration - Batch size: {config.batch_size}, Learning rate: {config.learning_rate}")

            for epoch in range(trainer.config.num_epochs):
                print(f"\n[TRAIN] Epoch {epoch+1}/{trainer.config.num_epochs}")
                logger.info(f"[TRAIN] Epoch {epoch+1}/{trainer.config.num_epochs}")
                
                # DEBUG: Check DataLoader state at start of each epoch
                print(f"[TRAIN] DEBUG: Epoch {epoch + 1} - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
                logger.info(f"DEBUG: Epoch {epoch + 1} - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
                
                # Train epoch
                print(f"[TRAIN] Starting training epoch {epoch+1}...")
                logger.info(f"[TRAIN] Starting training epoch {epoch+1}...")
                train_loss, train_acc = trainer.train_epoch(train_loader)
                
                # Check for infinite values
                print(f"[TRAIN] Epoch {epoch+1} training results - Loss: {train_loss:.6f}, Acc: {train_acc:.6f}")
                logger.info(f"[TRAIN] Epoch {epoch+1} training results - Loss: {train_loss:.6f}, Acc: {train_acc:.6f}")
                
                if np.isinf(train_loss) or np.isnan(train_loss):
                    print(f"[TRAIN] ERROR: Training loss is infinite/NaN: {train_loss}")
                    logger.error(f"Training loss is infinite/NaN: {train_loss}")
                    return float('inf')
                
                trainer.train_losses.append(train_loss)
                trainer.train_accuracies.append(train_acc)
                print(f"[TRAIN] Training metrics stored successfully")
                logger.info(f"[TRAIN] Training metrics stored successfully")
                
                # DEBUG: Check val_loader state right before validation
                print(f"[TRAIN] DEBUG: About to call validate - val_loader length: {len(val_loader)}")
                logger.info(f"DEBUG: About to call validate - val_loader length: {len(val_loader)}")
                if len(val_loader) == 0:
                    print("[TRAIN] ERROR: val_loader is empty right before validate call!")
                    logger.error("DEBUG: val_loader is empty right before validate call!")
                    logger.error("DEBUG: This suggests the DataLoader was exhausted or corrupted between validation check and actual validation")
                    return float('inf')
                
                # Validate
                print(f"[TRAIN] Starting validation epoch {epoch+1}...")
                logger.info(f"[TRAIN] Starting validation epoch {epoch+1}...")
                val_loss, val_acc = trainer.validate(val_loader)
                
                # Check for infinite values
                print(f"[TRAIN] Epoch {epoch+1} validation results - Loss: {val_loss:.6f}, Acc: {val_acc:.6f}")
                logger.info(f"[TRAIN] Epoch {epoch+1} validation results - Loss: {val_loss:.6f}, Acc: {val_acc:.6f}")
                
                if np.isinf(val_loss) or np.isnan(val_loss):
                    print(f"[TRAIN] ERROR: Validation loss is infinite/NaN: {val_loss}")
                    logger.error(f"Validation loss is infinite/NaN: {val_loss}")
                    return float('inf')
                
                trainer.val_losses.append(val_loss)
                trainer.val_accuracies.append(val_acc)
                print(f"[TRAIN] Validation metrics stored successfully")
                logger.info(f"[TRAIN] Validation metrics stored successfully")
                
                # Early stopping
                print(f"[TRAIN] Early stopping check - Current val_loss: {val_loss:.6f}, Best val_loss: {trainer.best_val_loss:.6f}")
                logger.info(f"[TRAIN] Early stopping check - Current val_loss: {val_loss:.6f}, Best val_loss: {trainer.best_val_loss:.6f}")
                
                if val_loss < trainer.best_val_loss:
                    print(f"[TRAIN] New best validation loss! Updating best model...")
                    logger.info(f"[TRAIN] New best validation loss! Updating best model...")
                    trainer.best_val_loss = val_loss
                    trainer.best_model_state = trainer.model.state_dict().copy()
                    patience_counter = 0
                    print(f"[TRAIN] Best model updated, patience counter reset to 0")
                    logger.info(f"[TRAIN] Best model updated, patience counter reset to 0")
                else:
                    patience_counter += 1
                    print(f"[TRAIN] No improvement, patience counter: {patience_counter}/{trainer.config.early_stopping_patience}")
                    logger.info(f"[TRAIN] No improvement, patience counter: {patience_counter}/{trainer.config.early_stopping_patience}")
                    
                if patience_counter >= trainer.config.early_stopping_patience:
                    print(f"[TRAIN] Early stopping triggered at epoch {epoch + 1}")
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break
                    
                if (epoch + 1) % 5 == 0:  # Log every 5 epochs for ultra-minimal training
                    print(f"\n[TRAIN] Progress Update - Epoch {epoch + 1}/{trainer.config.num_epochs}:")
                    print(f"[TRAIN] Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}")
                    print(f"[TRAIN] Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}")
                    logger.info(f'Epoch {epoch + 1}/{trainer.config.num_epochs}:')
                    logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                    logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                    
            # Load best model
            print("[TRAIN] Training loop completed, loading best model...")
            logger.info("[TRAIN] Training loop completed, loading best model...")
            
            if trainer.best_model_state is not None:
                print("[TRAIN] Loading best model state...")
                logger.info("[TRAIN] Loading best model state...")
                trainer.model.load_state_dict(trainer.best_model_state)
                print("[TRAIN] Best model state loaded successfully")
                logger.info("[TRAIN] Best model state loaded successfully")
            else:
                print("[TRAIN] WARNING: No best model state found, using current model")
                logger.warning("[TRAIN] WARNING: No best model state found, using current model")
                
            print("[TRAIN] Training complete. Loading best model state.")
            logger.info("[TRAIN] Training complete. Loading best model state.")
            logger.debug("Training complete. Loading best model state.")

            print("[TRAIN] Preparing training history for return...")
            logger.info("[TRAIN] Preparing training history for return...")
            
            training_history = {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'train_accuracies': trainer.train_accuracies,
                'val_accuracies': trainer.val_accuracies,
                'X_val': X_val,  # Store validation data for evaluation
                'y_val': y_val,   # Store validation labels for evaluation
                'smote_info': {
                    'class_distribution_before': 'Skipped for memory efficiency',
                    'class_distribution_after': 'Skipped for memory efficiency',
                    'original_batch_size': X_train.shape[0],
                    'balanced_batch_size': X_train.shape[0]  # No change since SMOTE skipped
                }
            }
            
            print(f"[TRAIN] Training history prepared - {len(trainer.train_losses)} epochs completed")
            logger.info(f"[TRAIN] Training history prepared - {len(trainer.train_losses)} epochs completed")
            
            return training_history
        
        # Replace the train method
        trainer.train = train_without_smote
        
        # Override class weights calculation with multipliers
        def calculate_weighted_class_weights(y_orig_returns):
            """Calculate class weights with trial multipliers using provided data"""
            print("[WEIGHTS] Calculating class weights using provided data...")
            logger.info("Calculating class weights using provided data...")
            
            # Convert to classes using optimized price threshold
            print(f"[WEIGHTS] Converting to classes using price threshold: {config_dict['price_threshold']}")
            logger.info(f"[WEIGHTS] Converting to classes using price threshold: {config_dict['price_threshold']}")
            
            y_flat = y_orig_returns.flatten().numpy()
            classes = np.ones(len(y_flat), dtype=int)  # Default to no direction
            classes[y_flat > config_dict['price_threshold']] = 2   # Up
            classes[y_flat < -config_dict['price_threshold']] = 0  # Down
            
            # Calculate base weights
            class_counts = Counter(classes)
            total_samples = len(classes)
            
            print(f"[WEIGHTS] Class distribution: {dict(class_counts)}")
            logger.info(f"[WEIGHTS] Class distribution: {dict(class_counts)}")
            print(f"[WEIGHTS] Total samples: {total_samples}")
            logger.info(f"[WEIGHTS] Total samples: {total_samples}")
            
            # Apply multipliers
            print("[WEIGHTS] Applying class multipliers...")
            logger.info("[WEIGHTS] Applying class multipliers...")
            
            class_weights = []
            for i in range(3):
                if class_counts[i] > 0:
                    base_weight = total_samples / (len(class_counts) * class_counts[i])
                    multiplier = config_dict[f'class_multiplier_{i}']
                    final_weight = base_weight * multiplier
                    class_weights.append(final_weight)
                    print(f"[WEIGHTS] Class {i}: base_weight={base_weight:.4f}, multiplier={multiplier:.4f}, final_weight={final_weight:.4f}")
                    logger.info(f"[WEIGHTS] Class {i}: base_weight={base_weight:.4f}, multiplier={multiplier:.4f}, final_weight={final_weight:.4f}")
                else:
                    class_weights.append(0.0)
                    print(f"[WEIGHTS] Class {i}: no samples, weight=0.0")
                    logger.info(f"[WEIGHTS] Class {i}: no samples, weight=0.0")
            
            print(f"[WEIGHTS] Final class weights: {class_weights}")
            logger.info(f"[WEIGHTS] Final class weights: {class_weights}")
            
            return torch.FloatTensor(class_weights)
        
        # Train model with production epochs for thorough optimization
        print("[OBJECTIVE] Setting up training configuration...")
        logger.info("[OBJECTIVE] Setting up training configuration...")
        
        original_epochs = trainer.config.num_epochs
        trainer.config.num_epochs = 10  # Production: 10 epochs with early stopping
        trainer.config.early_stopping_patience = 2  # Early stopping patience
        print(f"[OBJECTIVE] Training epochs: {original_epochs} -> {trainer.config.num_epochs} (production)")
        logger.info(f"[OBJECTIVE] Training epochs: {original_epochs} -> {trainer.config.num_epochs} (production)")
        print(f"[OBJECTIVE] Early stopping patience: {trainer.config.early_stopping_patience}")
        logger.info(f"[OBJECTIVE] Early stopping patience: {trainer.config.early_stopping_patience}")
        
        # Force memory cleanup before training
        print("[OBJECTIVE] Performing memory cleanup before training...")
        logger.info("[OBJECTIVE] Performing memory cleanup before training...")
        manage_memory()
        
        # Add validation to ensure training completes successfully
        print("[OBJECTIVE] Starting training execution...")
        logger.info("[OBJECTIVE] Starting training execution...")
        
        try:
            # Verify device placement before training
            print(f"[OBJECTIVE] Training on device: {device}")
            logger.info(f"Training on device: {device}")
            print(f"[OBJECTIVE] Model device: {next(trainer.model.parameters()).device}")
            logger.info(f"Model device: {next(trainer.model.parameters()).device}")
            
            # Add more detailed logging for debugging
            print(f"[OBJECTIVE] Training config: hidden_dim={config_dict['hidden_dim']}, seq_len={config_dict['seq_len']}, batch_size={config_dict['batch_size']}")
            logger.info(f"Starting training with config: hidden_dim={config_dict['hidden_dim']}, "
                       f"seq_len={config_dict['seq_len']}, batch_size={config_dict['batch_size']}")
            
            print("[OBJECTIVE] Calling trainer.train()...")
            logger.info("[OBJECTIVE] Calling trainer.train()...")
            training_history = trainer.train()
            
            # Validate training history
            print("[OBJECTIVE] Validating training results...")
            logger.info("[OBJECTIVE] Validating training results...")
            
            if training_history is None:
                print("[OBJECTIVE] ERROR: Training returned None - this indicates a critical error")
                logger.error("Training returned None - this indicates a critical error")
                return float('inf')
                
            # Check if training actually happened
            if 'train_losses' not in training_history or len(training_history['train_losses']) == 0:
                print("[OBJECTIVE] ERROR: No training losses recorded - training may have failed")
                logger.error("No training losses recorded - training may have failed")
                return float('inf')
                
            # Check for infinite or NaN losses
            train_losses = training_history.get('train_losses', [])
            if any(np.isnan(loss) or np.isinf(loss) for loss in train_losses):
                print("[OBJECTIVE] ERROR: Training produced NaN or infinite losses")
                logger.error("Training produced NaN or infinite losses")
                logger.error(f"Loss values: {train_losses}")
                return float('inf')
                
            print(f"[OBJECTIVE] Training completed successfully with {len(training_history['train_losses'])} epochs")
            logger.info(f"Training completed successfully with {len(training_history['train_losses'])} epochs")
            logger.debug(f"Training completed successfully with {len(training_history['train_losses'])} epochs")
            print(f"[OBJECTIVE] Final training loss: {train_losses[-1] if train_losses else 'N/A'}")
            logger.info(f"Final training loss: {train_losses[-1] if train_losses else 'N/A'}")
            
        except Exception as e:
            print(f"[OBJECTIVE] ERROR: Training failed with error: {e}")
            logger.error(f"Training failed with error: {e}")
            import traceback
            print(f"[OBJECTIVE] ERROR: Traceback: {traceback.format_exc()}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return float('inf')
        
        # Force memory cleanup after training
        print("[OBJECTIVE] Performing memory cleanup after training...")
        logger.info("[OBJECTIVE] Performing memory cleanup after training...")
        manage_memory()
        
        # Restore original epochs
        print(f"[OBJECTIVE] Restoring original epochs: {trainer.config.num_epochs} -> {original_epochs}")
        logger.info(f"[OBJECTIVE] Restoring original epochs: {trainer.config.num_epochs} -> {original_epochs}")
        trainer.config.num_epochs = original_epochs
        
        # Use the same validation data that was used during training
        print("[OBJECTIVE] Starting evaluation on validation data...")
        logger.info("[OBJECTIVE] Starting evaluation on validation data...")
        print("[OBJECTIVE] Evaluating on validation data using consistent dataset...")
        logger.debug("Evaluating on validation data using consistent dataset...")
        try:
            # Use the validation data that was already prepared and split during training
            # This ensures we're evaluating on the exact same data used for validation during training
            print("[OBJECTIVE] Extracting validation data from training history...")
            logger.info("[OBJECTIVE] Extracting validation data from training history...")
            
            X_val = training_history['X_val']
            y_val = training_history['y_val']
            print(f"[OBJECTIVE] Validation data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
            logger.info(f"[OBJECTIVE] Validation data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
            
            print("[OBJECTIVE] Calling trainer.evaluate()...")
            logger.info("[OBJECTIVE] Calling trainer.evaluate()...")
            evaluation_results = trainer.evaluate(X_val, y_val)
            
            # Validate evaluation results
            print("[OBJECTIVE] Validating evaluation results...")
            logger.info("[OBJECTIVE] Validating evaluation results...")
            
            if evaluation_results is None:
                print("[OBJECTIVE] ERROR: Evaluation returned None - this indicates a critical error")
                logger.error("Evaluation returned None - this indicates a critical error")
                return float('inf')
                
            # Check required keys exist
            required_keys = ['f1', 'precision', 'recall', 'probabilities', 'true_labels']
            print(f"[OBJECTIVE] Checking required evaluation keys: {required_keys}")
            logger.info(f"[OBJECTIVE] Checking required evaluation keys: {required_keys}")
            
            for key in required_keys:
                if key not in evaluation_results:
                    print(f"[OBJECTIVE] ERROR: Evaluation results missing key: {key}")
                    logger.error(f"Evaluation results missing key: {key}")
                    return float('inf')
                else:
                    print(f"[OBJECTIVE] Found evaluation key: {key}")
                    logger.debug(f"[OBJECTIVE] Found evaluation key: {key}")
                    
            print("[OBJECTIVE] Evaluation completed successfully")
            logger.info("Evaluation completed successfully")
            
        except Exception as e:
            print(f"[OBJECTIVE] ERROR: Evaluation failed with error: {e}")
            logger.error(f"Evaluation failed with error: {e}")
            import traceback
            print(f"[OBJECTIVE] ERROR: Evaluation traceback: {traceback.format_exc()}")
            logger.error(f"Evaluation traceback: {traceback.format_exc()}")
            return float('inf')
        
        # Force memory cleanup after evaluation
        print("[OBJECTIVE] Performing memory cleanup after evaluation...")
        logger.info("[OBJECTIVE] Performing memory cleanup after evaluation...")
        manage_memory()
        
        # Extract metrics for objective calculation
        print("[OBJECTIVE] Extracting metrics for objective calculation...")
        logger.info("[OBJECTIVE] Extracting metrics for objective calculation...")
        
        f1_scores = evaluation_results['f1']  # [down_f1, no_dir_f1, up_f1]
        precision_scores = evaluation_results['precision']  # [down_prec, no_dir_prec, up_prec]
        recall_scores = evaluation_results['recall']  # [down_rec, no_dir_rec, up_rec]
        
        print(f"[OBJECTIVE] Extracted metrics - F1: {f1_scores}, Precision: {precision_scores}, Recall: {recall_scores}")
        logger.info(f"[OBJECTIVE] Extracted metrics - F1: {f1_scores}, Precision: {precision_scores}, Recall: {recall_scores}")
        
        # Calculate log loss for confidence measurement
        print("[OBJECTIVE] Calculating log loss for confidence measurement...")
        logger.info("[OBJECTIVE] Calculating log loss for confidence measurement...")
        
        probabilities = evaluation_results['probabilities']
        true_labels = evaluation_results['true_labels']
        
        print(f"[OBJECTIVE] Log loss inputs - Probabilities shape: {probabilities.shape}, True labels shape: {true_labels.shape}")
        logger.info(f"[OBJECTIVE] Log loss inputs - Probabilities shape: {probabilities.shape}, True labels shape: {true_labels.shape}")
        
        # Calculate multi-class log loss
        def calculate_multiclass_log_loss(y_true, y_pred_proba):
            """Calculate multi-class log loss"""
            print("[OBJECTIVE] Calculating multi-class log loss...")
            logger.info("[OBJECTIVE] Calculating multi-class log loss...")
            
            epsilon = 1e-15  # Small value to avoid log(0)
            y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
            n_samples = y_true.shape[0]
            n_classes = y_pred_proba.shape[1]
            
            print(f"[OBJECTIVE] Log loss parameters - n_samples: {n_samples}, n_classes: {n_classes}")
            logger.info(f"[OBJECTIVE] Log loss parameters - n_samples: {n_samples}, n_classes: {n_classes}")
            
            # One-hot encode true labels
            y_true_onehot = np.zeros((n_samples, n_classes))
            y_true_onehot[np.arange(n_samples), y_true] = 1
            
            # Calculate log loss
            log_loss = -np.sum(y_true_onehot * np.log(y_pred_proba)) / n_samples
            print(f"[OBJECTIVE] Log loss calculated: {log_loss:.6f}")
            logger.info(f"[OBJECTIVE] Log loss calculated: {log_loss:.6f}")
            return log_loss
        
        log_loss_val = calculate_multiclass_log_loss(true_labels, probabilities)
        
        # Define objective to MINIMIZE: Lower is better
        print("[OBJECTIVE] Calculating objective function components...")
        logger.info("[OBJECTIVE] Calculating objective function components...")
        
        # Prioritize directional F1s more heavily for 0.5% event-based prediction
        directional_f1_avg = (f1_scores[0] + f1_scores[2]) / 2
        directional_f1_avg_penalty = (1 - directional_f1_avg) * 3.0  # Weighted more heavily
        print(f"[OBJECTIVE] Directional F1 average: {directional_f1_avg:.6f}, Penalty: {directional_f1_avg_penalty:.6f}")
        logger.info(f"[OBJECTIVE] Directional F1 average: {directional_f1_avg:.6f}, Penalty: {directional_f1_avg_penalty:.6f}")
        
        no_direction_f1_penalty = (1 - f1_scores[1]) * 1.5  # Less heavily weighted, but still important
        print(f"[OBJECTIVE] No-direction F1: {f1_scores[1]:.6f}, Penalty: {no_direction_f1_penalty:.6f}")
        logger.info(f"[OBJECTIVE] No-direction F1: {f1_scores[1]:.6f}, Penalty: {no_direction_f1_penalty:.6f}")
        
        # Add confidence penalty (log loss)
        confidence_penalty = log_loss_val * 0.2  # Adjust weight as needed
        print(f"[OBJECTIVE] Log loss: {log_loss_val:.6f}, Confidence penalty: {confidence_penalty:.6f}")
        logger.info(f"[OBJECTIVE] Log loss: {log_loss_val:.6f}, Confidence penalty: {confidence_penalty:.6f}")
        
        # Add precision penalty for directional classes (important for trading)
        directional_precision_avg = (precision_scores[0] + precision_scores[2]) / 2
        directional_precision_penalty = (1 - directional_precision_avg) * 2.0
        print(f"[OBJECTIVE] Directional precision average: {directional_precision_avg:.6f}, Penalty: {directional_precision_penalty:.6f}")
        logger.info(f"[OBJECTIVE] Directional precision average: {directional_precision_avg:.6f}, Penalty: {directional_precision_penalty:.6f}")
        
        # Combined objective
        combined_objective = (
            directional_f1_avg_penalty + 
            no_direction_f1_penalty + 
            confidence_penalty + 
            directional_precision_penalty
        )
        
        print(f"[OBJECTIVE] Combined objective: {combined_objective:.6f}")
        logger.info(f"[OBJECTIVE] Combined objective: {combined_objective:.6f}")
        
        # Handle cases where F1 scores might be NaN/inf
        if np.any(np.isnan(f1_scores)) or np.any(np.isinf(f1_scores)):
            logger.warning(f"NaN/Inf F1 scores detected: {f1_scores}")
            return float('inf')
        
        # Log trial results for monitoring
        print(f"[OBJECTIVE] Production Trial Results:")
        print(f"[OBJECTIVE]  Hidden dim: {config_dict['hidden_dim']}, Layers: {config_dict['num_layers']}")
        print(f"[OBJECTIVE]  Batch size: {config_dict['batch_size']}, Seq len: {config_dict['seq_len']}")
        print(f"[OBJECTIVE]  F1 Scores: Down={f1_scores[0]:.4f}, NoDir={f1_scores[1]:.4f}, Up={f1_scores[2]:.4f}")
        print(f"[OBJECTIVE]  Precision: Down={precision_scores[0]:.4f}, NoDir={precision_scores[1]:.4f}, Up={precision_scores[2]:.4f}")
        print(f"[OBJECTIVE]  Log Loss: {log_loss_val:.4f}")
        print(f"[OBJECTIVE]  Combined Objective: {combined_objective:.4f}")
        logger.info(f"Production Trial Results:")
        logger.info(f"  Hidden dim: {config_dict['hidden_dim']}, Layers: {config_dict['num_layers']}")
        logger.info(f"  Batch size: {config_dict['batch_size']}, Seq len: {config_dict['seq_len']}")
        logger.info(f"  Data window: 5 years, Epochs: 50 (with early stopping)")
        logger.info(f"  F1 Scores: Down={f1_scores[0]:.4f}, NoDir={f1_scores[1]:.4f}, Up={f1_scores[2]:.4f}")
        logger.info(f"  Precision: Down={precision_scores[0]:.4f}, NoDir={precision_scores[1]:.4f}, Up={precision_scores[2]:.4f}")
        logger.info(f"  Log Loss: {log_loss_val:.4f}")
        logger.info(f"  Combined Objective: {combined_objective:.4f}")
        
        # Force memory cleanup after trial
        manage_memory()
        
        # Additional cleanup for HPC environment
        del trainer, data_processor, config
        if 'X_val' in locals():
            del X_val, y_val # Removed y_val_classes
        if 'evaluation_results' in locals():
            del evaluation_results, probabilities, true_labels
        if 'f1_scores' in locals():
            del f1_scores, precision_scores, recall_scores
        if 'log_loss_val' in locals():
            del log_loss_val
        
        print("[OBJECTIVE] Trial complete. Returning objective value.")
        return combined_objective
        
    except Exception as e:
        print(f"[ERROR] Exception in trial: {e}")
        logger.error(f"[ERROR] Exception in trial: {e}")
        # Force memory cleanup on error
        manage_memory()
        return float('inf')  # Return high penalty for failed trials

def main():
    print("\n==================== HYPEROPT MAIN ====================")
    logger.info("Starting hyperparameter optimization main()...")
    # Log initial device and memory information
    device = get_device()
    logger.info(f"Starting hyperparameter optimization with device: {device}")
    
    # Log initial memory usage
    manage_memory()
    
    # Monitor memory usage throughout optimization
    def memory_monitor():
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        logger.info(f"Current CPU memory usage: {memory_gb:.2f} GB")
        
        if memory_gb > 100:  # Warning at 100GB
            logger.warning(f"CPU memory usage is high: {memory_gb:.2f} GB")
        return memory_gb
    
    # Log initial memory
    memory_monitor()
    
    # Adjust study creation for distributed/parallel execution with ultra-minimal memory optimization
    device_suffix = "cpu"
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
    print(f"[MAIN] Starting production Optuna optimization with {best_params_info} previous best parameters.")
    logger.info(f"Starting production Optuna optimization with {best_params_info} previous best parameters.")
    
    # PRODUCTION number of trials for comprehensive optimization
    study.optimize(
        objective,
        n_trials=50,  # Production: 50 trials for comprehensive search
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
    device_suffix = "cpu"
    with open(f'config/stgnn_ultra_minimal_best_params_{device_suffix}_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Print optimization summary
    device_suffix = "CPU"
    print(f'\nProduction Optimization Summary ({device_suffix}):')
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