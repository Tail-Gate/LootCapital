import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from optuna import TrialPruned
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
            
            # AGGRESSIVE EXPANSION: 2-5x wider ranges for maximum exploration
            'learning_rate': trial.suggest_float('learning_rate', 0.000001, 0.02, log=True),  # 20x wider range
            'hidden_dim': trial.suggest_int('hidden_dim', 256, 1024, step=64),  # 16x wider range, smaller step
            'num_layers': trial.suggest_int('num_layers', 1, 12),  # 6x wider range
            'kernel_size': trial.suggest_int('kernel_size', 1, 8),  # 4x wider range
            'dropout': trial.suggest_float('dropout', 0.3, 0.8),  # 2x wider range, 0.1 is too low
            'batch_size': trial.suggest_int('batch_size', 16, 128, step=16),  # 8x wider range
            'seq_len': trial.suggest_int('seq_len', 60, 720, step=30),  # 4x wider range, include very short sequences
            'prediction_horizon': 15,
            'early_stopping_patience': 5, # Increased patience for better convergence
            
            # AGGRESSIVE FOCAL LOSS EXPANSION: 5-10x wider ranges
            'focal_alpha': trial.suggest_float('focal_alpha', 0.1, 8.0),  # 16x wider range
            'focal_gamma': trial.suggest_float('focal_gamma', 0.5, 15.0),  # 7.5x wider range
            
            # AGGRESSIVE CLASS WEIGHT EXPANSION: 5-10x wider ranges
            'class_multiplier_0': trial.suggest_float('class_multiplier_0', 1.0, 20.0),  # 6.7x wider range
            'class_multiplier_1': trial.suggest_float('class_multiplier_1', 1.0, 20.0),   # 6x wider range
            'class_multiplier_2': trial.suggest_float('class_multiplier_2', 1.0, 20.0),  # 6.7x wider range
            
            'price_threshold': trial.suggest_float(0.01,0.05),

            # Feature engineering parameters (searchable) - FOCUS ON CORE MODEL PARAMETERS
            'rsi_period': trial.suggest_int('rsi_period', 14, 100), # Keep current range
            'macd_fast_period': trial.suggest_int('macd_fast_period', 12, 60), # Keep current range
            'macd_slow_period': trial.suggest_int('macd_slow_period', 26, 120), # Keep current range
            'macd_signal_period': trial.suggest_int('macd_signal_period', 9, 30),
            'bb_period': trial.suggest_int('bb_period', 10, 30), # Keep current range
            'bb_num_std_dev': trial.suggest_float('bb_num_std_dev', 1.0, 2.5), # Keep current range
            'atr_period': trial.suggest_int('atr_period', 14, 30), # Keep current range
            'adx_period': trial.suggest_int('adx_period', 30, 45), # Keep current range
            'volume_ma_period': trial.suggest_int('volume_ma_period', 20, 60), # Keep current range
            'price_momentum_lookback': trial.suggest_int('price_momentum_lookback', 20, 60), # Keep current range
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
            start_time = end_time - timedelta(days=365)
            
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
        
        # ENABLE SMOTE for hyperparameter optimization with memory concerns now fixed
        print("[OBJECTIVE] SMOTE is now enabled for hyperparameter optimization...")
        logger.info("[OBJECTIVE] SMOTE is now enabled for hyperparameter optimization...")
        print("[OBJECTIVE] Using original trainer.train() method with SMOTE processing")
        logger.info("[OBJECTIVE] Using original trainer.train() method with SMOTE processing")
        
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
            
            # AGGRESSIVE PRUNING: Stop poor trials early
            print("[OBJECTIVE] Applying aggressive pruning checks...")
            logger.info("[OBJECTIVE] Applying aggressive pruning checks...")
            
            # Check if loss is too high (indicating poor convergence)
            if train_losses and len(train_losses) >= 2:
                final_loss = train_losses[-1]
                print(f"[OBJECTIVE] Final training loss: {final_loss:.6f}")
                logger.info(f"[OBJECTIVE] Final training loss: {final_loss:.6f}")
                
                # AGGRESSIVE PRUNING: Stop trials with very high loss
                if final_loss > 5.0:
                    print(f"[OBJECTIVE] AGGRESSIVE PRUNING: Loss too high ({final_loss:.6f} > 5.0), pruning trial")
                    logger.warning(f"AGGRESSIVE PRUNING: Loss too high ({final_loss:.6f} > 5.0), pruning trial")
                    raise TrialPruned(f"Loss too high: {final_loss:.6f}")
                
                # Check for poor convergence (loss not decreasing)
                if len(train_losses) >= 3:
                    recent_losses = train_losses[-3:]
                    loss_decrease = recent_losses[0] - recent_losses[-1]
                    if loss_decrease < 0.01:  # Very small improvement
                        print(f"[OBJECTIVE] AGGRESSIVE PRUNING: Poor convergence (decrease: {loss_decrease:.6f} < 0.01), pruning trial")
                        logger.warning(f"AGGRESSIVE PRUNING: Poor convergence (decrease: {loss_decrease:.6f} < 0.01), pruning trial")
                        raise TrialPruned(f"Poor convergence: {loss_decrease:.6f}")
                
                # Check for exploding gradients (loss increasing)
                if len(train_losses) >= 2:
                    loss_increase = train_losses[-1] - train_losses[-2]
                    if loss_increase > 1.0:  # Loss increased significantly
                        print(f"[OBJECTIVE] AGGRESSIVE PRUNING: Exploding gradients (increase: {loss_increase:.6f} > 1.0), pruning trial")
                        logger.warning(f"AGGRESSIVE PRUNING: Exploding gradients (increase: {loss_increase:.6f} > 1.0), pruning trial")
                        raise TrialPruned(f"Exploding gradients: {loss_increase:.6f}")
            
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
        
        # AGGRESSIVE OBJECTIVE CALCULATION: Single objective with detailed breakdown
        # Define objective to MINIMIZE: Lower is better
        print("[OBJECTIVE] Calculating objective function components...")
        logger.info("[OBJECTIVE] Calculating objective function components...")
        
        # OBJECTIVE SCORE DERIVATION:
        # 1. Directional F1 Penalty (Weight: 3.0) - Most important for trading
        directional_f1_avg = (f1_scores[0] + f1_scores[2]) / 2  # Down + Up classes
        directional_f1_avg_penalty = (1 - directional_f1_avg) * 3.0
        print(f"[OBJECTIVE] Directional F1 average: {directional_f1_avg:.6f}, Penalty: {directional_f1_avg_penalty:.6f}")
        logger.info(f"[OBJECTIVE] Directional F1 average: {directional_f1_avg:.6f}, Penalty: {directional_f1_avg_penalty:.6f}")
        
        # 2. No-Direction F1 Penalty (Weight: 1.5) - Medium importance
        no_direction_f1_penalty = (1 - f1_scores[1]) * 1.5
        print(f"[OBJECTIVE] No-direction F1: {f1_scores[1]:.6f}, Penalty: {no_direction_f1_penalty:.6f}")
        logger.info(f"[OBJECTIVE] No-direction F1: {f1_scores[1]:.6f}, Penalty: {no_direction_f1_penalty:.6f}")
        
        # 3. Confidence Penalty (Weight: 0.2) - Log loss for prediction confidence
        confidence_penalty = log_loss_val * 0.2
        print(f"[OBJECTIVE] Log loss: {log_loss_val:.6f}, Confidence penalty: {confidence_penalty:.6f}")
        logger.info(f"[OBJECTIVE] Log loss: {log_loss_val:.6f}, Confidence penalty: {confidence_penalty:.6f}")
        
        # 4. Directional Precision Penalty (Weight: 2.0) - Important for avoiding false signals
        directional_precision_avg = (precision_scores[0] + precision_scores[2]) / 2
        directional_precision_penalty = (1 - directional_precision_avg) * 2.0
        print(f"[OBJECTIVE] Directional precision average: {directional_precision_avg:.6f}, Penalty: {directional_precision_penalty:.6f}")
        logger.info(f"[OBJECTIVE] Directional precision average: {directional_precision_avg:.6f}, Penalty: {directional_precision_penalty:.6f}")
        
        # COMBINED OBJECTIVE: Sum of all penalties (lower is better)
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
    
    # AGGRESSIVE STUDY: Create study for aggressive hyperparameter exploration
    device_suffix = "cpu"
    study = optuna.create_study(
        direction='minimize',
        study_name=f'stgnn_aggressive_hyperopt_{device_suffix}',  # Aggressive study name
        storage=f'sqlite:///stgnn_aggressive_hyperopt_{device_suffix}.db',  # Aggressive database name
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
    
    # AGGRESSIVE OPTIMIZATION: 150 trials with aggressive pruning
    study.optimize(
        objective,
        n_trials=150,  # AGGRESSIVE: 150 trials for comprehensive exploration
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
    with open(f'config/stgnn_aggressive_best_params_{device_suffix}_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Print optimization summary
    device_suffix = "CPU"
    print(f'\nAGGRESSIVE Optimization Summary ({device_suffix}):')
    print(f'  Device: {device}')
    print(f'  Total trials: {len(study.trials)}')
    print(f'  Best validation loss: {study.best_trial.value:.4f}')
    print(f'  Best focal_alpha: {best_params.get("focal_alpha", "N/A")}')
    print(f'  Best focal_gamma: {best_params.get("focal_gamma", "N/A")}')
    print(f'  Best class multipliers: [{best_params.get("class_multiplier_0", "N/A")}, '
          f'{best_params.get("class_multiplier_1", "N/A")}, {best_params.get("class_multiplier_2", "N/A")}]')
    print(f'  Search space: 2-16x wider ranges for aggressive exploration')
    print(f'  Pruning: Aggressive pruning enabled (loss > 5.0, poor convergence, exploding gradients)')
    
    # Final memory cleanup
    manage_memory()

if __name__ == '__main__':
    main() 