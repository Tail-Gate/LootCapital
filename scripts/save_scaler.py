#!/usr/bin/env python3
"""
Save Scaler Script

This script saves the scaler exactly like the walk_forward_optimization.py file does,
using the specific feature list provided.
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
import joblib
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from utils.feature_generator import FeatureGenerator
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/save_scaler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        
        # CRITICAL: Filter to EXACTLY the 25 features specified BEFORE any processing
        logger.info(f"Original features generated: {len(features.columns)}")
        logger.info(f"Original feature columns: {list(features.columns)}")
        
        # Ensure all required features from config are present
        for feat in self.config.features:
            if feat not in features.columns:
                logger.warning(f"Feature '{feat}' not found in FeatureGenerator output, using 0")
                features[feat] = 0
        
        # Select ONLY the features specified in config (exactly 25 features)
        features = features[self.config.features]
        
        logger.info(f"After filtering to specified features: {len(features.columns)}")
        logger.info(f"Filtered feature columns: {list(features.columns)}")
        
        # Handle missing/infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill()
        features = features.fillna(0)
        
        logger.info(f"Final features after cleanup: {len(features.columns)} features")
        logger.info(f"Final feature columns: {list(features.columns)}")
        
        return features
        
    def set_scaler(self, scaler_type: str = 'minmax'):
        """Set scaler type for feature normalization"""
        super().set_scaler(scaler_type)
        logger.info(f"Enhanced data processor scaler set to: {scaler_type}")
        
    def fit_scaler(self, features: pd.DataFrame):
        """Fit scaler on training data - ensure only 25 features"""
        # Ensure we only fit on the exact 25 features specified
        if len(features.columns) != 25:
            logger.warning(f"Fitting scaler on {len(features.columns)} features, expected 25")
            logger.warning(f"Feature columns: {list(features.columns)}")
        
        # Filter to only the 25 specified features if needed
        if set(features.columns) != set(self.config.features):
            logger.warning("Feature columns don't match config, filtering...")
            features = features[self.config.features]
            logger.info(f"Filtered to {len(features.columns)} features: {list(features.columns)}")
        
        super().fit_scaler(features)
        logger.info(f"Enhanced data processor scaler fitted successfully on {len(features.columns)} features")
        
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler - ensure only 25 features"""
        # Ensure we only transform the exact 25 features specified
        if len(features.columns) != 25:
            logger.warning(f"Transforming {len(features.columns)} features, expected 25")
            logger.warning(f"Feature columns: {list(features.columns)}")
        
        # Filter to only the 25 specified features if needed
        if set(features.columns) != set(self.config.features):
            logger.warning("Feature columns don't match config during transform, filtering...")
            features = features[self.config.features]
            logger.info(f"Filtered to {len(features.columns)} features for transform: {list(features.columns)}")
        
        return super().transform_features(features)

def create_config():
    """Create configuration with the specific feature list"""
    
    # Focus on ETH/USD
    assets = ['ETH/USD']
    
    # Specific feature list provided by user
    features = [
        "returns",
        "log_returns",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "volume_ma",
        "volume_std",
        "volume_surge",
        "volume_ratio",
        "price_momentum",
        "volatility_regime",
        "support",
        "resistance",
        "vwap_ratio",
        "ma_crossover",
        "swing_rsi",
        "breakout_intensity",
        "adx",
        "cumulative_delta",
        "atr"
    ]
    
    # Create configuration
    config = STGNNConfig(
        num_nodes=len(assets),
        input_dim=len(features),
        hidden_dim=352,  # Optimized value
        output_dim=3,  # 3 classes: down/no direction/up
        num_layers=7,  # Optimized value
        dropout=0.5505394438393119,  # Optimized value
        kernel_size=2,  # Optimized value
        learning_rate=0.002096519264498227,  # Optimized value
        batch_size=64,  # Optimized value
        num_epochs=40,  # Keep fixed for walk-forward
        early_stopping_patience=8,
        seq_len=70,  # Optimized value
        prediction_horizon=15,
        features=features,
        assets=assets,
        confidence_threshold=0.51,
        buy_threshold=0.6,
        sell_threshold=0.4,
        retrain_interval=24,
        focal_alpha=1.59329918873379,  # Optimized value
        focal_gamma=3.6839150893608212,  # Optimized value
        # Optimized feature engineering parameters
        rsi_period=21,  # Optimized value
        macd_fast_period=14,  # Optimized value
        macd_slow_period=31,  # Optimized value
        macd_signal_period=17,  # Optimized value
        bb_period=14,  # Optimized value
        bb_num_std_dev=1.3402253735507053,  # Optimized value
        atr_period=20,  # Optimized value
        adx_period=35,  # Optimized value
        volume_ma_period=39,  # Optimized value
        price_momentum_lookback=42,  # Optimized value
        price_threshold=0.02  # 2% threshold
    )
    
    return config

def save_scaler():
    """Save the scaler exactly like walk_forward_optimization.py does"""
    
    logger.info("Starting scaler save process...")
    start_time = time.time()
    
    try:
        # Create configuration
        logger.info("Creating configuration...")
        config = create_config()
        logger.info(f"Configuration created successfully")
        logger.info(f"Assets: {config.assets}")
        logger.info(f"Features: {len(config.features)} features")
        logger.info(f"Feature list: {config.features}")
        
        # Initialize components
        logger.info("Initializing components...")
        market_data = MarketData(data_source_path='data')  # Use local data directory
        technical_indicators = TechnicalIndicators()
        
        # Create enhanced data processor
        logger.info("Creating enhanced data processor...")
        data_processor = EnhancedSTGNNDataProcessor(config, market_data, technical_indicators)
        
        # Set scaler type
        logger.info("Setting scaler type to minmax...")
        data_processor.set_scaler('minmax')
        
        # Prepare data to fit the scaler
        logger.info("Preparing data to fit scaler...")
        X, adj, y = data_processor.prepare_data()
        logger.info(f"Data prepared successfully - X: {X.shape}, adj: {adj.shape}, y: {y.shape}")
        
        # The scaler is already fitted during prepare_data() in the data processor
        logger.info("Scaler fitted successfully during data preparation")
        
        # Create output directory
        output_dir = Path('models')
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")
        
        # Save fitted scaler exactly like walk_forward_optimization.py
        logger.info("Saving fitted scaler...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaler_path = output_dir / f'stgnn_scaler_{timestamp}.joblib'
        logger.info(f"Scaler save path: {scaler_path}")
        logger.info(f"Scaler type: {type(data_processor.scaler)}")
        logger.info(f"Scaler attributes: {dir(data_processor.scaler)}")
        
        joblib.dump(data_processor.scaler, scaler_path)
        logger.info(f"Scaler save completed")
        
        # Verify file was created
        if scaler_path.exists():
            file_size = scaler_path.stat().st_size
            logger.info(f"Scaler file created successfully - Size: {file_size} bytes")
        else:
            logger.error(f"Scaler file was NOT created!")
            raise Exception("Scaler file was not created")
        
        # Save feature list (order is critical for inference)
        logger.info("Saving feature list...")
        features_path = output_dir / f'stgnn_features_{timestamp}.json'
        logger.info(f"Features save path: {features_path}")
        logger.info(f"Features list: {config.features}")
        logger.info(f"Features count: {len(config.features)}")
        
        with open(features_path, 'w') as f:
            json.dump(config.features, f, indent=4)
        logger.info(f"Features save completed")
        
        # Verify file was created
        if features_path.exists():
            file_size = features_path.stat().st_size
            logger.info(f"Features file created successfully - Size: {file_size} bytes")
            
            # Verify content
            with open(features_path, 'r') as f:
                loaded_features = json.load(f)
            logger.info(f"Loaded features count: {len(loaded_features)}")
            logger.info(f"Features match: {loaded_features == config.features}")
        else:
            logger.error(f"Features file was NOT created!")
            raise Exception("Features file was not created")
        
        # Save inference metadata
        logger.info("Saving inference metadata...")
        inference_metadata = {
            'scaler_path': str(scaler_path),
            'features_path': str(features_path),
            'config': {
                'num_nodes': config.num_nodes,
                'input_dim': config.input_dim,
                'seq_len': config.seq_len,
                'output_dim': config.output_dim
            },
            'timestamp': timestamp
        }
        
        logger.info(f"Metadata keys: {list(inference_metadata.keys())}")
        
        metadata_path = output_dir / f'stgnn_inference_metadata_{timestamp}.json'
        logger.info(f"Metadata save path: {metadata_path}")
        
        with open(metadata_path, 'w') as f:
            json.dump(inference_metadata, f, indent=4)
        logger.info(f"Metadata save completed")
        
        # Verify file was created
        if metadata_path.exists():
            file_size = metadata_path.stat().st_size
            logger.info(f"Metadata file created successfully - Size: {file_size} bytes")
            
            # Verify content
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)
            logger.info(f"Loaded metadata keys: {list(loaded_metadata.keys())}")
        else:
            logger.error(f"Metadata file was NOT created!")
            raise Exception("Metadata file was not created")
        
        # Final verification - list all created files
        logger.info("Final file verification:")
        expected_files = [
            scaler_path,
            features_path,
            metadata_path
        ]
        
        for file_path in expected_files:
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.info(f"✓ {file_path.name} - {file_size} bytes")
            else:
                logger.error(f"✗ {file_path.name} - NOT FOUND!")
                raise Exception(f"Expected file {file_path.name} was not created")
        
        # List all files in output directory
        logger.info("All files in output directory:")
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                file_size = file_path.stat().st_size
                logger.info(f"  {file_path.name} - {file_size} bytes")
        
        total_time = time.time() - start_time
        logger.info(f"Scaler save process completed successfully in {total_time:.2f} seconds")
        
        return {
            'scaler_path': str(scaler_path),
            'features_path': str(features_path),
            'metadata_path': str(metadata_path),
            'config': config
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Scaler save process failed after {total_time:.2f} seconds")
        logger.error(f"Error: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    """Main function"""
    logger.info("="*80)
    logger.info("STARTING SCALER SAVE PROCESS")
    logger.info("="*80)
    
    try:
        results = save_scaler()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SCALER SAVE PROCESS COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"Files created:")
        logger.info(f"  Scaler: {results['scaler_path']}")
        logger.info(f"  Features: {results['features_path']}")
        logger.info(f"  Metadata: {results['metadata_path']}")
        
        return results
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"SCALER SAVE PROCESS FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 