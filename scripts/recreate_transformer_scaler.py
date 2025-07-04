import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.feature_generator import FeatureGenerator

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_eth_data():
    """Load ETH OHLCV data"""
    data_dir = project_root / "data" / "historical"
    # Load 15-minute ETH-USDT data
    data = pd.read_csv(data_dir / "ETH-USDT-SWAP_ohlcv_15m.csv", index_col=0, parse_dates=True)
    return data

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

def scale_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler"""
    logging.info("\nScaling features...")
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )
    return scaled_features, scaler

def main():
    setup_logging()
    
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
    
    # Clean features
    features = clean_features(features)
    logging.info("Features cleaned")
    
    # Scale features and get scaler
    scaled_features, scaler = scale_features(features)
    logging.info("Features scaled")
    
    # Save the scaler
    scaler_path = project_root / "models" / "transformer_feature_scaler.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"\nScaler saved to: {scaler_path}")
    
    # Save features for reference
    features_path = project_root / "models" / "transformer_features.parquet"
    scaled_features.to_parquet(features_path)
    logging.info(f"Features saved to: {features_path}")

if __name__ == "__main__":
    main() 