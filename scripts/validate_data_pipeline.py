import pandas as pd
import numpy as np
import logging
import os
from utils.data_preprocessor import DataPreprocessor
from utils.feature_generator import FeatureGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load real OHLCV data
    data_path = "data/historical/ETH-USDT-SWAP_ohlcv_15m.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    logger.info(f"Loading data from {data_path}")
    ohlcv_data = pd.read_csv(data_path, parse_dates=[0])
    logger.info(f"Loaded {len(ohlcv_data)} rows.")

    # 2. Initialize preprocessor
    preprocessor = DataPreprocessor()

    # 3. Clean data
    logger.info("Cleaning data...")
    cleaned_data = preprocessor.clean_data(
        ohlcv_data,
        drop_duplicates=True,
        handle_outliers=True,
        outlier_threshold=2.0
    )
    logger.info(f"Cleaned data shape: {cleaned_data.shape}")
    logger.info(f"Any NaNs after cleaning? {cleaned_data.isnull().any().any()}")

    # 4. Scale features
    logger.info("Scaling features...")
    scaled_data = preprocessor.scale_features(
        cleaned_data,
        method="standard",
        fit=True
    )
    logger.info(f"Scaled data shape: {scaled_data.shape}")

    # 5. Handle missing values
    logger.info("Handling missing values...")
    processed_data = preprocessor.handle_missing_values(
        scaled_data,
        strategy="mean",
        fit=True
    )
    logger.info(f"Any NaNs after missing value handling? {processed_data.isnull().any().any()}")

    # 6. Feature generation
    logger.info("Generating features...")
    feature_generator = FeatureGenerator()
    technical_features = feature_generator.generate_technical_features(
        processed_data,
        price_col='close',
        volume_col='volume'
    )
    logger.info(f"Technical features shape: {technical_features.shape}")

    # 7. Data validation
    logger.info("Validating data...")
    is_valid, errors = preprocessor.validate_data(
        processed_data,
        required_features=['close', 'volume'],
        feature_types={'close': 'numeric', 'volume': 'numeric'},
        check_missing=True
    )
    if is_valid:
        logger.info("Data validation passed!")
    else:
        logger.error(f"Data validation failed: {errors}")

if __name__ == "__main__":
    main() 