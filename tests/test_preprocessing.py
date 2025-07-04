import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.synthetic_data_generator import SyntheticDataGenerator
from utils.data_preprocessor import DataPreprocessor
from utils.feature_generator import FeatureGenerator
import logging
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_preprocessing_pipeline():
    """Test the complete data preprocessing pipeline."""
    
    # 1. Generate synthetic data
    logger.info("Generating synthetic data...")
    data_generator = SyntheticDataGenerator(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 7),  # 7 days of data
        base_price=100.0,
        volatility=0.02,
        trend_strength=0.1,
        volume_base=1000.0,
        seed=42
    )
    
    # Generate complete dataset
    ohlcv_data, order_book_data, features, target = data_generator.generate_dataset(
        interval_minutes=15,
        n_levels=5,
        lookahead=5,
        threshold=0.001
    )
    
    # Drop initial rows with NaN values from rolling calculations
    features = features.dropna()
    target = target[features.index]
    
    # 2. Initialize preprocessor
    logger.info("Initializing data preprocessor...")
    preprocessor = DataPreprocessor()
    
    # 3. Test data cleaning
    logger.info("Testing data cleaning...")
    cleaned_data = preprocessor.clean_data(
        features,
        drop_duplicates=True,
        handle_outliers=True,
        outlier_threshold=2.0
    )
    
    # Verify cleaning results
    assert len(cleaned_data) == len(features), "Data cleaning should not remove valid rows"
    assert not cleaned_data.isnull().any().any(), "Cleaned data should not contain NaN values"
    
    # 4. Test feature scaling
    logger.info("Testing feature scaling...")
    scaled_data = preprocessor.scale_features(
        cleaned_data,
        method="standard",
        fit=True
    )
    
    # Verify scaling results
    numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert abs(scaled_data[col].mean()) < 1e-10, f"Column {col} should have zero mean"
        assert abs(scaled_data[col].std() - 1.0) < 1e-10, f"Column {col} should have unit variance"
    
    # 5. Test missing value handling
    logger.info("Testing missing value handling...")
    # Introduce some missing values
    data_with_missing = scaled_data.copy()
    data_with_missing.iloc[0:5, 0] = np.nan
    
    processed_data = preprocessor.handle_missing_values(
        data_with_missing,
        strategy="mean",
        fit=True
    )
    
    # Verify missing value handling
    assert not processed_data.isnull().any().any(), "Processed data should not contain NaN values"
    
    # 6. Test feature generation
    logger.info("Testing feature generation...")
    feature_generator = FeatureGenerator()
    
    # Generate technical features
    technical_features = feature_generator.generate_technical_features(
        ohlcv_data,
        price_col='close',
        volume_col='volume'
    )
    
    # Generate momentum features
    momentum_features = feature_generator.generate_momentum_features(
        ohlcv_data,
        price_col='close',
        volume_col='volume'
    )
    
    # Generate order book features
    order_book_features = feature_generator.generate_order_book_features(
        ohlcv_data,
        order_book_data
    )
    
    # Verify feature generation
    assert not technical_features.empty, "Technical features should not be empty"
    assert not momentum_features.empty, "Momentum features should not be empty"
    assert not order_book_features.empty, "Order book features should not be empty"
    
    # 7. Test data validation
    logger.info("Testing data validation...")
    is_valid, errors = preprocessor.validate_data(
        processed_data,
        required_features=['close', 'volume'],
        feature_types={'close': 'numeric', 'volume': 'numeric'},
        check_missing=True
    )
    
    assert is_valid, f"Data validation failed: {errors}"
    
    # 8. Test state saving and loading
    logger.info("Testing state saving and loading...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save state
        preprocessor.save_state(temp_dir)
        
        # Create new preprocessor and load state
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_state(temp_dir)
        
        # Verify state was loaded correctly
        assert len(new_preprocessor.scalers) == len(preprocessor.scalers)
        assert len(new_preprocessor.imputers) == len(preprocessor.imputers)
    
    logger.info("All preprocessing pipeline tests completed successfully!")

if __name__ == "__main__":
    test_preprocessing_pipeline() 