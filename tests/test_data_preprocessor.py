import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from utils.data_preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        'numeric_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'numeric_2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
        'categorical': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'missing': [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan],
        'outliers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
    })
    return data

@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance for testing."""
    return DataPreprocessor()

def test_clean_data(preprocessor, sample_data):
    """Test data cleaning functionality."""
    # Test duplicate handling
    data_with_dupes = pd.concat([sample_data, sample_data.iloc[:2]])
    cleaned = preprocessor.clean_data(data_with_dupes, drop_duplicates=True)
    assert len(cleaned) == len(sample_data)
    
    # Test outlier handling
    cleaned = preprocessor.clean_data(sample_data, handle_outliers=True)
    assert cleaned['outliers'].max() < 100  # Outlier should be clipped
    
    # Test without any cleaning
    cleaned = preprocessor.clean_data(sample_data, drop_duplicates=False, handle_outliers=False)
    assert cleaned.equals(sample_data)

def test_scale_features(preprocessor, sample_data):
    """Test feature scaling functionality."""
    # Test standard scaling
    scaled = preprocessor.scale_features(sample_data, method="standard")
    assert scaled['numeric_1'].mean() == pytest.approx(0, abs=1e-10)
    assert scaled['numeric_1'].std() == pytest.approx(1, abs=1e-10)
    
    # Test robust scaling
    scaled = preprocessor.scale_features(sample_data, method="robust")
    assert scaled['numeric_1'].median() == pytest.approx(0, abs=1e-10)
    
    # Test scaling specific features
    scaled = preprocessor.scale_features(sample_data, features=['numeric_1'])
    assert 'numeric_2' in scaled.columns
    assert not np.allclose(scaled['numeric_2'], sample_data['numeric_2'])

def test_handle_missing_values(preprocessor, sample_data):
    """Test missing value handling."""
    # Test mean imputation
    imputed = preprocessor.handle_missing_values(sample_data, strategy="mean")
    assert not imputed['missing'].isnull().any()
    assert imputed['missing'].mean() == sample_data['missing'].mean()
    
    # Test median imputation
    imputed = preprocessor.handle_missing_values(sample_data, strategy="median")
    assert not imputed['missing'].isnull().any()
    assert imputed['missing'].median() == sample_data['missing'].median()
    
    # Test most frequent imputation
    imputed = preprocessor.handle_missing_values(sample_data, strategy="most_frequent")
    assert not imputed['missing'].isnull().any()

def test_compute_feature_stats(preprocessor, sample_data):
    """Test feature statistics computation."""
    stats = preprocessor.compute_feature_stats(sample_data)
    
    # Check numeric feature stats
    assert 'mean' in stats['numeric_1']
    assert 'std' in stats['numeric_1']
    assert 'min' in stats['numeric_1']
    assert 'max' in stats['numeric_1']
    
    # Check categorical feature stats
    assert 'unique' in stats['categorical']
    assert 'most_common' in stats['categorical']
    
    # Verify values
    assert stats['numeric_1']['mean'] == sample_data['numeric_1'].mean()
    assert stats['numeric_1']['std'] == sample_data['numeric_1'].std()

def test_save_load_state(preprocessor, sample_data):
    """Test saving and loading preprocessor state."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process some data
        preprocessor.scale_features(sample_data)
        preprocessor.handle_missing_values(sample_data)
        preprocessor.compute_feature_stats(sample_data)
        
        # Save state
        preprocessor.save_state(temp_dir)
        
        # Create new preprocessor and load state
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_state(temp_dir)
        
        # Verify scalers and imputers were loaded
        assert len(new_preprocessor.scalers) == len(preprocessor.scalers)
        assert len(new_preprocessor.imputers) == len(preprocessor.imputers)
        
        # Verify feature stats were loaded
        assert new_preprocessor.feature_stats == preprocessor.feature_stats

def test_validate_data(preprocessor, sample_data):
    """Test data validation functionality."""
    # Test with valid data
    is_valid, errors = preprocessor.validate_data(
        sample_data,
        required_features=['numeric_1', 'numeric_2'],
        feature_types={'numeric_1': 'numeric', 'categorical': 'categorical'}
    )
    assert is_valid
    assert len(errors) == 0
    
    # Test with missing required feature
    is_valid, errors = preprocessor.validate_data(
        sample_data,
        required_features=['missing_feature']
    )
    assert not is_valid
    assert any('missing_feature' in error for error in errors)
    
    # Test with incorrect feature type
    is_valid, errors = preprocessor.validate_data(
        sample_data,
        feature_types={'numeric_1': 'categorical'}
    )
    assert not is_valid
    assert any('should be categorical' in error for error in errors)
    
    # Test with missing values
    is_valid, errors = preprocessor.validate_data(sample_data)
    assert not is_valid
    assert any('missing values' in error for error in errors) 