import pandas as pd
import numpy as np
import pytest
from utils.class_balancing import ClassBalancer

def test_class_balancer_initialization():
    """Test ClassBalancer initialization with different methods."""
    # Test valid methods
    for method in ['smote', 'undersample', 'oversample', 'none']:
        balancer = ClassBalancer(method=method)
        assert balancer.method == method
    
    # Test invalid method
    with pytest.raises(ValueError):
        ClassBalancer(method='invalid_method')

def test_analyze_distribution():
    """Test label distribution analysis."""
    # Create test data
    labels = pd.Series([0, 0, 0, 1, 1, 2])
    balancer = ClassBalancer()
    
    # Get distribution stats
    stats = balancer.analyze_distribution(labels)
    
    # Verify basic statistics
    assert stats['total_samples'] == 6
    assert stats['counts'] == {0: 3, 1: 2, 2: 1}
    assert stats['imbalance_ratio'] == 3.0  # 3/1
    
    # Test binary classification metrics
    binary_labels = pd.Series([0, 0, 0, 1, 1])
    binary_stats = balancer.analyze_distribution(binary_labels)
    assert binary_stats['minority_class'] == 1
    assert binary_stats['majority_class'] == 0
    assert binary_stats['minority_ratio'] == 0.4  # 2/5

def test_balance_smote():
    """Test SMOTE balancing."""
    # Create imbalanced dataset with enough samples for SMOTE (at least 6 per class)
    X = pd.DataFrame({
        'feature1': list(range(1, 26)),  # 25 samples total
        'feature2': list(range(10, 260, 10))
    })
    y = pd.Series([0]*19 + [1]*6)  # 19 samples of class 0, 6 samples of class 1
    
    balancer = ClassBalancer(method='smote')
    X_balanced, y_balanced = balancer.balance(X, y)
    
    # Verify balancing
    assert len(X_balanced) > len(X)  # SMOTE should add samples
    assert len(np.unique(y_balanced)) == 2  # Should maintain both classes
    assert X_balanced.shape[1] == X.shape[1]  # Should maintain feature count
    
    # Verify that minority class was oversampled
    class_counts = pd.Series(y_balanced).value_counts()
    assert abs(class_counts[0] - class_counts[1]) <= 1  # Classes should be roughly balanced

def test_balance_undersample():
    """Test undersampling."""
    # Create imbalanced dataset
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [10, 20, 30, 40, 50, 60]
    })
    y = pd.Series([0, 0, 0, 0, 1, 1])
    
    balancer = ClassBalancer(method='undersample')
    X_balanced, y_balanced = balancer.balance(X, y)
    
    # Verify balancing
    assert len(X_balanced) < len(X)  # Undersampling should reduce samples
    assert len(np.unique(y_balanced)) == 2  # Should maintain both classes
    assert X_balanced.shape[1] == X.shape[1]  # Should maintain feature count

def test_balance_oversample():
    """Test oversampling."""
    # Create imbalanced dataset
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [10, 20, 30, 40, 50, 60]
    })
    y = pd.Series([0, 0, 0, 0, 1, 1])
    
    balancer = ClassBalancer(method='oversample')
    X_balanced, y_balanced = balancer.balance(X, y)
    
    # Verify balancing
    assert len(X_balanced) > len(X)  # Oversampling should add samples
    assert len(np.unique(y_balanced)) == 2  # Should maintain both classes
    assert X_balanced.shape[1] == X.shape[1]  # Should maintain feature count

def test_get_balancing_report():
    """Test balancing report generation."""
    # Create test data with imbalanced classes
    X = pd.DataFrame({
        'feature1': list(range(1, 26)),  # 25 samples total
        'feature2': list(range(10, 260, 10))
    })
    y = pd.Series([0]*19 + [1]*6)  # 19 samples of class 0, 6 samples of class 1
    
    balancer = ClassBalancer(method='smote')
    X_balanced, y_balanced = balancer.balance(X, y)
    
    # Get report
    report = balancer.get_balancing_report(X, y, X_balanced, y_balanced)
    
    # Verify report contents
    assert 'original_distribution' in report
    assert 'balanced_distribution' in report
    assert 'balancing_method' in report
    assert 'feature_count' in report
    assert 'original_sample_count' in report
    assert 'balanced_sample_count' in report
    assert 'samples_added' in report
    
    # Verify specific values
    assert report['feature_count'] == 2
    assert report['original_sample_count'] == 25
    assert report['samples_added'] > 0  # SMOTE should add samples
    assert report['balanced_sample_count'] > report['original_sample_count'] 