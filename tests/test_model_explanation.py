import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from utils.model_explanation import ModelExplainer
from utils.model_trainer import ModelTrainer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create features
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000)
    })
    
    # Create target (binary classification)
    y = pd.Series(np.random.randint(0, 2, 1000))
    
    return X, y

@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    X, y = sample_data
    trainer = ModelTrainer(model_type="xgboost")
    trainer.train(X, y)
    return trainer.model

@pytest.fixture
def model_explainer(trained_model):
    """Create a ModelExplainer instance for testing."""
    return ModelExplainer(
        model=trained_model,
        feature_names=['feature1', 'feature2', 'feature3']
    )

def test_fit_explainer(model_explainer, sample_data):
    """Test fitting the explainer."""
    X, _ = sample_data
    model_explainer.fit_explainer(X)
    assert model_explainer.explainer is not None

def test_get_shap_values(model_explainer, sample_data):
    """Test getting SHAP values."""
    X, _ = sample_data
    model_explainer.fit_explainer(X)
    shap_values = model_explainer.get_shap_values(X)
    assert isinstance(shap_values, np.ndarray)
    assert shap_values.shape[0] == len(X)
    assert shap_values.shape[1] == len(X.columns)

def test_plot_summary(model_explainer, sample_data):
    """Test plotting summary."""
    X, _ = sample_data
    model_explainer.fit_explainer(X)
    
    # Test with output path
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "summary.png"
        model_explainer.plot_summary(X, str(output_path))
        assert output_path.exists()
    
    # Test without output path (should not raise error)
    model_explainer.plot_summary(X)

def test_plot_dependence(model_explainer, sample_data):
    """Test plotting dependence."""
    X, _ = sample_data
    model_explainer.fit_explainer(X)
    
    # Test with output path
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "dependence.png"
        model_explainer.plot_dependence(X, "feature1", str(output_path))
        assert output_path.exists()
    
    # Test without output path (should not raise error)
    model_explainer.plot_dependence(X, "feature1")

def test_get_feature_importance(model_explainer, sample_data):
    """Test getting feature importance."""
    X, _ = sample_data
    model_explainer.fit_explainer(X)
    importance = model_explainer.get_feature_importance(X)
    
    assert isinstance(importance, dict)
    assert len(importance) == len(X.columns)
    assert all(isinstance(v, float) for v in importance.values())
    assert all(v >= 0 for v in importance.values())

def test_explain_prediction(model_explainer, sample_data):
    """Test explaining a single prediction."""
    X, _ = sample_data
    model_explainer.fit_explainer(X)
    
    # Test with output path
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "prediction.png"
        model_explainer.explain_prediction(X, 0, str(output_path))
        assert output_path.exists()
    
    # Test without output path (should not raise error)
    model_explainer.explain_prediction(X, 0)

def test_error_handling(model_explainer, sample_data):
    """Test error handling."""
    X, _ = sample_data
    
    # Test getting SHAP values before fitting
    with pytest.raises(ValueError):
        model_explainer.get_shap_values(X)
    
    # Test plotting before fitting
    with pytest.raises(ValueError):
        model_explainer.plot_summary(X)
    
    with pytest.raises(ValueError):
        model_explainer.plot_dependence(X, "feature1")
    
    with pytest.raises(ValueError):
        model_explainer.get_feature_importance(X)
    
    with pytest.raises(ValueError):
        model_explainer.explain_prediction(X, 0)
    
    # Test invalid model type
    with pytest.raises(ValueError):
        model_explainer.fit_explainer(X, model_type="invalid") 