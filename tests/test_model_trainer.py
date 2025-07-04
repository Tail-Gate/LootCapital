import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
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
def model_trainer():
    """Create a ModelTrainer instance for testing."""
    return ModelTrainer(model_type="xgboost")

def test_optimize_hyperparameters(model_trainer, sample_data):
    """Test hyperparameter optimization."""
    X, y = sample_data
    
    # Run optimization with fewer trials for testing
    best_params = model_trainer.optimize_hyperparameters(
        X, y,
        n_trials=5,  # Reduced for testing
        cv_splits=3  # Reduced for testing
    )
    
    # Check that best parameters are found
    assert isinstance(best_params, dict)
    assert "max_depth" in best_params
    assert "learning_rate" in best_params
    assert "n_estimators" in best_params

def test_train(model_trainer, sample_data):
    """Test model training."""
    X, y = sample_data
    
    # Train model
    metrics = model_trainer.train(X, y, validation_split=0.2)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    
    # Check that model was trained
    assert model_trainer.model is not None
    
    # Check feature importance
    feature_importance = model_trainer.get_feature_importance()
    assert isinstance(feature_importance, dict)
    assert len(feature_importance) == len(X.columns)

def test_predict(model_trainer, sample_data):
    """Test model prediction."""
    X, y = sample_data
    
    # Train model first
    model_trainer.train(X, y)
    
    # Make predictions
    predictions = model_trainer.predict(X)
    probabilities = model_trainer.predict_proba(X)
    
    # Check predictions
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert set(np.unique(predictions)) <= {0, 1}
    
    # Check probabilities
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

def test_save_load_model(model_trainer, sample_data):
    """Test saving and loading model."""
    X, y = sample_data
    
    # Train model
    model_trainer.train(X, y)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_trainer.save_model(temp_dir)
        
        # Create new trainer and load model
        new_trainer = ModelTrainer(model_type="xgboost")
        new_trainer.load_model(temp_dir)
        
        # Verify model was loaded correctly
        assert new_trainer.model is not None
        assert new_trainer.best_params == model_trainer.best_params
        assert new_trainer.feature_importance == model_trainer.feature_importance
        
        # Verify predictions match
        original_preds = model_trainer.predict(X)
        loaded_preds = new_trainer.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

def test_training_history(model_trainer, sample_data):
    """Test training history tracking."""
    X, y = sample_data
    
    # Train model multiple times
    for _ in range(3):
        model_trainer.train(X, y)
    
    # Get training history
    history = model_trainer.get_training_history()
    
    # Check history
    assert isinstance(history, list)
    assert len(history) == 3
    
    for entry in history:
        assert "timestamp" in entry
        assert "metrics" in entry
        assert "params" in entry
        assert isinstance(entry["metrics"], dict)
        assert isinstance(entry["params"], dict)

def test_invalid_model_type():
    """Test handling of invalid model type."""
    with pytest.raises(ValueError):
        ModelTrainer(model_type="invalid_type")

def test_predict_before_training(model_trainer, sample_data):
    """Test prediction before training."""
    X, _ = sample_data
    
    with pytest.raises(ValueError):
        model_trainer.predict(X)
    
    with pytest.raises(ValueError):
        model_trainer.predict_proba(X)

def test_save_before_training(model_trainer):
    """Test saving before training."""
    with pytest.raises(ValueError):
        model_trainer.save_model()

def test_model_explanation(model_trainer, sample_data):
    """Test model explanation functionality."""
    X, y = sample_data
    
    # Train model
    model_trainer.train(X, y)
    
    # Test explaining a prediction
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "explanation.png"
        model_trainer.explain_prediction(X, 0, str(output_path))
        assert output_path.exists()
    
    # Test plotting feature importance
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "importance.png"
        model_trainer.plot_feature_importance(X, str(output_path))
        assert output_path.exists()
    
    # Test plotting feature dependence
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "dependence.png"
        model_trainer.plot_feature_dependence(X, "feature1", str(output_path))
        assert output_path.exists()

def test_explanation_before_training(model_trainer, sample_data):
    """Test explanation methods before training."""
    X, _ = sample_data
    
    with pytest.raises(ValueError):
        model_trainer.explain_prediction(X, 0)
    
    with pytest.raises(ValueError):
        model_trainer.plot_feature_importance(X)
    
    with pytest.raises(ValueError):
        model_trainer.plot_feature_dependence(X, "feature1") 