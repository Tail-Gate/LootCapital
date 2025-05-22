import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from utils.model_registry import ModelRegistry
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
def model_trainer(sample_data):
    """Create a trained ModelTrainer instance for testing."""
    X, y = sample_data
    trainer = ModelTrainer(model_type="xgboost")
    trainer.train(X, y)
    return trainer

@pytest.fixture
def model_registry():
    """Create a ModelRegistry instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(registry_dir=temp_dir)
        yield registry

def test_register_model(model_registry, model_trainer):
    """Test model registration."""
    # Register model
    version_id = model_registry.register_model(
        model_trainer,
        model_name="test_model",
        description="Test model",
        tags={"type": "test"}
    )
    
    # Check that model was registered
    assert version_id in model_registry.metadata["models"]["test_model"]
    assert model_registry.metadata["latest_versions"]["test_model"] == version_id
    
    # Check version metadata
    version_metadata = model_registry.metadata["models"]["test_model"][version_id]
    assert version_metadata["description"] == "Test model"
    assert version_metadata["tags"] == {"type": "test"}
    assert "metrics" in version_metadata
    assert "params" in version_metadata
    assert "feature_importance" in version_metadata

def test_get_model_version(model_registry, model_trainer):
    """Test getting a model version."""
    # Register model
    version_id = model_registry.register_model(model_trainer, "test_model")
    
    # Get model version
    loaded_trainer, metadata = model_registry.get_model_version("test_model", version_id)
    
    # Check that model was loaded correctly
    assert loaded_trainer.model is not None
    assert loaded_trainer.best_params == model_trainer.best_params
    assert loaded_trainer.feature_importance == model_trainer.feature_importance
    
    # Check metadata
    assert metadata["version_id"] == version_id

def test_list_models(model_registry, model_trainer):
    """Test listing registered models."""
    # Register multiple models
    model_registry.register_model(model_trainer, "model1")
    model_registry.register_model(model_trainer, "model2")
    
    # List models
    models = model_registry.list_models()
    
    # Check that all models are listed
    assert set(models) == {"model1", "model2"}

def test_list_versions(model_registry, model_trainer):
    """Test listing model versions."""
    # Register multiple versions
    version1 = model_registry.register_model(model_trainer, "test_model")
    version2 = model_registry.register_model(model_trainer, "test_model")
    
    # List versions
    versions = model_registry.list_versions("test_model")
    
    # Check that all versions are listed
    version_ids = [v["version_id"] for v in versions]
    assert set(version_ids) == {version1, version2}

def test_deploy_model(model_registry, model_trainer):
    """Test model deployment."""
    # Register model
    version_id = model_registry.register_model(model_trainer, "test_model")
    
    # Deploy model
    deployment_id = model_registry.deploy_model(
        "test_model",
        version_id,
        deployment_name="test_deployment"
    )
    
    # Check that model was deployed
    assert deployment_id in model_registry.metadata["deployed_models"]
    
    # Check deployment metadata
    deployment_metadata = model_registry.metadata["deployed_models"][deployment_id]
    assert deployment_metadata["model_name"] == "test_model"
    assert deployment_metadata["version_id"] == version_id
    assert deployment_metadata["status"] == "active"

def test_get_deployment(model_registry, model_trainer):
    """Test getting a deployed model."""
    # Register and deploy model
    version_id = model_registry.register_model(model_trainer, "test_model")
    deployment_id = model_registry.deploy_model("test_model", version_id)
    
    # Get deployment
    loaded_trainer, metadata = model_registry.get_deployment(deployment_id)
    
    # Check that model was loaded correctly
    assert loaded_trainer.model is not None
    assert loaded_trainer.best_params == model_trainer.best_params
    assert loaded_trainer.feature_importance == model_trainer.feature_importance
    
    # Check metadata
    assert metadata["deployment_id"] == deployment_id
    assert metadata["model_name"] == "test_model"
    assert metadata["version_id"] == version_id

def test_list_deployments(model_registry, model_trainer):
    """Test listing deployments."""
    # Register and deploy multiple models
    version1 = model_registry.register_model(model_trainer, "model1")
    version2 = model_registry.register_model(model_trainer, "model2")
    deployment1 = model_registry.deploy_model("model1", version1)
    deployment2 = model_registry.deploy_model("model2", version2)
    
    # List deployments
    deployments = model_registry.list_deployments()
    
    # Check that all deployments are listed
    deployment_ids = [d["deployment_id"] for d in deployments]
    assert set(deployment_ids) == {deployment1, deployment2}

def test_delete_deployment(model_registry, model_trainer):
    """Test deleting a deployment."""
    # Register and deploy model
    version_id = model_registry.register_model(model_trainer, "test_model")
    deployment_id = model_registry.deploy_model("test_model", version_id)
    
    # Delete deployment
    model_registry.delete_deployment(deployment_id)
    
    # Check that deployment was deleted
    assert deployment_id not in model_registry.metadata["deployed_models"]
    
    # Check that deployment directory was deleted
    deployment_dir = model_registry.registry_dir / "deployments" / deployment_id
    assert not deployment_dir.exists()

def test_compare_versions(model_registry, model_trainer):
    """Test comparing model versions."""
    # Register multiple versions
    version1 = model_registry.register_model(model_trainer, "test_model")
    version2 = model_registry.register_model(model_trainer, "test_model")
    
    # Compare versions
    comparison = model_registry.compare_versions("test_model", [version1, version2])
    
    # Check comparison data
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 2
    assert set(comparison["version_id"]) == {version1, version2}
    assert "accuracy" in comparison.columns
    assert "precision" in comparison.columns
    assert "recall" in comparison.columns
    assert "f1" in comparison.columns

def test_invalid_model_name(model_registry):
    """Test handling of invalid model name."""
    with pytest.raises(ValueError):
        model_registry.get_model_version("nonexistent_model")
    
    with pytest.raises(ValueError):
        model_registry.list_versions("nonexistent_model")
    
    with pytest.raises(ValueError):
        model_registry.deploy_model("nonexistent_model")
    
    with pytest.raises(ValueError):
        model_registry.compare_versions("nonexistent_model", ["version1"])

def test_invalid_version_id(model_registry, model_trainer):
    """Test handling of invalid version ID."""
    # Register model
    model_registry.register_model(model_trainer, "test_model")
    
    with pytest.raises(ValueError):
        model_registry.get_model_version("test_model", "nonexistent_version")
    
    with pytest.raises(ValueError):
        model_registry.deploy_model("test_model", "nonexistent_version")
    
    with pytest.raises(ValueError):
        model_registry.compare_versions("test_model", ["nonexistent_version"])

def test_invalid_deployment_id(model_registry):
    """Test handling of invalid deployment ID."""
    with pytest.raises(ValueError):
        model_registry.get_deployment("nonexistent_deployment")
    
    with pytest.raises(ValueError):
        model_registry.delete_deployment("nonexistent_deployment") 