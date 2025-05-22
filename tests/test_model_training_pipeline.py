import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from utils.synthetic_data_generator import SyntheticDataGenerator
from utils.data_preprocessor import DataPreprocessor
from utils.feature_generator import FeatureGenerator
from utils.model_trainer import ModelTrainer
from utils.model_registry import ModelRegistry
from utils.training_orchestrator import TrainingOrchestrator

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def data_generator():
    """Create a SyntheticDataGenerator instance for testing."""
    return SyntheticDataGenerator(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 7),  # 7 days of data
        base_price=100.0,
        volatility=0.02,
        trend_strength=0.1,
        volume_base=1000.0,
        seed=42
    )

@pytest.fixture
def model_registry(temp_dir):
    """Create a ModelRegistry instance for testing."""
    return ModelRegistry(registry_dir=temp_dir)

@pytest.fixture
def training_orchestrator(model_registry, temp_dir):
    """Create a TrainingOrchestrator instance for testing."""
    return TrainingOrchestrator(
        registry=model_registry,
        max_workers=2,
        max_memory_gb=1.0,
        max_cpu_percent=50.0,
        job_timeout_minutes=5,
        log_dir=temp_dir
    )

def test_end_to_end_pipeline(data_generator, training_orchestrator, temp_dir):
    """Test the entire model training pipeline."""
    # 1. Generate synthetic dataset
    ohlcv_data, order_book_data, features, target = data_generator.generate_dataset(
        interval_minutes=15,
        n_levels=5,
        lookahead=5,
        threshold=0.001
    )
    
    # 2. Create training components
    preprocessor = DataPreprocessor()
    feature_generator = FeatureGenerator()
    model_trainer = ModelTrainer(
        model_type="xgboost",
        config={
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100
        },
        cache_dir=temp_dir
    )
    
    # 3. Schedule training job
    job_id = training_orchestrator.schedule_training(
        model_name="test_model",
        data=features,
        target=target,
        config={
            "tags": {"type": "test"},
            "model_type": "xgboost",
            "hyperparameters": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100
            }
        }
    )
    
    # 4. Wait for job completion
    import time
    max_wait = 60  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        job_info = training_orchestrator.get_job_status(job_id)
        if job_info["status"] in ["completed", "failed"]:
            break
        time.sleep(1)
    
    # 5. Verify job completion
    assert job_info["status"] == "completed", f"Training job failed: {job_info.get('error', 'Unknown error')}"
    assert "version_id" in job_info, "No model version ID in job info"
    assert "metrics" in job_info, "No metrics in job info"
    
    # 6. Verify model metrics
    metrics = job_info["metrics"]
    assert "accuracy" in metrics, "Accuracy metric missing"
    assert "precision" in metrics, "Precision metric missing"
    assert "recall" in metrics, "Recall metric missing"
    assert "f1" in metrics, "F1 metric missing"
    
    # 7. Verify model version
    version_id = job_info["version_id"]
    model_version = training_orchestrator.registry.get_model_version("test_model", version_id)
    assert model_version is not None, "Model version not found in registry"
    
    # 8. Test model prediction
    model = model_version["model"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    
    assert len(predictions) == len(features), "Prediction length mismatch"
    assert len(probabilities) == len(features), "Probability length mismatch"
    assert all(pred in [0, 1] for pred in predictions), "Invalid prediction values"
    assert all(0 <= prob <= 1 for prob in probabilities), "Invalid probability values"
    
    # 9. Verify feature importance
    feature_importance = model.get_feature_importance()
    assert isinstance(feature_importance, dict), "Feature importance should be a dictionary"
    assert len(feature_importance) > 0, "No feature importance scores"
    
    # 10. Test model deployment
    deployment_id = training_orchestrator.registry.deploy_model("test_model", version_id)
    assert deployment_id is not None, "Deployment failed"
    
    # 11. Verify deployment
    deployment = training_orchestrator.registry.get_deployment(deployment_id)
    assert deployment is not None, "Deployment not found"
    assert deployment["model_name"] == "test_model", "Wrong model name in deployment"
    assert deployment["version_id"] == version_id, "Wrong version ID in deployment"
    
    # 12. Clean up
    training_orchestrator.cleanup()
    assert not training_orchestrator.monitoring, "Monitoring not stopped"
    
    # 13. Verify history was saved
    history_file = Path(temp_dir) / "training_history.json"
    assert history_file.exists(), "Training history not saved"

def test_pipeline_with_invalid_data(data_generator, training_orchestrator):
    """Test pipeline behavior with invalid data."""
    # Generate data with NaN values
    ohlcv_data, order_book_data, features, target = data_generator.generate_dataset()
    features.iloc[0, 0] = np.nan  # Introduce NaN
    
    # Schedule training job
    job_id = training_orchestrator.schedule_training(
        model_name="test_model_invalid",
        data=features,
        target=target
    )
    
    # Wait for job completion
    import time
    max_wait = 60  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        job_info = training_orchestrator.get_job_status(job_id)
        if job_info["status"] in ["completed", "failed"]:
            break
        time.sleep(1)
    
    # Verify job failed gracefully
    assert job_info["status"] == "failed", "Job should fail with invalid data"
    assert "error" in job_info, "No error message in job info"

def test_concurrent_training_jobs(data_generator, training_orchestrator):
    """Test handling of concurrent training jobs."""
    # Generate dataset
    ohlcv_data, order_book_data, features, target = data_generator.generate_dataset()
    
    # Schedule multiple jobs
    job_ids = []
    for i in range(3):  # More than max_workers
        job_id = training_orchestrator.schedule_training(
            model_name=f"test_model_{i}",
            data=features,
            target=target
        )
        job_ids.append(job_id)
    
    # Wait for jobs to complete
    import time
    max_wait = 120  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        job_infos = [training_orchestrator.get_job_status(job_id) for job_id in job_ids]
        if all(info["status"] in ["completed", "failed"] for info in job_infos):
            break
        time.sleep(1)
    
    # Verify all jobs completed
    for job_id in job_ids:
        job_info = training_orchestrator.get_job_status(job_id)
        assert job_info["status"] == "completed", f"Job {job_id} failed: {job_info.get('error', 'Unknown error')}"
    
    # Verify job ordering
    job_times = [job_info["end_time"] - job_info["start_time"] for job_info in job_infos]
    assert len(set(job_times)) > 1, "Jobs should not all take the same time" 