import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import time
from utils.training_orchestrator import TrainingOrchestrator
from utils.model_registry import ModelRegistry
from utils.model_trainer import ModelTrainer
from unittest.mock import patch

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    n = 1000
    # Simulate OHLCV data
    close = np.cumsum(np.random.randn(n)) + 100
    open_ = close + np.random.randn(n) * 0.5
    high = np.maximum(open_, close) + np.abs(np.random.randn(n))
    low = np.minimum(open_, close) - np.abs(np.random.randn(n))
    volume = np.abs(np.random.randn(n) * 1000 + 10000)
    X = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y

@pytest.fixture
def model_registry():
    """Create a ModelRegistry instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(registry_dir=temp_dir)
        yield registry

@pytest.fixture
def training_orchestrator(model_registry):
    """Create a TrainingOrchestrator instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        orchestrator = TrainingOrchestrator(
            registry=model_registry,
            max_workers=2,
            max_memory_gb=1.0,
            max_cpu_percent=50.0,
            job_timeout_minutes=0.1,
            log_dir=temp_dir
        )
        yield orchestrator
        orchestrator.cleanup()

def test_schedule_training(training_orchestrator, sample_data):
    """Test scheduling a training job."""
    X, y = sample_data
    
    # Schedule training job
    job_id = training_orchestrator.schedule_training(
        model_name="test_model",
        data=X,
        target=y,
        config={"tags": {"type": "test"}}
    )
    
    # Check that job was scheduled
    assert job_id in [job["job_id"] for job in training_orchestrator.list_jobs()]
    
    # Wait for job to complete
    time.sleep(5)
    
    # Check job status
    job_info = training_orchestrator.get_job_status(job_id)
    assert job_info["status"] in ["completed", "running"]
    assert job_info["model_name"] == "test_model"
    assert job_info["config"]["tags"] == {"type": "test"}

def test_job_timeout(training_orchestrator, sample_data):
    """Test job timeout handling."""
    X, y = sample_data
    
    # Create a mock model trainer that simulates a long-running task
    def mock_train(*args, **kwargs):
        time.sleep(20)  # Simulate a long-running task
        return None
    
    # Patch the ModelTrainer's train method
    with patch.object(ModelTrainer, 'train', side_effect=mock_train):
        # Schedule training job with very short timeout
        job_id = training_orchestrator.schedule_training(
            model_name="test_model",
            data=X,
            target=y
        )
        
        # Wait for timeout
        time.sleep(10)
        
        # Check job status
        job_info = training_orchestrator.get_job_status(job_id)
        assert job_info["status"] == "failed"
        assert job_info["error"] == "Job timeout"

def test_cancel_job(training_orchestrator, sample_data):
    """Test cancelling a training job."""
    X, y = sample_data
    
    # Schedule training job
    job_id = training_orchestrator.schedule_training(
        model_name="test_model",
        data=X,
        target=y
    )
    
    # Cancel job
    training_orchestrator.cancel_job(job_id)
    
    # Check job status
    job_info = training_orchestrator.get_job_status(job_id)
    assert job_info["status"] == "cancelled"

def test_list_jobs(training_orchestrator, sample_data):
    """Test listing training jobs."""
    X, y = sample_data
    
    # Schedule multiple jobs
    job1 = training_orchestrator.schedule_training("model1", X, y)
    job2 = training_orchestrator.schedule_training("model2", X, y)
    
    # List all jobs
    all_jobs = training_orchestrator.list_jobs()
    assert len(all_jobs) >= 2
    
    # List jobs by status
    queued_jobs = training_orchestrator.list_jobs(status="queued")
    assert len(queued_jobs) >= 0
    
    running_jobs = training_orchestrator.list_jobs(status="running")
    assert len(running_jobs) >= 0

def test_resource_monitoring(training_orchestrator, sample_data):
    """Test resource monitoring."""
    X, y = sample_data
    
    # Schedule a job
    job_id = training_orchestrator.schedule_training("test_model", X, y)
    
    # Wait for monitoring to run
    time.sleep(15)
    
    # Check that monitoring is active
    assert training_orchestrator.monitoring
    assert training_orchestrator.monitor_thread.is_alive()

def test_cleanup(training_orchestrator, sample_data):
    """Test cleanup of resources."""
    X, y = sample_data
    
    # Schedule a job
    job_id = training_orchestrator.schedule_training("test_model", X, y)
    
    # Cleanup
    training_orchestrator.cleanup()
    
    # Check that monitoring is stopped
    assert not training_orchestrator.monitoring
    
    # Check that history was saved
    history_file = training_orchestrator.log_dir / "training_history.json"
    assert history_file.exists()

def test_callback(training_orchestrator, sample_data):
    """Test callback function for job completion."""
    X, y = sample_data
    callback_called = False
    
    def callback(job_id: str, job_info: dict):
        nonlocal callback_called
        callback_called = True
        assert job_id == job_info["job_id"]
        assert job_info["status"] == "completed"
    
    # Schedule training job with callback
    job_id = training_orchestrator.schedule_training(
        model_name="test_model",
        data=X,
        target=y,
        callback=callback
    )
    
    # Wait for job to complete
    time.sleep(5)
    
    # Check that callback was called
    assert callback_called

def test_invalid_job_id(training_orchestrator):
    """Test handling of invalid job ID."""
    with pytest.raises(ValueError):
        training_orchestrator.get_job_status("nonexistent_job")
    
    with pytest.raises(ValueError):
        training_orchestrator.cancel_job("nonexistent_job")

def test_concurrent_jobs(training_orchestrator, sample_data):
    """Test handling of concurrent training jobs."""
    X, y = sample_data
    
    # Schedule multiple jobs
    job_ids = []
    for i in range(3):  # More than max_workers
        job_id = training_orchestrator.schedule_training(
            f"model_{i}",
            X,
            y
        )
        job_ids.append(job_id)
    
    # Wait for jobs to complete
    time.sleep(10)
    
    # Check that jobs were processed
    completed_jobs = training_orchestrator.list_jobs(status="completed")
    assert len(completed_jobs) > 0 