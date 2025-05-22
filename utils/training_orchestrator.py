import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import threading
import queue
import time
import psutil
import mlflow
from concurrent.futures import ThreadPoolExecutor, as_completed
from .model_trainer import ModelTrainer
from .model_registry import ModelRegistry
from .data_preprocessor import DataPreprocessor
from .feature_generator import FeatureGenerator

class TrainingOrchestrator:
    """
    Handles training job scheduling, resource monitoring, and error handling
    for the momentum strategy.
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        max_workers: int = 2,
        max_memory_gb: float = 8.0,
        max_cpu_percent: float = 80.0,
        job_timeout_minutes: int = 60,
        log_dir: Optional[str] = None
    ):
        """
        Initialize the training orchestrator.
        
        Args:
            registry: ModelRegistry instance for model storage
            max_workers: Maximum number of concurrent training jobs
            max_memory_gb: Maximum memory usage in GB
            max_cpu_percent: Maximum CPU usage percentage
            job_timeout_minutes: Maximum runtime for a training job in minutes
            log_dir: Directory for training logs
        """
        self.registry = registry
        self.max_workers = max_workers
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent
        self.job_timeout_minutes = job_timeout_minutes
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path("logs/training")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Setup job tracking with thread safety
        self.job_queue = queue.Queue()
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        self.failed_jobs: Dict[str, Dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_resources(self) -> None:
        """Monitor system resources and manage active jobs."""
        while self.monitoring:
            try:
                # Check memory usage
                memory_gb = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
                if memory_gb > self.max_memory_gb:
                    self.logger.warning(f"Memory usage ({memory_gb:.2f} GB) exceeds limit ({self.max_memory_gb} GB)")
                    self._handle_resource_limit()
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent()
                if cpu_percent > self.max_cpu_percent:
                    self.logger.warning(f"CPU usage ({cpu_percent}%) exceeds limit ({self.max_cpu_percent}%)")
                    self._handle_resource_limit()
                
                # Check job timeouts - do this less frequently
                current_time = datetime.now()
                jobs_to_timeout = []
                
                # First, identify jobs that need to timeout
                with self.jobs_lock:
                    for job_id, job_info in list(self.active_jobs.items()):
                        if job_info["start_time"] and (current_time - job_info["start_time"]) > timedelta(minutes=self.job_timeout_minutes):
                            jobs_to_timeout.append(job_id)
                
                # Then handle timeouts outside the lock
                for job_id in jobs_to_timeout:
                    self._handle_job_timeout(job_id)
                
                # Use a shorter sleep time for more responsive timeout checks
                time.sleep(1)  # Check every second instead of 5 seconds
            
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(1)  # Reduced from 5 to 1 second on error
    
    def _handle_resource_limit(self) -> None:
        """Handle resource limit exceeded."""
        # Pause new job starts
        self.logger.info("Pausing new job starts due to resource limits")
        # Could implement more sophisticated resource management here
    
    def _handle_job_timeout(self, job_id: str) -> None:
        """Handle job timeout."""
        with self.jobs_lock:
            if job_id in self.active_jobs:
                job_info = self.active_jobs[job_id]
                job_info["status"] = "failed"
                job_info["error"] = "Job timeout"
                job_info["end_time"] = datetime.now()
                self.failed_jobs[job_id] = job_info
                del self.active_jobs[job_id]
                self.logger.error(f"Job {job_id} timed out and was terminated")
                
                # Execute callback if provided
                if job_info.get("callback"):
                    try:
                        job_info["callback"](job_id, job_info)
                    except Exception as e:
                        self.logger.error(f"Error executing callback for timed out job {job_id}: {str(e)}")
    
    def schedule_training(
        self,
        model_name: str,
        data: pd.DataFrame,
        target: pd.Series,
        config: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> str:
        """
        Schedule a training job.
        
        Args:
            model_name: Name of the model to train
            data: Training data
            target: Target variable
            config: Optional training configuration
            callback: Optional callback function for job completion
            
        Returns:
            Job ID
        """
        job_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job_info = {
            "job_id": job_id,
            "model_name": model_name,
            "status": "queued",
            "start_time": None,
            "end_time": None,
            "config": config or {},
            "callback": callback
        }
        
        self.job_queue.put((job_id, data, target, job_info))
        self.logger.info(f"Scheduled training job {job_id} for model {model_name}")
        
        # Start worker if not already running
        if len(self.active_jobs) < self.max_workers:
            self._start_worker()
        
        return job_id
    
    def _start_worker(self) -> None:
        """Start a worker thread to process training jobs."""
        worker = threading.Thread(target=self._process_jobs)
        worker.daemon = True
        worker.start()
    
    def _process_jobs(self) -> None:
        """Process training jobs from the queue."""
        while self.monitoring:  # Changed from while True to use monitoring flag
            try:
                job_id, data, target, job_info = self.job_queue.get(timeout=1)
                
                # Single lock acquisition for all job status updates
                with self.jobs_lock:
                    if job_id in self.failed_jobs:  # Job was cancelled or failed
                        self.job_queue.task_done()
                        continue
                    
                    # Check if job is already timed out
                    if job_info["start_time"] and (datetime.now() - job_info["start_time"]) > timedelta(minutes=self.job_timeout_minutes):
                        job_info["status"] = "failed"
                        job_info["error"] = "Job timeout"
                        self.failed_jobs[job_id] = job_info
                        self.job_queue.task_done()
                        continue
                    
                    job_info["status"] = "running"
                    job_info["start_time"] = datetime.now()
                    self.active_jobs[job_id] = job_info
                
                try:
                    # Initialize components
                    self.logger.info(f"Starting job {job_id} processing")
                    preprocessor = DataPreprocessor()
                    feature_generator = FeatureGenerator()
                    
                    # Initialize ModelTrainer with job configuration
                    model_config = job_info["config"].get("model_type", "xgboost")
                    trainer = ModelTrainer(model_type=model_config)
                    
                    # Process data and train model
                    cleaned_data = preprocessor.clean_data(data)
                    self.logger.info(f"Job {job_id}: Data cleaned and scaled")
                    scaled_data = preprocessor.scale_features(cleaned_data)
                    processed_data = preprocessor.handle_missing_values(scaled_data)
                    features = feature_generator.generate_features(processed_data)
                    self.logger.info(f"Job {job_id}: Features generated, shape: {features.shape}")
                    
                    # Diagnostic: Check if features is empty or has no columns
                    if features.empty or features.shape[1] == 0:
                        self.logger.error(f"Error processing job {job_id}: Features DataFrame is empty or has no columns.")
                        with self.jobs_lock:
                            if job_id in self.active_jobs:
                                job_info = self.active_jobs[job_id]
                                job_info["status"] = "failed"
                                job_info["error"] = "Features DataFrame is empty or has no columns."
                                job_info["end_time"] = datetime.now()
                                self.failed_jobs[job_id] = job_info
                                del self.active_jobs[job_id]
                        self.job_queue.task_done()
                        return
                    
                    # Log the shape and head of features and target for debugging
                    self.logger.info(f"Features shape: {features.shape}, Target shape: {target.shape}")
                    self.logger.info(f"Features head: {features.head()}")
                    
                    # Proceed with training
                    self.logger.info(f"Job {job_id}: Starting model training")
                    model = trainer.train(features, target)
                    self.logger.info(f"Job {job_id}: Model training completed")
                    
                    # Save model to registry
                    self.logger.info(f"Job {job_id}: Attempting to register model")
                    self.registry.register_model(trainer, job_info["model_name"])
                    self.logger.info(f"Job {job_id}: Model registered successfully")
                    
                    # Update job status to completed
                    with self.jobs_lock:
                        if job_id in self.active_jobs:  # Check if job wasn't timed out
                            job_info = self.active_jobs[job_id]
                            job_info["status"] = "completed"
                            job_info["end_time"] = datetime.now()
                            self.completed_jobs[job_id] = job_info
                            del self.active_jobs[job_id]
                            
                            # Execute callback if provided
                            if job_info.get("callback"):
                                try:
                                    job_info["callback"](job_id, job_info)
                                except Exception as e:
                                    self.logger.error(f"Error executing callback for job {job_id}: {str(e)}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
                    with self.jobs_lock:
                        if job_id in self.active_jobs:  # Check if job wasn't timed out
                            job_info = self.active_jobs[job_id]
                            job_info["status"] = "failed"
                            job_info["error"] = str(e)
                            job_info["end_time"] = datetime.now()
                            self.failed_jobs[job_id] = job_info
                            del self.active_jobs[job_id]
                
                finally:
                    self.job_queue.task_done()
                
                if job_info.get("callback"):
                    try:
                        job_info["callback"](job_id, job_info)
                    except Exception as e:
                        self.logger.error(f"Error executing callback for job {job_id}: {str(e)}")
            
            except queue.Empty:
                time.sleep(0.1)  # Short sleep when queue is empty
            except Exception as e:
                self.logger.error(f"Error in job processing: {str(e)}")
                time.sleep(1)  # Sleep on error to prevent tight loop
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a training job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status information
        """
        with self.jobs_lock:
            if job_id in self.active_jobs:
                return self.active_jobs[job_id]
            elif job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            elif job_id in self.failed_jobs:
                return self.failed_jobs[job_id]
            else:
                raise ValueError(f"Job {job_id} not found")
    
    def list_jobs(
        self,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List training jobs.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of job information dictionaries
        """
        with self.jobs_lock:
            all_jobs = (
                list(self.active_jobs.values()) +
                list(self.completed_jobs.values()) +
                list(self.failed_jobs.values())
            )
            
            if status:
                return [job for job in all_jobs if job["status"] == status]
            return all_jobs
    
    def cancel_job(self, job_id: str) -> None:
        """
        Cancel a training job.
        
        Args:
            job_id: Job ID
        """
        with self.jobs_lock:
            if job_id in self.active_jobs:
                job_info = self.active_jobs[job_id]
                job_info["status"] = "cancelled"
                job_info["end_time"] = datetime.now()
                self.failed_jobs[job_id] = job_info
                del self.active_jobs[job_id]
                self.logger.info(f"Cancelled training job {job_id}")
            else:
                raise ValueError(f"Job {job_id} not found or not active")
    
    def cleanup(self) -> None:
        """Clean up resources and save training history."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        # Save training history
        history = {
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs
        }
        
        history_file = self.log_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(history, f, default=str) 