import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import shutil
import mlflow
from mlflow.models import infer_signature
import joblib
from .model_trainer import ModelTrainer

class ModelRegistry:
    """
    Handles model storage, versioning, and deployment for the momentum strategy.
    """
    
    def __init__(
        self,
        registry_dir: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "momentum_strategy"
    ):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory to store model artifacts
            mlflow_tracking_uri: URI for MLflow tracking server
            experiment_name: Name of the MLflow experiment
        """
        self.registry_dir = Path(registry_dir) if registry_dir else Path("models/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup MLflow if URI provided
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.use_mlflow = True
        else:
            self.use_mlflow = False
        
        # Load registry metadata
        self.metadata_file = self.registry_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load registry metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {
            "models": {},
            "latest_versions": {},
            "deployed_models": {}
        }
    
    def _save_metadata(self) -> None:
        """Save registry metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(
        self,
        model_trainer: ModelTrainer,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a trained model in the registry.
        
        Args:
            model_trainer: Trained ModelTrainer instance
            model_name: Name of the model
            description: Optional description of the model
            tags: Optional tags for the model
            
        Returns:
            Version ID of the registered model
        """
        if model_trainer.model is None:
            raise ValueError("Model not trained yet")
        
        # Generate version ID
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create version directory
        version_dir = self.registry_dir / model_name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        model_trainer.save_model(str(version_dir))
        
        # Create version metadata
        version_metadata = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "tags": tags or {},
            "metrics": model_trainer.get_training_history()[-1]["metrics"],
            "params": model_trainer.best_params,
            "feature_importance": model_trainer.get_feature_importance()
        }
        
        # Update registry metadata
        if model_name not in self.metadata["models"]:
            self.metadata["models"][model_name] = {}
        
        self.metadata["models"][model_name][version_id] = version_metadata
        self.metadata["latest_versions"][model_name] = version_id
        
        # Save metadata
        self._save_metadata()
        
        # Log to MLflow if enabled
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"{model_name}_{version_id}"):
                mlflow.log_params(version_metadata["params"])
                mlflow.log_metrics(version_metadata["metrics"])
                mlflow.log_dict(version_metadata, "version_metadata.json")
                
                # Log model
                signature = infer_signature(
                    pd.DataFrame(model_trainer.model.feature_names_),
                    model_trainer.predict(pd.DataFrame(model_trainer.model.feature_names_))
                )
                mlflow.sklearn.log_model(
                    model_trainer.model,
                    "model",
                    signature=signature
                )
        
        self.logger.info(f"Registered model {model_name} version {version_id}")
        return version_id
    
    def get_model_version(
        self,
        model_name: str,
        version_id: Optional[str] = None
    ) -> Tuple[ModelTrainer, Dict[str, Any]]:
        """
        Get a specific version of a model.
        
        Args:
            model_name: Name of the model
            version_id: Version ID (defaults to latest)
            
        Returns:
            Tuple of (ModelTrainer instance, version metadata)
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version_id is None:
            version_id = self.metadata["latest_versions"][model_name]
        
        if version_id not in self.metadata["models"][model_name]:
            raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        # Get version metadata
        version_metadata = self.metadata["models"][model_name][version_id]
        
        # Initialize model trainer with best parameters
        model_trainer = ModelTrainer(
            model_type="xgboost",  # Assuming XGBoost for now
            config=version_metadata["params"]
        )
        
        # Load model
        version_dir = self.registry_dir / model_name / version_id
        model_trainer.load_model(str(version_dir))
        
        return model_trainer, version_metadata
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.metadata["models"].keys())
    
    def list_versions(
        self,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version metadata dictionaries
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return [
            {"version_id": v, **metadata}
            for v, metadata in self.metadata["models"][model_name].items()
        ]
    
    def deploy_model(
        self,
        model_name: str,
        version_id: Optional[str] = None,
        deployment_name: Optional[str] = None
    ) -> str:
        """
        Deploy a model version.
        
        Args:
            model_name: Name of the model
            version_id: Version ID (defaults to latest)
            deployment_name: Name for the deployment (defaults to model_name)
            
        Returns:
            Deployment ID
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version_id is None:
            version_id = self.metadata["latest_versions"][model_name]
        
        if version_id not in self.metadata["models"][model_name]:
            raise ValueError(f"Version {version_id} not found for model {model_name}")
        
        deployment_name = deployment_name or model_name
        deployment_id = f"{deployment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create deployment directory
        deployment_dir = self.registry_dir / "deployments" / deployment_id
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        version_dir = self.registry_dir / model_name / version_id
        shutil.copytree(version_dir, deployment_dir / "model", dirs_exist_ok=True)
        
        # Create deployment metadata
        deployment_metadata = {
            "deployment_id": deployment_id,
            "model_name": model_name,
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Update registry metadata
        self.metadata["deployed_models"][deployment_id] = deployment_metadata
        
        # Save metadata
        self._save_metadata()
        
        self.logger.info(f"Deployed model {model_name} version {version_id} as {deployment_id}")
        return deployment_id
    
    def get_deployment(
        self,
        deployment_id: str
    ) -> Tuple[ModelTrainer, Dict[str, Any]]:
        """
        Get a deployed model.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Tuple of (ModelTrainer instance, deployment metadata)
        """
        if deployment_id not in self.metadata["deployed_models"]:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment_metadata = self.metadata["deployed_models"][deployment_id]
        
        # Get model version metadata to get best parameters
        model_name = deployment_metadata["model_name"]
        version_id = deployment_metadata["version_id"]
        version_metadata = self.metadata["models"][model_name][version_id]
        
        # Initialize model trainer with best parameters
        model_trainer = ModelTrainer(
            model_type="xgboost",  # Assuming XGBoost for now
            config=version_metadata["params"]
        )
        
        # Load model
        deployment_dir = self.registry_dir / "deployments" / deployment_id
        model_trainer.load_model(str(deployment_dir / "model"))
        
        return model_trainer, deployment_metadata
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments."""
        return list(self.metadata["deployed_models"].values())
    
    def delete_deployment(self, deployment_id: str) -> None:
        """
        Delete a deployment.
        
        Args:
            deployment_id: Deployment ID
        """
        if deployment_id not in self.metadata["deployed_models"]:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        # Delete deployment directory
        deployment_dir = self.registry_dir / "deployments" / deployment_id
        if deployment_dir.exists():
            shutil.rmtree(deployment_dir)
        
        # Update metadata
        del self.metadata["deployed_models"][deployment_id]
        self._save_metadata()
        
        self.logger.info(f"Deleted deployment {deployment_id}")
    
    def compare_versions(
        self,
        model_name: str,
        version_ids: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple versions of a model.
        
        Args:
            model_name: Name of the model
            version_ids: List of version IDs to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        comparison_data = []
        
        for version_id in version_ids:
            if version_id not in self.metadata["models"][model_name]:
                raise ValueError(f"Version {version_id} not found for model {model_name}")
            
            version_metadata = self.metadata["models"][model_name][version_id]
            comparison_data.append({
                "version_id": version_id,
                "timestamp": version_metadata["timestamp"],
                **version_metadata["metrics"]
            })
        
        return pd.DataFrame(comparison_data) 