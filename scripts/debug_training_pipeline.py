#!/usr/bin/env python3
"""
Debug Training Pipeline - Isolate None Tensor Error
==================================================

This script isolates the exact location where the 'NoneType' object has no attribute 'dim'
error occurs in the STGNN training pipeline.
"""

import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.stgnn_data import STGNNDataProcessor
from utils.stgnn_hyperopt import create_stgnn_model
from scripts.train_stgnn_improved import ClassificationSTGNNTrainer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_tensor(tensor, name, location):
    """Validate tensor and log detailed information."""
    if tensor is None:
        logger.error(f"❌ {name} is None at {location}")
        return False
    
    if not isinstance(tensor, torch.Tensor):
        logger.error(f"❌ {name} is not a tensor: {type(tensor)} at {location}")
        return False
    
    logger.info(f"✅ {name}: {tensor.shape}, dtype={tensor.dtype}, device={tensor.device} at {location}")
    return True

def debug_training_pipeline():
    """Debug the training pipeline step by step."""
    logger.info("🔍 Starting comprehensive training pipeline debug...")
    
    # Step 1: Data Preparation
    logger.info("\n" + "="*50)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("="*50)
    
    try:
        # Initialize data processor
        data_processor = STGNNDataProcessor()
        logger.info("✅ Data processor initialized")
        
        # Prepare data with ultra-minimal parameters
        config_dict = {
            'hidden_dim': 4,
            'batch_size': 1,
            'seq_len': 5,
            'num_layers': 1,
            'kernel_size': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'class_multiplier_0': 1.0,
            'class_multiplier_1': 1.0,
            'class_multiplier_2': 1.0
        }
        
        # Prepare data
        data = data_processor.prepare_data(
            start_time='2020-01-01',
            end_time='2025-05-29',
            features=['returns', 'volume'],
            seq_len=config_dict['seq_len'],
            max_sequences=1000
        )
        
        logger.info("✅ Data preparation completed")
        
        # Validate data tensors
        X, adj, y = data['X'], data['adj'], data['y']
        validate_tensor(X, "X", "data preparation")
        validate_tensor(adj, "adj", "data preparation")
        validate_tensor(y, "y", "data preparation")
        
        # Step 2: Model Creation
        logger.info("\n" + "="*50)
        logger.info("STEP 2: MODEL CREATION")
        logger.info("="*50)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create model
        model = create_stgnn_model(
            input_dim=X.shape[-1],
            hidden_dim=config_dict['hidden_dim'],
            num_layers=config_dict['num_layers'],
            kernel_size=config_dict['kernel_size'],
            dropout=config_dict['dropout'],
            num_classes=3
        )
        
        logger.info("✅ Model created")
        validate_tensor(next(model.parameters()), "model parameters", "model creation")
        
        # Move model to device
        model = model.to(device)
        logger.info("✅ Model moved to device")
        
        # Step 3: Trainer Initialization
        logger.info("\n" + "="*50)
        logger.info("STEP 3: TRAINER INITIALIZATION")
        logger.info("="*50)
        
        trainer = ClassificationSTGNNTrainer(
            model=model,
            device=device,
            learning_rate=config_dict['learning_rate'],
            focal_alpha=config_dict['focal_alpha'],
            focal_gamma=config_dict['focal_gamma'],
            class_multipliers=[
                config_dict['class_multiplier_0'],
                config_dict['class_multiplier_1'],
                config_dict['class_multiplier_2']
            ]
        )
        
        logger.info("✅ Trainer initialized")
        
        # Step 4: Data Loading Debug
        logger.info("\n" + "="*50)
        logger.info("STEP 4: DATA LOADING DEBUG")
        logger.info("="*50)
        
        # Validate data before passing to trainer
        logger.info("Validating data before trainer...")
        validate_tensor(X, "X", "before trainer")
        validate_tensor(adj, "adj", "before trainer")
        validate_tensor(y, "y", "before trainer")
        
        # Check data types and devices
        logger.info(f"X device: {X.device}")
        logger.info(f"adj device: {adj.device}")
        logger.info(f"y device: {y.device}")
        
        # Move data to device if needed
        if X.device != device:
            X = X.to(device)
            logger.info("✅ X moved to device")
        if adj.device != device:
            adj = adj.to(device)
            logger.info("✅ adj moved to device")
        if y.device != device:
            y = y.to(device)
            logger.info("✅ y moved to device")
        
        # Step 5: Training Loop Debug
        logger.info("\n" + "="*50)
        logger.info("STEP 5: TRAINING LOOP DEBUG")
        logger.info("="*50)
        
        # Try to run one training step
        try:
            logger.info("Attempting to run one training step...")
            
            # Set model to training mode
            model.train()
            logger.info("✅ Model set to training mode")
            
            # Zero gradients
            trainer.optimizer.zero_grad()
            logger.info("✅ Gradients zeroed")
            
            # Forward pass with detailed logging
            logger.info("Starting forward pass...")
            
            # Validate inputs before forward pass
            validate_tensor(X, "X", "before forward pass")
            validate_tensor(adj, "adj", "before forward pass")
            
            # Try forward pass
            outputs = model(X, adj)
            logger.info("✅ Forward pass completed")
            validate_tensor(outputs, "outputs", "after forward pass")
            
            # Validate targets
            validate_tensor(y, "y", "before loss calculation")
            
            # Calculate loss
            loss = trainer.criterion(outputs, y.squeeze())
            logger.info("✅ Loss calculated")
            validate_tensor(loss, "loss", "after loss calculation")
            
            # Backward pass
            loss.backward()
            logger.info("✅ Backward pass completed")
            
            # Optimizer step
            trainer.optimizer.step()
            logger.info("✅ Optimizer step completed")
            
            logger.info("🎉 SUCCESS: Complete training step completed!")
            
        except Exception as e:
            logger.error(f"❌ Training step failed: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            # Additional debugging for specific error
            if "'NoneType' object has no attribute 'dim'" in str(e):
                logger.error("🔍 This is the exact error we're looking for!")
                logger.error("Need to identify which tensor is None before .dim is called")
        
        # Step 6: Evaluation Debug
        logger.info("\n" + "="*50)
        logger.info("STEP 6: EVALUATION DEBUG")
        logger.info("="*50)
        
        try:
            logger.info("Attempting evaluation...")
            model.eval()
            
            with torch.no_grad():
                # Validate inputs before evaluation
                validate_tensor(X, "X", "before evaluation")
                validate_tensor(adj, "adj", "before evaluation")
                validate_tensor(y, "y", "before evaluation")
                
                # Forward pass for evaluation
                outputs = model(X, adj)
                logger.info("✅ Evaluation forward pass completed")
                validate_tensor(outputs, "outputs", "after evaluation forward pass")
                
                # Calculate evaluation metrics
                predictions = torch.argmax(outputs, dim=1)
                logger.info("✅ Predictions calculated")
                validate_tensor(predictions, "predictions", "after argmax")
                
                logger.info("🎉 SUCCESS: Complete evaluation completed!")
                
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
    
    except Exception as e:
        logger.error(f"❌ Debug failed at early stage: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    debug_training_pipeline() 