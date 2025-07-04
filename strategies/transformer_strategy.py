import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
from utils.transformer_utils import (
    TransformerModel,
    train_transformer,
    predict_transformer,
    save_transformer,
    load_transformer,
    visualize_attention
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

@dataclass
class TransformerConfig:
    """Configuration for Transformer Strategy"""
    input_size: int = 5  # OHLCV
    d_model: int = 32  # Reduced from 64
    nhead: int = 4  # Reduced from 8
    num_layers: int = 2  # Reduced from 3
    dim_feedforward: int = 128  # Reduced from 256
    dropout: float = 0.1
    sequence_length: int = 20
    num_classes: int = 3  # -1, 0, 1
    device: Optional[str] = None
    batch_size: int = 16  # Reduced from 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    num_epochs: int = 50
    validation_split: float = 0.2
    memory_optimization: bool = True
    focal_loss_gamma: float = 2.0
    class_weights: Optional[List[float]] = None  # Weights for imbalanced classes

class TransformerStrategy:
    """Trading strategy based on transformer architecture"""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize the transformer strategy
        
        Args:
            config: TransformerConfig object containing model parameters
        """
        self.config = config
        
        # Set device with Apple GPU (MPS) support
        if config.device:
            self.device = torch.device(config.device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Silicon MPS (GPU) for training.")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA GPU for training.")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for training.")
        
        # Initialize model
        self.model = TransformerModel(
            input_size=config.input_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            num_classes=config.num_classes,
            dropout=config.dropout
        ).to(self.device)
        
        # Set sequence length in model for dataset creation
        self.model.sequence_length = config.sequence_length
        
        # Store attention weights for visualization
        self.attention_weights = None
        
        # Initialize class weights with fixed values
        self.class_weights = torch.tensor([50.0, 1.0, 50.0], dtype=torch.float32)
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for transformer model with memory optimization"""
        # Log input shapes
        logging.info(f"\nPrepare data method input:")
        logging.info(f"Features shape: {features.shape}")
        logging.info(f"Target shape: {target.shape}")
        logging.info(f"Sequence length: {self.config.sequence_length}")
        
        # Ensure features are numeric
        features = features.select_dtypes(include=[np.number])
        logging.info(f"Numeric features shape: {features.shape}")
        
        # Convert to numpy arrays
        X = features.values
        y = target.values
        logging.info(f"X array shape: {X.shape}")
        logging.info(f"y array shape: {y.shape}")
        
        # Create sequences in batches to reduce memory usage
        if self.config.memory_optimization:
            X_sequences = []
            y_sequences = []
            batch_size = 1000  # Process 1000 sequences at a time
            
            for i in range(0, len(X) - self.config.sequence_length, batch_size):
                end_idx = min(i + batch_size, len(X) - self.config.sequence_length)
                # Create sequences with correct shape [batch_size, sequence_length, features]
                X_batch = np.array([X[j:j + self.config.sequence_length] for j in range(i, end_idx)])
                y_batch = np.array([y[j + self.config.sequence_length - 1] for j in range(i, end_idx)])
                
                X_sequences.append(X_batch)
                y_sequences.append(y_batch)
                
                # Log batch progress
                if i % (batch_size * 10) == 0:
                    logging.info(f"Processed {i} to {end_idx} sequences")
                
                # Force garbage collection after each batch
                if i % (batch_size * 10) == 0:
                    import gc
                    gc.collect()
            
            # Concatenate all batches
            X_tensor = torch.FloatTensor(np.concatenate(X_sequences, axis=0))
            y_tensor = torch.LongTensor(np.concatenate(y_sequences, axis=0))
        else:
            # Original implementation for smaller datasets
            X_sequences = np.array([X[i:i + self.config.sequence_length] for i in range(len(X) - self.config.sequence_length)])
            y_sequences = np.array([y[i + self.config.sequence_length - 1] for i in range(len(X) - self.config.sequence_length)])
            
            X_tensor = torch.FloatTensor(X_sequences)
            y_tensor = torch.LongTensor(y_sequences)
        
        # Log final tensor shapes
        logging.info(f"\nFinal tensor shapes:")
        logging.info(f"X_tensor shape: {X_tensor.shape}")
        logging.info(f"y_tensor shape: {y_tensor.shape}")
        
        return X_tensor, y_tensor
    
    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the transformer model with memory optimization"""
        # Prepare data
        X, y = self.prepare_data(features, target)
        
        # Split data
        train_size = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Clear memory before training
        if self.config.memory_optimization:
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Train model
        history = train_transformer(
            model=self.model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            num_epochs=self.config.num_epochs,
            early_stopping_patience=self.config.early_stopping_patience,
            device=self.device,
            batch_size=self.config.batch_size,
            memory_optimization=self.config.memory_optimization,
            class_weights=self.class_weights
        )
        
        # Clear memory after training
        if self.config.memory_optimization:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Set model to eval mode after training
        self.model.eval()
        
        return history
    
    def predict(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with memory optimization
        
        Returns:
            Tuple of (class_predictions, class_probabilities)
        """
        # Log input data shape
        logging.info(f"\nPredict method input:")
        logging.info(f"Data shape: {data.shape}")
        
        # Prepare data
        X, _ = self.prepare_data(data, pd.Series(0, index=data.index))
        logging.info(f"Prepared data shape: {X.shape}")
        
        # Make predictions
        predictions, probabilities = predict_transformer(
            self.model,
            X,
            self.device,
            batch_size=self.config.batch_size,
            memory_optimization=self.config.memory_optimization
        )
        logging.info(f"Raw predictions shape: {predictions.shape}")
        logging.info(f"Probabilities shape: {probabilities.shape}")
        
        # Ensure predictions are 1D array
        predictions = predictions.flatten()
        logging.info(f"Flattened predictions shape: {predictions.shape}")
        
        return predictions, probabilities
    
    def evaluate(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        return_metrics: bool = False
    ) -> Union[float, Dict[str, float]]:
        """
        Evaluate the model on test data
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            return_metrics: If True, return all metrics. If False, return balanced accuracy only
            
        Returns:
            If return_metrics is False: Balanced accuracy score (float)
            If return_metrics is True: Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix, balanced_accuracy_score
        )
        
        # Log input shapes
        logging.info(f"\nEvaluate method input shapes:")
        logging.info(f"Features shape: {features.shape}")
        logging.info(f"Targets shape: {targets.shape}")
        logging.info(f"Sequence length: {self.config.sequence_length}")
        
        # Make predictions
        predictions, _ = self.predict(features)
        logging.info(f"Predictions shape: {predictions.shape}")
        
        # Ensure predictions and targets have the same length
        targets_aligned = targets.iloc[self.config.sequence_length:].values
        logging.info(f"Aligned targets shape: {targets_aligned.shape}")
        
        # Log index ranges for debugging
        logging.info(f"Original targets index range: {targets.index[0]} to {targets.index[-1]}")
        logging.info(f"Aligned targets index range: {targets.index[self.config.sequence_length]} to {targets.index[-1]}")
        
        # Verify lengths match
        if len(predictions) != len(targets_aligned):
            logging.error(f"Length mismatch details:")
            logging.error(f"Predictions length: {len(predictions)}")
            logging.error(f"Aligned targets length: {len(targets_aligned)}")
            logging.error(f"Sequence length: {self.config.sequence_length}")
            logging.error(f"Original targets length: {len(targets)}")
            raise ValueError(
                f"Length mismatch: predictions ({len(predictions)}) != targets_aligned ({len(targets_aligned)}). "
                f"Sequence length: {self.config.sequence_length}"
            )
        
        # Shift predictions from [0,1,2] to [-1,0,1] to match target classes
        predictions = predictions - 1
        
        # Calculate metrics
        accuracy = accuracy_score(targets_aligned, predictions)
        balanced_accuracy = balanced_accuracy_score(targets_aligned, predictions)
        precision = precision_score(targets_aligned, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets_aligned, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets_aligned, predictions, average='weighted', zero_division=0)
        
        # Generate classification report with proper labels
        class_report = classification_report(
            targets_aligned,
            predictions,
            labels=[-1, 0, 1],  # Explicitly specify the labels
            target_names=['Down (-1)', 'Neutral (0)', 'Up (1)'],
            zero_division=0
        )
        
        # Generate confusion matrix with proper labels
        conf_matrix = confusion_matrix(targets_aligned, predictions, labels=[-1, 0, 1])
        
        # Log metrics
        logging.info(f"\nEvaluation metrics:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        
        logging.info("\nClassification Report:")
        logging.info("\n" + class_report)
        
        logging.info("\nConfusion Matrix:")
        logging.info("\n" + str(conf_matrix))
        
        if return_metrics:
            return {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        else:
            return balanced_accuracy
    
    def generate_signals(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            data: DataFrame with features
            target_column: Optional column to predict (not used in this implementation)
            
        Returns:
            DataFrame with signals and probabilities
        """
        # Make predictions
        predictions, probabilities = self.predict(data, target_column)
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=data.index[self.config.sequence_length:])
        signals['prediction'] = predictions
        signals['probability'] = np.max(probabilities, axis=1)  # Highest class probability
        
        # Generate signals based on predictions
        signals['signal'] = predictions - 1  # Convert 0,1,2 to -1,0,1
        
        return signals
    
    def save_model(self, path: str) -> None:
        """Save the model"""
        save_transformer(self.model, path)
    
    def load_model(self, path: str) -> None:
        """Load a saved model"""
        self.model = load_transformer(self.model, path)
    
    def get_attention_visualization(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Get attention weights for visualization
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with processed attention weights
        """
        if self.attention_weights is None:
            return {}
            
        return visualize_attention(
            self.attention_weights,
            self.config.sequence_length,
            feature_names
        ) 