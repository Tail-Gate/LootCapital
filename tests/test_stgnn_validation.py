import unittest
import torch
import numpy as np
from datetime import datetime, timedelta
from utils.stgnn_config import STGNNConfig
from utils.stgnn_utils import STGNNModel
from utils.stgnn_data import STGNNDataProcessor
from utils.stgnn_trainer import STGNNTrainer
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

class TestSTGNNValidation(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create configuration
        self.config = STGNNConfig(
            num_nodes=3,
            input_dim=5,
            hidden_dim=64,
            output_dim=1,
            num_layers=2,
            dropout=0.1,
            kernel_size=3,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=10,
            early_stopping_patience=5,
            seq_len=10,
            prediction_horizon=15,
            features=['close', 'volume', 'rsi', 'macd', 'bbands'],
            assets=['BTC', 'ETH', 'SOL'],
            confidence_threshold=0.51,
            buy_threshold=0.6,
            sell_threshold=0.4,
            retrain_interval=24
        )
        
    # Create mock market data
        self.market_data = MarketData()
        self.technical_indicators = TechnicalIndicators()
    
        # Create data processor
        self.data_processor = STGNNDataProcessor(
            self.config,
            self.market_data,
            self.technical_indicators
        )
    
        # Create trainer
        self.trainer = STGNNTrainer(
            self.config,
            self.data_processor
        )
        
        # Create model
        self.model = STGNNModel(
            num_nodes=self.config.num_nodes,
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            kernel_size=self.config.kernel_size
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def test_model_metrics(self):
        """Test model performance metrics"""
        # Prepare test data
        features, adj_matrix, targets = self.data_processor.prepare_data()
    
    # Train model
        history = self.trainer.train()
    
    # Make predictions
        predictions = self.trainer.predict(features)
    
    # Calculate metrics
        mse = torch.mean((predictions - targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # Assert metrics are reasonable
        self.assertLess(mse, 1.0)  # MSE should be less than 1
        self.assertLess(mae, 1.0)  # MAE should be less than 1

    def test_feature_importance(self):
        """Test feature importance analysis"""
        # Prepare test data
        features, adj_matrix, targets = self.data_processor.prepare_data()
        
        # Train model
        history = self.trainer.train()
        
        # Calculate feature importance using gradient-based method
        features = features.clone().detach().requires_grad_(True)  # Create a new tensor that requires gradients
        predictions = self.trainer.predict(features)  # Get predictions directly
        loss = torch.mean((predictions - targets) ** 2)
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(features.grad)
        self.assertEqual(features.grad.shape, features.shape)
        
        # Get feature importance scores
        importance_scores = torch.abs(features.grad).mean(dim=(0, 1, 2))
        
        # Assert importance scores are reasonable
        self.assertEqual(importance_scores.shape[0], self.config.input_dim)
        self.assertTrue(torch.all(importance_scores >= 0))

    def test_model_explanation(self):
        """Test model explanation capabilities"""
        # Prepare test data
        features, adj_matrix, targets = self.data_processor.prepare_data()
    
        # Train model
        history = self.trainer.train()
        
        # Get attention weights
        attention_weights = self.model.get_attention_weights(features, adj_matrix)
        
        # Get the last temporal layer's attention weights
        last_layer_key = f'layer_{self.model.num_layers-1}_temporal'
        last_layer_attention = attention_weights[last_layer_key]
        
        # Assert attention weights are reasonable
        self.assertEqual(last_layer_attention.shape[1], self.config.num_nodes)  # Check number of nodes
        self.assertTrue(torch.all(last_layer_attention >= 0))  # Check non-negativity
        self.assertTrue(torch.all(last_layer_attention <= 1))  # Check normalization

    def test_model_robustness(self):
        """Test model robustness to input perturbations"""
        # Prepare test data
        features, adj_matrix, targets = self.data_processor.prepare_data()
    
    # Train model
        history = self.trainer.train()
        
        # Add noise to features
        noise = torch.randn_like(features) * 0.1
        noisy_features = features + noise
        
        # Make predictions with original and noisy features
        original_predictions = self.trainer.predict(features)
        noisy_predictions = self.trainer.predict(noisy_features)
        
        # Calculate prediction difference
        prediction_diff = torch.mean(torch.abs(original_predictions - noisy_predictions)).item()
        
        # Assert model is robust to small perturbations
        self.assertLess(prediction_diff, 0.5)  # Predictions should not change too much
        
if __name__ == '__main__':
    unittest.main() 