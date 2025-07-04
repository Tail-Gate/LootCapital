import unittest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.stgnn_config import STGNNConfig
from utils.stgnn_data import STGNNDataProcessor
from utils.stgnn_trainer import STGNNTrainer
from strategies.stgnn_strategy import STGNNStrategy
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

class TestSTGNNStrategy(unittest.TestCase):
    """Test cases for STGNN strategy"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test configuration
        self.config = STGNNConfig(
            num_nodes=3,
            input_dim=5,
            hidden_dim=32,
            output_dim=1,
            num_layers=2,
            dropout=0.1,
            kernel_size=3,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=10,
            early_stopping_patience=3,
            seq_len=20,
            prediction_horizon=15,
            features=['close', 'volume', 'rsi', 'macd', 'bollinger_bands'],
            assets=['BTC', 'ETH', 'SOL'],
            confidence_threshold=0.51,
            buy_threshold=0.5,
            sell_threshold=-0.5,
            retrain_interval=3600
        )
        
        # Create mock market data
        self.market_data = MarketData()
        self.technical_indicators = TechnicalIndicators()
        
        # Create strategy instance
        self.strategy = STGNNStrategy(
            config=self.config,
            market_data=self.market_data,
            technical_indicators=self.technical_indicators
        )
        
    def test_prepare_data(self):
        """Test data preparation"""
        X, adj, y = self.strategy.prepare_data()
        
        # Check shapes
        self.assertEqual(X.shape[1], self.config.num_nodes)  # Number of assets
        self.assertEqual(X.shape[2], self.config.seq_len)  # Sequence length
        self.assertEqual(X.shape[3], self.config.input_dim)  # Number of features
        self.assertEqual(adj.shape, (self.config.num_nodes, self.config.num_nodes))
        self.assertEqual(y.shape[1], self.config.num_nodes)
        
    def test_train(self):
        """Test model training"""
        # Train model
        history = self.strategy.train()
        
        # Check training history
        self.assertIn('train_losses', history)
        self.assertIn('val_losses', history)
        self.assertTrue(len(history['train_losses']) > 0)
        self.assertTrue(len(history['val_losses']) > 0)
        
    def test_predict(self):
        """Test model prediction"""
        # Prepare test data
        X, adj, _ = self.strategy.prepare_data()
        
        # Make prediction
        predictions, confidence = self.strategy.predict(X[-1:])
        
        # Check prediction shape
        self.assertEqual(predictions.shape[0], self.config.num_nodes)
        
        # Check confidence
        self.assertIsInstance(confidence, dict)
        self.assertEqual(len(confidence), self.config.num_nodes)
        for asset in self.config.assets:
            self.assertIn(asset, confidence)
            self.assertIsInstance(confidence[asset], float)
        
    def test_generate_signals(self):
        """Test signal generation"""
        # Create test predictions
        predictions = torch.tensor([[0.8, -0.6, 0.3]])  # [1, num_nodes]
        
        # Generate signals
        signals = self.strategy.generate_signals(predictions[0])
        
        # Check signals
        self.assertEqual(signals['BTC'], 1)  # Above buy threshold
        self.assertEqual(signals['ETH'], -1)  # Below sell threshold
        self.assertEqual(signals['SOL'], 0)  # Between thresholds
        
        # Test edge cases
        edge_predictions = torch.tensor([[0.6, -0.5, 0.0]])  # Exactly at thresholds
        edge_signals = self.strategy.generate_signals(edge_predictions[0])
        self.assertEqual(edge_signals['BTC'], 1)  # At buy threshold
        self.assertEqual(edge_signals['ETH'], -1)  # At sell threshold
        self.assertEqual(edge_signals['SOL'], 0)  # At zero
        
    def test_update_positions(self):
        """Test position updates"""
        # Create test signals and prices
        signals = {'BTC': 1, 'ETH': -1, 'SOL': 0}
        current_prices = {'BTC': 50000, 'ETH': 3000, 'SOL': 100}
        
        # Update positions
        self.strategy.update_positions(signals, current_prices)
        
        # Check positions
        self.assertEqual(self.strategy.positions['BTC'], 1)
        self.assertEqual(self.strategy.positions['ETH'], -1)
        self.assertEqual(self.strategy.positions['SOL'], 0)
        
    def test_should_retrain(self):
        """Test retraining logic"""
        # Should retrain if no last trade time
        self.assertTrue(self.strategy.should_retrain(pd.Timestamp.now()))
        
        # Set last trade time
        self.strategy.last_trade_time = pd.Timestamp.now()
        
        # Should not retrain if within interval
        self.assertFalse(self.strategy.should_retrain(pd.Timestamp.now()))
        
        # Should retrain if past interval
        future_time = pd.Timestamp.now() + timedelta(seconds=self.config.retrain_interval + 1)
        self.assertTrue(self.strategy.should_retrain(future_time))
        
    def test_execute_trades(self):
        """Test trade execution"""
        # Execute trades
        trades = self.strategy.execute_trades(pd.Timestamp.now())
        
        # Check trade information
        for asset in self.config.assets:
            self.assertIn(asset, trades)
            self.assertIn('signal', trades[asset])
            self.assertIn('position', trades[asset])
            self.assertIn('price', trades[asset])
            self.assertIn('prediction', trades[asset])
            
    def test_save_load_state(self):
        """Test state saving and loading"""
        # Set some state
        self.strategy.positions = {'BTC': 1, 'ETH': -1, 'SOL': 0}
        self.strategy.last_trade_time = pd.Timestamp.now()
        
        # Save state
        self.strategy.save_state('test_state.pt')
        
        # Create new strategy instance
        new_strategy = STGNNStrategy(
            config=self.config,
            market_data=self.market_data,
            technical_indicators=self.technical_indicators
        )
        
        # Load state
        new_strategy.load_state('test_state.pt')
        
        # Check state
        self.assertEqual(new_strategy.positions, self.strategy.positions)
        self.assertEqual(new_strategy.last_trade_time, self.strategy.last_trade_time)
        
if __name__ == '__main__':
    unittest.main() 