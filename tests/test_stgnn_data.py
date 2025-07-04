import unittest
import pandas as pd
import numpy as np
import torch
from utils.stgnn_data import STGNNDataProcessor
from utils.stgnn_config import STGNNConfig
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

class TestSTGNNDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create test configuration
        self.config = STGNNConfig(
            num_nodes=3,
            input_dim=7,
            hidden_dim=32,
            output_dim=1,
            features=['returns', 'log_returns', 'rsi', 'macd', 'volume_ma', 'volume_ratio', 'bb_width'],  # Engineered features only
            assets=['BTC', 'ETH', 'SOL']
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
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],  # Use float type
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [98.0, 99.0, 100.0, 101.0, 102.0],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        }, index=pd.date_range('2023-01-01', periods=5))
        
    def test_prepare_features(self):
        """Test feature preparation"""
        # Prepare features
        features = self.data_processor.prepare_features(self.sample_data)
        
        # Check basic properties
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.sample_data))
        self.assertEqual(len(features.columns), self.config.input_dim)
        
        # Check for missing values
        self.assertFalse(features.isna().any().any())
        self.assertFalse(np.isinf(features.values).any())
        
        # Check specific features
        self.assertIn('returns', features.columns)
        self.assertIn('log_returns', features.columns)
        self.assertIn('volume', features.columns)
        
        # Check numerical properties
        self.assertTrue(np.all(np.isfinite(features['returns'])))
        self.assertTrue(np.all(np.isfinite(features['log_returns'])))
        self.assertTrue(np.all(features['volume'] >= 0))
        
    def test_prepare_features_edge_cases(self):
        """Test feature preparation with edge cases"""
        # Test with empty data
        empty_data = pd.DataFrame(columns=['close', 'high', 'low', 'volume'])
        features = self.data_processor.prepare_features(empty_data)
        self.assertEqual(len(features), 0)
        self.assertEqual(len(features.columns), self.config.input_dim)
        
        # Test with single row
        single_row = pd.DataFrame({
            'close': [100.0],
            'high': [102.0],
            'low': [98.0],
            'volume': [1000.0]
        }, index=[pd.Timestamp('2023-01-01')])
        features = self.data_processor.prepare_features(single_row)
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features.columns), self.config.input_dim)
        
        # Test with missing values
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[data_with_nulls.index[0], 'close'] = np.nan
        features = self.data_processor.prepare_features(data_with_nulls)
        self.assertFalse(features.isna().any().any())
        
        # Test with infinite values
        data_with_inf = self.sample_data.copy()
        data_with_inf.loc[data_with_inf.index[0], 'close'] = float('inf')  # Use float('inf') instead of np.inf
        features = self.data_processor.prepare_features(data_with_inf)
        self.assertFalse(np.isinf(features.values).any())
        
    def test_prepare_data(self):
        """Test full data preparation"""
        # Prepare data
        X, adj, y = self.data_processor.prepare_data()
        
        # Check shapes
        self.assertEqual(X.shape[1], self.config.num_nodes)  # Number of assets
        self.assertEqual(X.shape[2], self.config.seq_len)  # Sequence length
        self.assertEqual(X.shape[3], self.config.input_dim)  # Number of features
        self.assertEqual(adj.shape, (self.config.num_nodes, self.config.num_nodes))
        self.assertEqual(y.shape[1], self.config.num_nodes)
        
        # Check for missing values
        self.assertFalse(torch.isnan(X).any())
        self.assertFalse(torch.isinf(X).any())
        self.assertFalse(torch.isnan(y).any())
        self.assertFalse(torch.isinf(y).any())
        
        # Check adjacency matrix properties
        self.assertTrue(torch.all(adj >= 0))  # Non-negative weights
        self.assertTrue(torch.all(adj <= 1))  # Normalized weights
        
if __name__ == '__main__':
    unittest.main() 