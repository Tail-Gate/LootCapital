import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils.stgnn_visualization import (
    plot_attention_heatmap,
    plot_feature_importance,
    plot_temporal_importance,
    plot_spatial_importance,
    visualize_explanation
)

class TestSTGNNVisualization(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.num_nodes = 3
        self.seq_len = 5
        self.num_features = 4
        
        # Create sample attention weights
        self.attention_weights = {
            'layer_0_temporal': np.random.rand(1, self.num_nodes, self.seq_len),
            'layer_1_temporal': np.random.rand(1, self.num_nodes, self.seq_len)
        }
        
        # Create sample importance scores
        self.feature_importance = np.random.rand(self.num_features)
        self.feature_importance = self.feature_importance / self.feature_importance.sum()
        
        self.temporal_importance = np.random.rand(self.seq_len)
        self.temporal_importance = self.temporal_importance / self.temporal_importance.sum()
        
        self.spatial_importance = np.random.rand(self.num_nodes)
        self.spatial_importance = self.spatial_importance / self.spatial_importance.sum()
        
        # Create sample names
        self.feature_names = ['close', 'volume', 'rsi', 'macd']
        self.asset_names = ['BTC', 'ETH', 'SOL']
        self.time_steps = [f't-{i}' for i in range(self.seq_len)]
        
        # Create sample explanation
        self.explanation = {
            'attention_weights': self.attention_weights,
            'feature_importance': self.feature_importance,
            'temporal_importance': self.temporal_importance,
            'spatial_importance': self.spatial_importance
        }
    
    def test_plot_attention_heatmap(self):
        """Test attention heatmap plotting"""
        # Test with default parameters
        fig = plot_attention_heatmap(self.attention_weights, 0)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with custom parameters
        fig = plot_attention_heatmap(
            self.attention_weights,
            0,
            asset_names=self.asset_names,
            time_steps=self.time_steps,
            figsize=(10, 6)
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with invalid layer
        with self.assertRaises(ValueError):
            plot_attention_heatmap(self.attention_weights, 2)
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting"""
        fig = plot_feature_importance(
            self.feature_importance,
            self.feature_names
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_temporal_importance(self):
        """Test temporal importance plotting"""
        # Test with default parameters
        fig = plot_temporal_importance(self.temporal_importance)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with custom parameters
        fig = plot_temporal_importance(
            self.temporal_importance,
            time_steps=self.time_steps,
            figsize=(10, 4)
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_spatial_importance(self):
        """Test spatial importance plotting"""
        fig = plot_spatial_importance(
            self.spatial_importance,
            self.asset_names
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_visualize_explanation(self):
        """Test full explanation visualization"""
        # Test without saving
        visualize_explanation(
            self.explanation,
            self.feature_names,
            self.asset_names,
            self.time_steps
        )
        
        # Test with saving
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'explanation')
            visualize_explanation(
                self.explanation,
                self.feature_names,
                self.asset_names,
                self.time_steps,
                save_path
            )
            
            # Check if files were created
            expected_files = [
                f'{save_path}_attention_layer_0.png',
                f'{save_path}_attention_layer_1.png',
                f'{save_path}_feature_importance.png',
                f'{save_path}_temporal_importance.png',
                f'{save_path}_spatial_importance.png'
            ]
            
            for file_path in expected_files:
                self.assertTrue(os.path.exists(file_path))

if __name__ == '__main__':
    unittest.main() 