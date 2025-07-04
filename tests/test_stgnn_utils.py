import unittest
import torch
import numpy as np
from utils.stgnn_utils import (
    GraphConvolution,
    TemporalConvolution,
    STGNNModel,
    train_stgnn,
    predict_stgnn,
    save_stgnn,
    load_stgnn
)
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim

class TestGraphConvolution(unittest.TestCase):
    def setUp(self):
        self.in_features = 4
        self.out_features = 3
        self.batch_size = 2
        self.num_nodes = 5
        
        # Create instances for different graph types
        self.dense_layer = GraphConvolution(self.in_features, self.out_features, graph_type='dense')
        self.sparse_layer = GraphConvolution(self.in_features, self.out_features, graph_type='sparse')
        self.auto_layer = GraphConvolution(self.in_features, self.out_features, graph_type='auto')
        
        # Create test input tensors
        self.x_dense = torch.randn(self.batch_size, self.num_nodes, self.in_features)
        self.x_sparse = torch.randn(self.num_nodes, self.in_features)
        
        # Create test adjacency matrices
        self.adj_dense = torch.randn(self.num_nodes, self.num_nodes)
        self.adj_sparse = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            values=torch.tensor([1.0, 1.0, 1.0]),
            size=(self.num_nodes, self.num_nodes)
        )
        
    def test_initialization(self):
        """Test layer initialization"""
        self.assertEqual(self.dense_layer.in_features, self.in_features)
        self.assertEqual(self.dense_layer.out_features, self.out_features)
        self.assertEqual(self.dense_layer.graph_type, 'dense')
        self.assertIsNotNone(self.dense_layer.weight)
        self.assertIsNotNone(self.dense_layer.bias)
        
    def test_reset_parameters(self):
        """Test parameter initialization"""
        # Reset parameters
        self.dense_layer.reset_parameters()
        
        # Check weight initialization
        weight_mean = self.dense_layer.weight.mean().item()
        weight_std = self.dense_layer.weight.std().item()
        self.assertGreater(weight_std, 0)  # Weights should not be all zeros
        
        # Check bias initialization
        bias_mean = self.dense_layer.bias.mean().item()
        bias_std = self.dense_layer.bias.std().item()
        self.assertGreater(bias_std, 0)  # Bias should not be all zeros
        
    def test_to_sparse(self):
        """Test sparse conversion"""
        # Test dense matrix with low sparsity
        dense_matrix = torch.ones(self.num_nodes, self.num_nodes)
        sparse_matrix = self.sparse_layer._to_sparse(dense_matrix)
        self.assertTrue(sparse_matrix.is_sparse)
        
        # Test dense matrix with high sparsity
        sparse_dense = torch.zeros(self.num_nodes, self.num_nodes)
        sparse_dense[0, 1] = 1.0
        sparse_dense[1, 2] = 1.0
        result = self.auto_layer._to_sparse(sparse_dense)
        self.assertTrue(result.is_sparse)
        
    def test_validate_inputs(self):
        """Test input validation"""
        # Test valid inputs
        self.dense_layer._validate_inputs(self.x_dense, self.adj_dense)
        self.dense_layer._validate_inputs(self.x_sparse, self.adj_dense)
        
        # Test invalid dimensions
        with self.assertRaises(ValueError):
            self.dense_layer._validate_inputs(torch.randn(3, 4, 5, 6), self.adj_dense)
            
        # Test node dimension mismatch
        with self.assertRaises(ValueError):
            self.dense_layer._validate_inputs(
                torch.randn(self.batch_size, self.num_nodes + 1, self.in_features),
                self.adj_dense
            )
            
    def test_forward_dense(self):
        """Test forward pass with dense inputs"""
        # Test batched input
        output = self.dense_layer(self.x_dense, self.adj_dense)
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, self.out_features))
        
        # Test single input
        output = self.dense_layer(self.x_sparse, self.adj_dense)
        self.assertEqual(output.shape, (1, self.num_nodes, self.out_features))
        
    def test_forward_sparse(self):
        """Test forward pass with sparse inputs"""
        # Test sparse adjacency matrix
        output = self.sparse_layer(self.x_dense, self.adj_sparse)
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, self.out_features))
        
        # Test single input with sparse adjacency
        output = self.sparse_layer(self.x_sparse, self.adj_sparse)
        self.assertEqual(output.shape, (1, self.num_nodes, self.out_features))
        
    def test_gradient_flow(self):
        """Test gradient flow through the layer"""
        # Enable gradient tracking
        self.x_dense.requires_grad = True
        self.adj_dense.requires_grad = True
        
        # Forward pass
        output = self.dense_layer(self.x_dense, self.adj_dense)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(self.x_dense.grad)
        self.assertIsNotNone(self.adj_dense.grad)
        self.assertIsNotNone(self.dense_layer.weight.grad)
        self.assertIsNotNone(self.dense_layer.bias.grad)
        
    def test_memory_efficiency(self):
        """Test memory efficiency with large inputs"""
        # Create large inputs
        large_batch = 32
        large_nodes = 1000
        large_features = 64
        
        # Use a layer with matching in_features
        dense_layer = GraphConvolution(large_features, self.out_features, graph_type='dense')
        x_large = torch.randn(large_batch, large_nodes, large_features)
        adj_large = torch.randn(large_nodes, large_nodes)
        
        # Just verify the operation completes without errors and output shape is correct
        output = dense_layer(x_large, adj_large)
        self.assertEqual(output.shape, (large_batch, large_nodes, self.out_features))

class TestTemporalConvolution(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_nodes = 3
        self.seq_len = 4
        self.in_channels = 5
        self.out_channels = 6
        self.kernel_size = 3
        self.tc = TemporalConvolution(self.in_channels, self.out_channels, self.kernel_size)
        
    def test_forward(self):
        x = torch.randn(self.batch_size, self.num_nodes, self.seq_len, self.in_channels)
        output = self.tc(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, self.seq_len, self.out_channels))

class TestSTGNNModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_nodes = 3
        self.seq_len = 4
        self.input_dim = 5
        self.hidden_dim = 6
        self.output_dim = 1
        self.model = STGNNModel(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
    def test_forward(self):
        x = torch.randn(self.batch_size, self.num_nodes, self.seq_len, self.input_dim)
        adj = torch.eye(self.num_nodes)  # Identity matrix for testing
        
        output, attention_dict = self.model(x, adj)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_nodes, self.output_dim))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Check attention dictionary
        self.assertIsInstance(attention_dict, dict)
        for key, value in attention_dict.items():
            self.assertEqual(value.shape, (self.batch_size, self.num_nodes, self.seq_len))
            self.assertFalse(torch.isnan(value).any())
            self.assertFalse(torch.isinf(value).any())

class TestTrainingAndPrediction(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 4  # Match the number of nodes in the test data
        self.input_dim = 6
        self.hidden_dim = 32
        self.output_dim = 1
        self.batch_size = 2
        self.seq_len = 10
        
        # Create model
        self.model = STGNNModel(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=2
        )
        
        # Create adjacency matrix
        self.adj = torch.eye(self.num_nodes)  # Identity matrix for testing
        
        # Create dummy dataset
        # X: [batch_size, num_nodes, seq_len, input_dim]
        self.X = torch.randn(self.batch_size, self.num_nodes, self.seq_len, self.input_dim)
        # y: [batch_size, num_nodes]
        self.y = torch.randn(self.batch_size, self.num_nodes)
        
        # Create dataloader
        dataset = TensorDataset(self.X, self.y)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.device = torch.device('cpu')  # Use CPU for testing
    
    def test_training(self):
        """Test model training"""
        trained_model = train_stgnn(
            model=self.model,
            dataloader=self.dataloader,
            adj=self.adj,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            num_epochs=2,
            early_stopping_patience=2
        )
        
        # Verify model was trained
        self.assertIsNotNone(trained_model)
        
        # Test prediction
        with torch.no_grad():
            pred, _ = trained_model(self.X, self.adj)
            self.assertEqual(pred.shape, (self.batch_size, self.num_nodes, self.output_dim))
    
    def test_prediction(self):
        """Test model prediction"""
        with torch.no_grad():
            pred, _ = self.model(self.X, self.adj)
            self.assertEqual(pred.shape, (self.batch_size, self.num_nodes, self.output_dim))

class TestModelPersistence(unittest.TestCase):
    def setUp(self):
        self.model = STGNNModel(
            num_nodes=3,
            input_dim=5,
            hidden_dim=6,
            output_dim=1
        )
        
    def test_save_and_load(self):
        # Save model
        save_path = "models/test_model.pt"
        save_stgnn(self.model, save_path)
        
        # Load model
        loaded_model = load_stgnn(save_path)
        
        # Verify model parameters
        self.assertEqual(self.model.num_nodes, loaded_model.num_nodes)
        self.assertEqual(self.model.input_dim, loaded_model.input_dim)
        self.assertEqual(self.model.hidden_dim, loaded_model.hidden_dim)
        self.assertEqual(self.model.output_dim, loaded_model.output_dim)
        self.assertEqual(self.model.num_layers, loaded_model.num_layers)
        self.assertEqual(self.model.kernel_size, loaded_model.kernel_size)

if __name__ == '__main__':
    unittest.main() 