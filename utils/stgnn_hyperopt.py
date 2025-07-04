import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from typing import Dict, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import json

from utils.stgnn_utils import STGNNModel, train_stgnn, predict_stgnn
from strategies.stgnn_strategy import STGNNStrategy
from market_analysis.market_data import MarketData
from market_analysis.technical_indicators import TechnicalIndicators

def load_stgnn_data(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and prepare data for STGNN training and validation
    
    Args:
        config: Configuration dictionary containing data parameters
        
    Returns:
        Tuple of (X_train, adj_train, y_train, X_val, adj_val, y_val)
    """
    # Initialize strategy components
    market_data = MarketData()
    technical_indicators = TechnicalIndicators()
    
    # Create strategy instance
    strategy = STGNNStrategy(config, market_data, technical_indicators)
    
    # Prepare data
    X, adj, y = strategy.prepare_data()
    
    # Split into train and validation sets (80/20 split)
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    adj_train = torch.FloatTensor(adj)
    adj_val = torch.FloatTensor(adj)
    
    return X_train, adj_train, y_train, X_val, adj_val, y_val

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Validation loss
    """
    # Define hyperparameter search space
    config = {
        'assets': ['BTC/USD', 'ETH/USD', 'BNB/USD'],  # Example assets
        'features': ['price', 'volume', 'rsi', 'macd', 'bollinger', 'atr', 'adx'],
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'hidden_dim': trial.suggest_int('hidden_dim', 16, 128, step=16),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'kernel_size': trial.suggest_int('kernel_size', 2, 5),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_int('batch_size', 16, 64, step=16),
        'seq_len': trial.suggest_int('seq_len', 10, 30, step=5),
        'prediction_horizon': trial.suggest_int('prediction_horizon', 15, 15),
        'early_stopping_patience': 5
    }
    
    # Load data
    X_train, adj_train, y_train, X_val, adj_val, y_val = load_stgnn_data(config)
    
    # Create model
    model = STGNNModel(
        num_nodes=len(config['assets']),
        input_dim=len(config['features']),
        hidden_dim=config['hidden_dim'],
        output_dim=1,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        kernel_size=config['kernel_size']
    )
    
    # Setup training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create dataset and dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train model
    model = train_stgnn(
        model=model,
        dataloader=train_dataloader,
        adj=adj_train,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,  # Reduced for hyperparameter optimization
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_predictions, _ = predict_stgnn(
            model=model,
            X=X_val,
            adj=adj_val,
            device=device
        )
        val_loss = criterion(val_predictions, y_val).item()
    
    return val_loss

def main():
    """
    Main function to run hyperparameter optimization
    """
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name='stgnn_hyperopt',
        storage='sqlite:///stgnn_hyperopt.db',
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=50,
        timeout=3600,  # 1 hour timeout
        show_progress_bar=True
    )
    
    # Print results
    print('Best trial:')
    print(f'  Value: {study.best_trial.value}')
    print('  Params:')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
    
    # Save best parameters
    best_params = study.best_trial.params
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'config/stgnn_best_params_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)

if __name__ == '__main__':
    main() 