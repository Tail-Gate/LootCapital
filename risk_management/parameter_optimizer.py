import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import random
from collections import deque
import os
import joblib

class DQNParameterOptimizer:
    """
    Uses Deep Q-Learning to dynamically adjust strategy parameters
    based on market conditions and recent performance.
    """
    
    def __init__(
        self,
        parameter_space: Dict[str, List[float]],
        feature_size: int = 20,
        memory_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        model_path: Optional[str] = None
    ):
        """
        Initialize parameter optimizer
        
        Args:
            parameter_space: Dictionary mapping parameter names to possible values
            feature_size: Size of state feature vector
            memory_size: Size of replay memory
            batch_size: Batch size for training
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay factor
            learning_rate: Learning rate for optimizer
            model_path: Path to load pre-trained model
        """
        self.parameter_space = parameter_space
        self.parameter_names = list(parameter_space.keys())
        self.feature_size = feature_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Create action space
        self.action_space = self._create_action_space()
        self.action_size = len(self.action_space)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Q-Networks
        self.q_network = self._build_q_network().to(self.device)
        self.target_network = self._build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Current parameters
        self.current_params = self._select_default_parameters()
        self.current_state = None
        
        # Metrics
        self.rewards_history = []
        self.loss_history = []
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _create_action_space(self) -> List[Dict[str, float]]:
        """
        Create action space from parameter space
        
        Returns:
            List of parameter combinations
        """
        # Generate all combinations of parameter values
        import itertools
        
        param_values = [self.parameter_space[param] for param in self.parameter_names]
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        action_space = []
        for combo in combinations:
            action = {}
            for i, param in enumerate(self.parameter_names):
                action[param] = combo[i]
            action_space.append(action)
        
        return action_space
    
    def _select_default_parameters(self) -> Dict[str, float]:
        """
        Select default parameters (middle of each range)
        
        Returns:
            Dictionary of default parameters
        """
        default_params = {}
        for param, values in self.parameter_space.items():
            # Select middle value
            default_params[param] = values[len(values) // 2]
        
        return default_params
    
    def _build_q_network(self) -> nn.Module:
        """
        Build a Q-Network for parameter optimization
        
        Returns:
            PyTorch neural network
        """
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, action_size)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
        
        return QNetwork(self.feature_size, self.action_size)
    
    def prepare_state(self, market_data: pd.DataFrame) -> torch.Tensor:
        """
        Extract state from market data
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Tensor representation of state
        """
        # Extract features for state
        state_features = []
        
        # Feature 1: Recent returns (last 5 periods)
        if 'returns' in market_data.columns:
            recent_returns = market_data['returns'].iloc[-5:].values
            state_features.extend(recent_returns)
        else:
            # Calculate returns if not available
            close = market_data['close']
            returns = close.pct_change().iloc[-5:].values
            state_features.extend(returns)
        
        # Feature 2: Volatility features
        for window in [5, 10, 20]:
            vol = market_data['returns'].iloc[-window:].std()
            state_features.append(vol)
        
        # Feature 3: Volume features (if available)
        if 'volume' in market_data.columns:
            vol_ratio = market_data['volume'].iloc[-1] / market_data['volume'].iloc[-20:].mean()
            state_features.append(vol_ratio)
        else:
            state_features.append(0.0)  # Placeholder
        
        # Feature 4: Trend features
        for window in [5, 10, 20]:
            price_start = market_data['close'].iloc[-window]
            price_end = market_data['close'].iloc[-1]
            trend = (price_end / price_start - 1)
            state_features.append(trend)
        
        # Feature 5: Autocorrelation (mean reversion indicator)
        returns_series = market_data['returns'].iloc[-20:]
        autocorr = returns_series.autocorr(lag=1) if len(returns_series.dropna()) > 1 else 0
        state_features.append(autocorr)
        
        # Feature 6: Natural gas specific - Seasonality
        if hasattr(market_data.index, 'month'):
            month = market_data.index[-1].month
            # One-hot encode quarter
            q1 = 1 if month in [1, 2, 3] else 0
            q2 = 1 if month in [4, 5, 6] else 0
            q3 = 1 if month in [7, 8, 9] else 0
            q4 = 1 if month in [10, 11, 12] else 0
            state_features.extend([q1, q2, q3, q4])
        else:
            state_features.extend([0, 0, 0, 0])  # Placeholders
        
        # Ensure we have exactly feature_size features
        if len(state_features) < self.feature_size:
            # Pad with zeros
            state_features.extend([0] * (self.feature_size - len(state_features)))
        elif len(state_features) > self.feature_size:
            # Truncate
            state_features = state_features[:self.feature_size]
        
        # Convert to tensor
        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        
        return state_tensor
    
    def select_parameters(
        self, 
        market_data: pd.DataFrame, 
        training: bool = True
    ) -> Dict[str, float]:
        """
        Select parameters based on current state
        
        Args:
            market_data: DataFrame with market data
            training: Whether we're in training mode (allow exploration)
            
        Returns:
            Dictionary of selected parameters
        """
        # Prepare state
        state = self.prepare_state(market_data)
        self.current_state = state.to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Exploration: random action
            action_idx = random.randrange(self.action_size)
        else:
            # Exploitation: best action
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(self.current_state)
                action_idx = q_values.max(1)[1].item()
            self.q_network.train()
        
        # Get parameters for selected action
        self.current_params = self.action_space[action_idx]
        
        return self.current_params
    
    def remember(
        self, 
        state: torch.Tensor, 
        action_idx: int, 
        reward: float, 
        next_state: torch.Tensor, 
        done: bool
    ):
        """
        Store experience in replay memory
        
        Args:
            state: Current state
            action_idx: Action index
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self):
        """Train Q-Network from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.cat([b[0] for b in batch]).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_states = torch.cat([b[3] for b in batch]).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track loss
        self.loss_history.append(loss.item())
    
    def update_target_network(self):
        """Update target network from q_network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def process_reward(
        self, 
        trade_result: float, 
        market_volatility: float, 
        drawdown: float
    ) -> float:
        """
        Process the reward for a parameter set
        
        Args:
            trade_result: Result of trading period (P&L)
            market_volatility: Current market volatility
            drawdown: Maximum drawdown during period
            
        Returns:
            Normalized reward
        """
        # Base reward is the trade result
        reward = trade_result
        
        # Normalize by volatility (higher reward for same return in low volatility)
        if market_volatility > 0:
            reward = reward / market_volatility
        
        # Penalize drawdown (risk-adjusted returns)
        if drawdown > 0:
            reward = reward - (drawdown * 0.5)
        
        # Scale reward to reasonable range (-1 to 1)
        reward = np.clip(reward, -1, 1)
        
        # Track rewards
        self.rewards_history.append(reward)
        
        return reward
    
    def update_from_trade_result(
        self, 
        market_data: pd.DataFrame, 
        trade_result: float, 
        market_volatility: float, 
        drawdown: float
    ):
        """
        Update model based on trade result
        
        Args:
            market_data: New market data after trade
            trade_result: Result of trade (P&L)
            market_volatility: Market volatility during trade
            drawdown: Maximum drawdown during trade
        """
        # Calculate reward
        reward = self.process_reward(trade_result, market_volatility, drawdown)
        
        # Get next state
        next_state = self.prepare_state(market_data)
        
        # Find action index for current parameters
        action_idx = self.action_space.index(self.current_params)
        
        # Store in replay memory
        self.remember(
            self.current_state, 
            action_idx, 
            reward, 
            next_state, 
            False  # We're not done with the episode
        )
        
        # Train the network
        self.replay()
        
        # Update exploration rate
        self.decay_epsilon()
    
    def save_model(self, path: str):
        """
        Save model to disk
        
        Args:
            path: File path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save Q-Network
        torch.save(self.q_network.state_dict(), f"{path}_q_network.pt")
        
        # Save other components
        optimizer_state = {
            'parameter_space': self.parameter_space,
            'parameter_names': self.parameter_names,
            'action_space': self.action_space,
            'feature_size': self.feature_size,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'rewards_history': self.rewards_history,
            'loss_history': self.loss_history
        }
        joblib.dump(optimizer_state, path)
    
    def load_model(self, path: str):
        """
        Load model from disk
        
        Args:
            path: File path to load model from
        """
        # Load optimizer state
        optimizer_state = joblib.load(path)
        
        self.parameter_space = optimizer_state['parameter_space']
        self.parameter_names = optimizer_state['parameter_names']
        self.action_space = optimizer_state['action_space']
        self.feature_size = optimizer_state['feature_size']
        self.gamma = optimizer_state['gamma']
        self.epsilon = optimizer_state['epsilon']
        self.epsilon_end = optimizer_state['epsilon_end']
        self.epsilon_decay = optimizer_state['epsilon_decay']
        self.rewards_history = optimizer_state['rewards_history']
        self.loss_history = optimizer_state['loss_history']
        
        # Rebuild Q-Networks
        self.q_network = self._build_q_network().to(self.device)
        self.target_network = self._build_q_network().to(self.device)
        
        # Load Q-Network weights
        self.q_network.load_state_dict(torch.load(f"{path}_q_network.pt", map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)