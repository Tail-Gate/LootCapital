from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TransformerMomentumStrategy:
    def validate_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """
        Validate the Transformer momentum model with specific handling for zero predictions.
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            
        Returns:
            Dictionary of validation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        try:
            # Get predictions
            probabilities, _ = self.predict(features)
            predictions = np.argmax(probabilities, axis=1) - 1  # Convert to -1, 0, 1
            
            # Align targets with predictions (due to sequence length offset)
            targets_aligned = targets.iloc[self.config.sequence_length-1:].values
            
            # Convert predictions to binary (1 for trade, 0 for no trade)
            trade_signals = np.abs(predictions)
            actual_trades = np.abs(targets_aligned) > 0
            
            # Check if we have any trades
            if np.sum(trade_signals) == 0:
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'directional_accuracy': 0.0,
                    'information_coefficient': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Calculate accuracy metrics
            accuracy = accuracy_score(trade_signals, actual_trades)
            precision = precision_score(trade_signals, actual_trades, zero_division=0)
            recall = recall_score(trade_signals, actual_trades, zero_division=0)
            f1 = f1_score(trade_signals, actual_trades, zero_division=0)
            
            # Calculate directional accuracy
            directional_accuracy = np.mean(np.sign(predictions) == np.sign(targets_aligned))
            
            # Calculate information coefficient (IC)
            ic = np.corrcoef(predictions, targets_aligned)[0, 1]
            
            # Calculate trading metrics
            strategy_returns = predictions * targets_aligned
            total_return = np.prod(1 + strategy_returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            
            # Calculate volatility
            volatility = np.std(strategy_returns) * np.sqrt(252)
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_returns = strategy_returns - risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
            
            # Calculate Sortino ratio
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1
            sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std != 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (running_max - cumulative_returns) / running_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'directional_accuracy': directional_accuracy,
                'information_coefficient': ic,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            print(f"Error in validate_model: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'directional_accuracy': 0.0,
                'information_coefficient': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0
            } 