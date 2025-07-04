import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd

def plot_attention_heatmap(
    attention_weights: Dict[str, np.ndarray],
    layer_idx: int,
    asset_names: Optional[List[str]] = None,
    time_steps: Optional[List[str]] = None,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot attention heatmap for a specific layer
    
    Args:
        attention_weights: Dictionary of attention weights per layer
        layer_idx: Index of the layer to visualize
        asset_names: Optional list of asset names
        time_steps: Optional list of time step labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Get attention weights for the specified layer
    layer_name = f'layer_{layer_idx}_temporal'
    if layer_name not in attention_weights:
        raise ValueError(f"Layer {layer_idx} not found in attention weights")
        
    attn = attention_weights[layer_name]
    
    # Average across batch dimension if present
    if attn.ndim > 2:
        attn = attn.mean(axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        attn,
        cmap='YlOrRd',
        xticklabels=time_steps if time_steps else 'auto',
        yticklabels=asset_names if asset_names else 'auto',
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Assets')
    ax.set_title(f'Attention Weights - Layer {layer_idx}')
    
    return fig

def plot_feature_importance(
    feature_importance: np.ndarray,
    feature_names: List[str],
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot feature importance scores
    
    Args:
        feature_importance: Array of feature importance scores
        feature_names: List of feature names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, feature_importance)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance')
    
    # Add value labels
    for i, v in enumerate(feature_importance):
        ax.text(v, i, f'{v:.2f}', va='center')
    
    return fig

def plot_temporal_importance(
    temporal_importance: np.ndarray,
    time_steps: Optional[List[str]] = None,
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Plot temporal importance scores
    
    Args:
        temporal_importance: Array of temporal importance scores
        time_steps: Optional list of time step labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create line plot
    x = np.arange(len(temporal_importance))
    ax.plot(x, temporal_importance, marker='o')
    
    # Set labels
    ax.set_xticks(x)
    ax.set_xticklabels(time_steps if time_steps else x)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Importance Score')
    ax.set_title('Temporal Importance')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def plot_spatial_importance(
    spatial_importance: np.ndarray,
    asset_names: List[str],
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot spatial importance scores
    
    Args:
        spatial_importance: Array of spatial importance scores
        asset_names: List of asset names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    y_pos = np.arange(len(asset_names))
    ax.barh(y_pos, spatial_importance)
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(asset_names)
    ax.set_xlabel('Importance Score')
    ax.set_title('Spatial Importance')
    
    # Add value labels
    for i, v in enumerate(spatial_importance):
        ax.text(v, i, f'{v:.2f}', va='center')
    
    return fig

def visualize_explanation(
    explanation: Dict[str, np.ndarray],
    feature_names: List[str],
    asset_names: List[str],
    time_steps: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create and save all explanation visualizations
    
    Args:
        explanation: Dictionary containing explanation data
        feature_names: List of feature names
        asset_names: List of asset names
        time_steps: Optional list of time step labels
        save_path: Optional path to save figures
    """
    # Plot attention heatmap for each layer
    for layer_idx in range(len(explanation['attention_weights'])):
        fig = plot_attention_heatmap(
            explanation['attention_weights'],
            layer_idx,
            asset_names,
            time_steps
        )
        if save_path:
            fig.savefig(f'{save_path}_attention_layer_{layer_idx}.png')
        plt.close(fig)
    
    # Plot feature importance
    fig = plot_feature_importance(
        explanation['feature_importance'],
        feature_names
    )
    if save_path:
        fig.savefig(f'{save_path}_feature_importance.png')
    plt.close(fig)
    
    # Plot temporal importance
    fig = plot_temporal_importance(
        explanation['temporal_importance'],
        time_steps
    )
    if save_path:
        fig.savefig(f'{save_path}_temporal_importance.png')
    plt.close(fig)
    
    # Plot spatial importance
    fig = plot_spatial_importance(
        explanation['spatial_importance'],
        asset_names
    )
    if save_path:
        fig.savefig(f'{save_path}_spatial_importance.png')
    plt.close(fig) 