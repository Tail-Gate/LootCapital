import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class WeightedFocalCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
        if class_weights is None:
            # Default: make minority classes 100x more important
            self.class_weights = torch.tensor([100.0, 1.0, 100.0])  # [-1, 0, 1] -> [0, 1, 2]
        else:
            self.class_weights = torch.tensor(class_weights)
    
    def forward(self, inputs, targets):
        # Move weights to same device as inputs
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # Calculate weighted cross entropy with focal loss
        ce_loss = nn.functional.cross_entropy(inputs, targets.long(), 
                                             weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.attention = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*directions)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim*directions)
        output = self.fc(context)  # (batch, output_dim)
        return output, attn_weights

def train_lstm(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 20,
    early_stopping_patience: int = 10,
    clip_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 4,
    model_path: str = None,  # Changed from checkpoint_path
    periodic_checkpoint_base_path: str = None,  # New parameter
    checkpoint_frequency: int = 5  # Save every N epochs
) -> nn.Module:
    """
    Train LSTM model with early stopping and gradient clipping.
    Args:
        model: LSTM model to train
        dataloader: DataLoader with training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        early_stopping_patience: Number of epochs to wait for improvement
        clip_grad_norm: Maximum gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating
        model_path: Path to save the best model
        periodic_checkpoint_base_path: Base path for periodic checkpoints
        checkpoint_frequency: How often to save checkpoints (in epochs)
    Returns:
        Trained model
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    final_epoch = 0  # Track the final epoch we reach
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    # Load checkpoint if exists
    start_epoch = 0
    # Prioritize loading from the designated 'best model' path (model_path)
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' not in checkpoint:
                raise KeyError("'model_state_dict' not found in checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                # If we've already reached or exceeded num_epochs, return the model
                # Note: start_epoch is 1-based (epoch + 1 in checkpoints), so we compare with num_epochs + 1
                if start_epoch >= num_epochs + 1:
                    print(f"Model already trained for {start_epoch} epochs (target: {num_epochs}). Returning loaded model.")
                    return model
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
            print(f"Loaded BEST model from {model_path} (epoch {start_epoch}, best_loss {best_loss:.4f})")
        except Exception as e:
            print(f"Error loading BEST model from {model_path}: {e}")
            print("Attempting to load from latest periodic checkpoint...")
            # Fallback to periodic checkpoints if main model_path fails or doesn't exist
            if periodic_checkpoint_base_path:
                latest_periodic_checkpoint = None
                latest_epoch = -1
                checkpoint_dir_for_periodic = os.path.dirname(periodic_checkpoint_base_path)
                base_name_for_periodic = os.path.basename(periodic_checkpoint_base_path)
                
                if os.path.exists(checkpoint_dir_for_periodic):
                    for f_name in os.listdir(checkpoint_dir_for_periodic):
                        if f_name.startswith(base_name_for_periodic + ".epoch_"):
                            try:
                                epoch_num = int(f_name.split('_')[-1])
                                if epoch_num > latest_epoch:
                                    latest_epoch = epoch_num
                                    latest_periodic_checkpoint = os.path.join(checkpoint_dir_for_periodic, f_name)
                            except ValueError:
                                continue
                    
                    if latest_periodic_checkpoint and os.path.exists(latest_periodic_checkpoint):
                        try:
                            checkpoint = torch.load(latest_periodic_checkpoint, map_location=device)
                            if 'model_state_dict' not in checkpoint:
                                raise KeyError("'model_state_dict' not found in periodic checkpoint")
                            model.load_state_dict(checkpoint['model_state_dict'])
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            start_epoch = checkpoint['epoch']
                            # If we've already reached or exceeded num_epochs, return the model
                            # Note: start_epoch is 1-based (epoch + 1 in checkpoints), so we compare with num_epochs + 1
                            if start_epoch >= num_epochs + 1:
                                print(f"Model already trained for {start_epoch} epochs (target: {num_epochs}). Returning loaded model.")
                                return model
                            best_loss = checkpoint['best_loss']
                            print(f"Loaded PERIODIC checkpoint from epoch {start_epoch}")
                            
                            # Clean up older periodic checkpoints after successfully loading a valid one
                            for f_name in os.listdir(checkpoint_dir_for_periodic):
                                if f_name.startswith(base_name_for_periodic + ".epoch_"):
                                    try:
                                        epoch_num = int(f_name.split('_')[-1])
                                        if epoch_num < latest_epoch:
                                            os.remove(os.path.join(checkpoint_dir_for_periodic, f_name))
                                            print(f"Cleaned up older checkpoint: {f_name}")
                                    except (ValueError, Exception) as e:
                                        print(f"Warning: Could not process checkpoint {f_name}: {e}")
                        except Exception as e:
                            print(f"Error loading PERIODIC checkpoint: {e}")
                            print("Starting training from scratch")
                    else:
                        print("No valid periodic checkpoints found. Starting training from scratch")
                else:
                    print("Periodic checkpoint directory does not exist. Starting training from scratch")
            else:
                print("No periodic checkpoint base path specified. Starting training from scratch")
    else:
        print("No main model path specified or model not found. Starting training from scratch")
    
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"):
        epoch_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for i, (batch_X, batch_y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            # Move batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs, _ = model(batch_X)  # Unpack the tuple, ignore attention weights
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps  # Scale back up for reporting
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        final_epoch = epoch + 1  # Update final epoch (1-based)
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            
            # Save best model
            if model_path:
                try:
                    checkpoint = {
                        'epoch': final_epoch,  # Save 1-based epoch number
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                    }
                    # Save to temporary file first
                    temp_path = f"{model_path}.tmp"
                    torch.save(checkpoint, temp_path)
                    # If save was successful, rename to final path
                    os.replace(temp_path, model_path)
                    print(f"\nSaved BEST checkpoint to {model_path} at epoch {final_epoch}")
                except Exception as e:
                    print(f"\nWarning: Error saving checkpoint: {e}")
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
        else:
            patience_counter += 1
            
        # Save periodic checkpoint
        if periodic_checkpoint_base_path and final_epoch % checkpoint_frequency == 0:
            try:
                checkpoint_path_periodic = f"{periodic_checkpoint_base_path}.epoch_{final_epoch}"
                checkpoint = {
                    'epoch': final_epoch,  # Save 1-based epoch number
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }
                # Save to temporary file first
                temp_path = f"{checkpoint_path_periodic}.tmp"
                torch.save(checkpoint, temp_path)
                # If save was successful, rename to final path
                os.replace(temp_path, checkpoint_path_periodic)
                print(f"\nSaved periodic checkpoint at epoch {final_epoch}")
            except Exception as e:
                print(f"\nWarning: Error saving periodic checkpoint: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            
        # Update progress bar description
        tqdm.write(f"Epoch {final_epoch}/{num_epochs} - Loss: {avg_epoch_loss:.4f} (Best: {best_loss:.4f}, Patience: {patience_counter}/{early_stopping_patience})")
        
        if patience_counter >= early_stopping_patience:
            tqdm.write(f"Early stopping triggered at epoch {final_epoch}")
            break
    
    # Save final model state regardless of whether it's the best
    if model_path:
        try:
            checkpoint = {
                'epoch': final_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }
            # Save to temporary file first
            temp_path = f"{model_path}.tmp"
            torch.save(checkpoint, temp_path)
            # If save was successful, rename to final path
            os.replace(temp_path, model_path)
            print(f"\nSaved FINAL model state to {model_path} at epoch {final_epoch}")
        except Exception as e:
            print(f"\nWarning: Error saving final model state: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    return model  # Return the trained model

def predict_lstm(model, X, device):
    model.eval()
    with torch.no_grad():
        # Move input to device
        X = X.to(device)
        output, attn_weights = model(X)
        # Move outputs back to CPU for numpy conversion
        output = output.cpu().numpy()
        attn_weights = attn_weights.cpu().numpy()
    return output, attn_weights

def save_lstm(model, path, save_full_checkpoint=False, epoch=None, optimizer=None, best_loss=None):
    """Save the LSTM model to disk
    
    Args:
        model: The LSTM model to save
        path: Path to save the model
        save_full_checkpoint: Whether to save a full checkpoint with additional info
        epoch: Current epoch number (required if save_full_checkpoint is True)
        optimizer: Optimizer state (required if save_full_checkpoint is True)
        best_loss: Best loss value (required if save_full_checkpoint is True)
    """
    try:
        # Create directory if it doesn't exist
        model_dir = os.path.dirname(path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")
        
        # Save model
        if save_full_checkpoint:
            if epoch is None or optimizer is None or best_loss is None:
                raise ValueError("epoch, optimizer, and best_loss are required for full checkpoint")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, path)
            print(f"Full checkpoint saved to: {path}")
        else:
            torch.save(model.state_dict(), path)
            print(f"Model state dict saved to: {path}")
        
        # Verify the file was created
        if not os.path.exists(path):
            raise RuntimeError(f"Model file was not created at {path}")
            
        print(f"File size: {os.path.getsize(path)} bytes")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

def load_lstm(model_class, path, *args, **kwargs):
    """Load an LSTM model from disk
    
    Args:
        model_class: The LSTM model class to instantiate
        path: Path to the saved model
        *args: Arguments to pass to model_class constructor
        **kwargs: Keyword arguments to pass to model_class constructor
        
    Returns:
        Loaded model and checkpoint info (if available)
    """
    try:
        # Create model instance
        model = model_class(*args, **kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Check if it's a full checkpoint or just state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_info = {
                'epoch': checkpoint.get('epoch'),
                'best_loss': checkpoint.get('best_loss'),
                'optimizer_state_dict': checkpoint.get('optimizer_state_dict')
            }
            print(f"Loaded full checkpoint from epoch {checkpoint_info['epoch']}")
        else:
            # Just state dict
            model.load_state_dict(checkpoint)
            checkpoint_info = None
            print("Loaded model state dict")
        
        model.eval()
        return model, checkpoint_info
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# TODO: Add advanced attention visualization utilities
# TODO: Add dataset preparation and batching utilities 