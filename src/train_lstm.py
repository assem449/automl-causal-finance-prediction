#!/usr/bin/env python3
"""
Comprehensive training loop for LSTM time series prediction.
Includes MSE loss, Adam optimizer, validation, and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os
from lstm_timeseries import TimeSeriesDataset, LSTMModel
import warnings
warnings.filterwarnings('ignore')

class LSTMTrainer:
    """
    Comprehensive LSTM trainer with validation and monitoring.
    """
    
    def __init__(self, model, device='auto'):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        model : LSTMModel
            LSTM model to train
        device : str
            Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def setup_training(self, learning_rate=0.001, weight_decay=1e-5):
        """
        Setup loss function and optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for Adam optimizer
        weight_decay : float
            Weight decay for regularization
        """
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)")
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
            
        Returns:
        --------
        tuple
            (epoch_loss, epoch_metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate loss and predictions
            total_loss += loss.item()
            all_predictions.extend(output.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch.
        
        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader
            
        Returns:
        --------
        tuple
            (epoch_loss, epoch_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Accumulate loss and predictions
                total_loss += loss.item()
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate evaluation metrics.
        
        Parameters:
        -----------
        predictions : list
            Model predictions
        targets : list
            Actual targets
            
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Flatten if needed
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(targets.shape) > 1:
            targets = targets.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=20):
        """
        Complete training loop with validation and early stopping.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        epochs : int
            Number of training epochs
        early_stopping_patience : int
            Number of epochs to wait before early stopping
            
        Returns:
        --------
        dict
            Training history
        """
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Early stopping
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"\nEpoch [{epoch+1}/{epochs}] ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Train R²: {train_metrics['R2']:.4f}, Val R²: {val_metrics['R2']:.4f}")
            print(f"  Train RMSE: {train_metrics['RMSE']:.4f}, Val RMSE: {val_metrics['RMSE']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ Loaded best model (Val Loss: {self.best_val_loss:.6f})")
        
        print("✅ Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss
        }
    
    def plot_training_history(self, save_plot=True):
        """
        Plot training history.
        
        Parameters:
        -----------
        save_plot : bool
            Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² curves
        train_r2 = [m['R2'] for m in self.train_metrics]
        val_r2 = [m['R2'] for m in self.val_metrics]
        axes[0, 1].plot(train_r2, label='Train R²', color='blue')
        axes[0, 1].plot(val_r2, label='Val R²', color='red')
        axes[0, 1].set_title('R² Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE curves
        train_rmse = [m['RMSE'] for m in self.train_metrics]
        val_rmse = [m['RMSE'] for m in self.val_metrics]
        axes[1, 0].plot(train_rmse, label='Train RMSE', color='blue')
        axes[1, 0].plot(val_rmse, label='Val RMSE', color='red')
        axes[1, 0].set_title('RMSE Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAE curves
        train_mae = [m['MAE'] for m in self.train_metrics]
        val_mae = [m['MAE'] for m in self.val_metrics]
        axes[1, 1].plot(train_mae, label='Train MAE', color='blue')
        axes[1, 1].plot(val_mae, label='Val MAE', color='red')
        axes[1, 1].set_title('MAE Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
            print("✅ Training history plot saved as 'lstm_training_history.png'")
        
        plt.show()
    
    def save_model(self, filepath='best_lstm_model.pth'):
        """
        Save the best model.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        torch.save({
            'model_state_dict': self.best_model_state,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': self.model.fc.out_features
            }
        }, filepath)
        print(f"💾 Best model saved to '{filepath}'")
    
    def load_model(self, filepath='best_lstm_model.pth'):
        """
        Load a saved model.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Model loaded from '{filepath}'")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

def prepare_data_for_training(data, sequence_length=12, target_column=None, 
                            train_split=0.8, batch_size=32, scaler=None):
    """
    Prepare data for training with proper scaling and splitting.
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.ndarray
        Time series data
    sequence_length : int
        Length of input sequences
    target_column : str or int, optional
        Target column name or index
    train_split : float
        Proportion of data for training
    batch_size : int
        Batch size for data loaders
    scaler : MinMaxScaler, optional
        Pre-fitted scaler
        
    Returns:
    --------
    tuple
        (train_loader, val_loader, scaler, data_info)
    """
    print("📊 Preparing data for training...")
    
    # Determine target column index
    if target_column is not None:
        if isinstance(target_column, str):
            if isinstance(data, pd.DataFrame):
                target_idx = data.columns.get_loc(target_column)
            else:
                raise ValueError("Cannot use string column name with numpy array")
        else:
            target_idx = target_column
    else:
        target_idx = None
    
    # Convert to numpy if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Scale the data
    if scaler is None:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)
    
    # Split data
    train_size = int(len(data_scaled) * train_split)
    train_data = data_scaled[:train_size]
    val_data = data_scaled[train_size:]
    
    print(f"Training data: {len(train_data)} samples")
    print(f"Validation data: {len(val_data)} samples")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length, target_idx)
    val_dataset = TimeSeriesDataset(val_data, sequence_length, target_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Data info
    data_info = {
        'input_size': data.shape[1],
        'output_size': 1 if target_idx is not None else data.shape[1],
        'sequence_length': sequence_length,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }
    
    return train_loader, val_loader, scaler, data_info

def main():
    """
    Main training function with example usage.
    """
    print("🚀 LSTM Training with Validation")
    print("="*50)
    
    # Load data
    try:
        df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded macroeconomic data: {df.shape}")
    except FileNotFoundError:
        print("⚠️  Using sample data (cleaned_macroeconomic_data.csv not found)")
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        time = np.arange(n_samples)
        df = pd.DataFrame({
            'Feature_1': 0.01 * time + 10 * np.sin(0.1 * time) + np.random.normal(0, 0.5, n_samples),
            'Feature_2': -0.005 * time + 5 * np.cos(0.15 * time) + np.random.normal(0, 0.3, n_samples),
            'Feature_3': 0.02 * time + 2 * np.sin(0.05 * time) + np.random.normal(0, 0.7, n_samples)
        })
        print(f"✅ Created sample data: {df.shape}")
    
    # Training parameters
    sequence_length = 12
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    batch_size = 32
    epochs = 50
    
    # Prepare data
    train_loader, val_loader, scaler, data_info = prepare_data_for_training(
        df, 
        sequence_length=sequence_length,
        target_column='Personal Loan Delinquency Rate' if 'Personal Loan Delinquency Rate' in df.columns else 'Feature_1',
        train_split=0.8,
        batch_size=batch_size
    )
    
    # Build model
    model = LSTMModel(
        input_size=data_info['input_size'],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=data_info['output_size']
    )
    
    print(f"🔧 Model built:")
    print(f"  Input size: {data_info['input_size']}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {num_layers}")
    print(f"  Output size: {data_info['output_size']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = LSTMTrainer(model)
    trainer.setup_training(learning_rate=learning_rate)
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=15
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model()
    
    # Final evaluation
    print("\n" + "="*50)
    print("📋 FINAL RESULTS")
    print("="*50)
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Final validation R²: {trainer.val_metrics[-1]['R2']:.4f}")
    print(f"Final validation RMSE: {trainer.val_metrics[-1]['RMSE']:.4f}")
    print(f"Training completed in {len(trainer.train_losses)} epochs")
    
    return trainer, history

if __name__ == "__main__":
    # Run the training
    trainer, history = main() 