#!/usr/bin/env python3
"""
LSTM Time Series Prediction Model using PyTorch.
Predicts the next value in a multivariate time series.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time series data.
    Creates sequences of specified length for LSTM input.
    """
    
    def __init__(self, data, sequence_length, target_column=None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Time series data of shape (n_samples, n_features)
        sequence_length : int
            Length of input sequences
        target_column : int, optional
            Index of target column. If None, predicts all columns
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.target_column = target_column
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(data) - sequence_length):
            # Input sequence
            seq = self.data[i:i + sequence_length]
            self.sequences.append(seq)
            
            # Target (next value)
            if target_column is not None:
                target = self.data[i + sequence_length, target_column]
            else:
                target = self.data[i + sequence_length]
            self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of hidden units in LSTM layers
        num_layers : int
            Number of LSTM layers
        output_size : int
            Number of output features
        dropout : float
            Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output from the sequence
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layer
        output = self.fc(lstm_out)
        
        return output

class TimeSeriesPredictor:
    """
    Complete time series prediction pipeline using LSTM.
    """
    
    def __init__(self, sequence_length=12, hidden_size=64, num_layers=2, 
                 learning_rate=0.001, batch_size=32, epochs=100):
        """
        Initialize the predictor.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        hidden_size : int
            Number of hidden units in LSTM
        num_layers : int
            Number of LSTM layers
        learning_rate : float
            Learning rate for optimization
        batch_size : int
            Batch size for training
        epochs : int
            Number of training epochs
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
    
    def prepare_data(self, data, target_column=None, train_split=0.8):
        """
        Prepare data for training and testing.
        
        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            Time series data
        target_column : str or int, optional
            Target column name or index
        train_split : float
            Proportion of data for training
            
        Returns:
        --------
        tuple
            (train_loader, test_loader, scaler)
        """
        print("📊 Preparing data...")
        
        # Determine target column index first
        if target_column is not None:
            if isinstance(target_column, str):
                # Find column index by name
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
        data_scaled = self.scaler.fit_transform(data)
        
        # Split data
        train_size = int(len(data_scaled) * train_split)
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        
        print(f"Training data: {len(train_data)} samples")
        print(f"Test data: {len(test_data)} samples")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, self.sequence_length, target_idx)
        test_dataset = TimeSeriesDataset(test_data, self.sequence_length, target_idx)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader, data_scaled
    
    def build_model(self, input_size, output_size):
        """
        Build the LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        output_size : int
            Number of output features
        """
        print("🔧 Building LSTM model...")
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=output_size
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def train_model(self, train_loader, test_loader=None):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader, optional
            Test data loader for validation
        """
        print("🚀 Training LSTM model...")
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        train_losses = []
        test_losses = []
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            if test_loader is not None:
                self.model.eval()
                test_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        test_loss += loss.item()
                
                test_loss /= len(test_loader)
                test_losses.append(test_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Test Loss: {test_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], "
                          f"Train Loss: {train_loss:.6f}")
        
        print("✅ Training completed!")
        
        return train_losses, test_losses
    
    def predict(self, data_loader):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        data_loader : DataLoader
            Data loader for prediction
            
        Returns:
        --------
        tuple
            (predictions, actual_values)
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        return np.array(predictions), np.array(actuals)
    
    def evaluate_model(self, predictions, actuals):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        predictions : numpy.ndarray
            Model predictions
        actuals : numpy.ndarray
            Actual values
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        print("📈 Evaluating model performance...")
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        
        # Calculate R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print("Model Performance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def plot_training_history(self, train_losses, test_losses=None):
        """
        Plot training history.
        
        Parameters:
        -----------
        train_losses : list
            Training losses
        test_losses : list, optional
            Test losses
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        
        if test_losses:
            plt.plot(test_losses, label='Test Loss', color='red')
        
        plt.title('Training History', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, predictions, actuals, save_plot=True):
        """
        Plot predictions vs actual values.
        
        Parameters:
        -----------
        predictions : numpy.ndarray
            Model predictions
        actuals : numpy.ndarray
            Actual values
        save_plot : bool
            Whether to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot predictions vs actuals
        plt.subplot(2, 1, 1)
        plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        plt.title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(2, 1, 2)
        residuals = actuals - predictions
        plt.plot(residuals, color='green', alpha=0.7)
        plt.title('Residuals', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('predictions_vs_actuals.png', dpi=300, bbox_inches='tight')
            print("✅ Predictions plot saved as 'predictions_vs_actuals.png'")
        
        plt.show()
    
    def save_model(self, filepath='lstm_model.pth'):
        """
        Save the trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }, filepath)
        print(f"💾 Model saved to '{filepath}'")
    
    def load_model(self, filepath='lstm_model.pth'):
        """
        Load a trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Rebuild model
        input_size = self.scaler.n_features_in_
        output_size = 1  # Assuming single target
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            output_size=output_size
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        
        print(f"✅ Model loaded from '{filepath}'")

def main():
    """
    Main function to demonstrate LSTM time series prediction.
    """
    print("🚀 LSTM Time Series Prediction")
    print("="*50)
    
    # Load the cleaned macroeconomic data
    print("📊 Loading data...")
    df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize predictor
    predictor = TimeSeriesPredictor(
        sequence_length=12,  # 12 months of history
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        epochs=50  # Reduced for demonstration
    )
    
    # Prepare data
    train_loader, test_loader, data_scaled = predictor.prepare_data(
        df, 
        target_column='Personal Loan Delinquency Rate',  # Predict delinquency rate
        train_split=0.8
    )
    
    # Build model
    input_size = df.shape[1]  # Number of features
    output_size = 1  # Single target
    model = predictor.build_model(input_size, output_size)
    
    # Train model
    train_losses, test_losses = predictor.train_model(train_loader, test_loader)
    
    # Make predictions
    predictions, actuals = predictor.predict(test_loader)
    
    # Evaluate model
    metrics = predictor.evaluate_model(predictions, actuals)
    
    # Plot results
    predictor.plot_training_history(train_losses, test_losses)
    predictor.plot_predictions(predictions, actuals)
    
    # Save model
    predictor.save_model()
    
    print("\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    print("✅ LSTM model trained successfully")
    print("✅ Predictions generated")
    print("✅ Model performance evaluated")
    print("✅ Results visualized and saved")
    
    return predictor, metrics

if __name__ == "__main__":
    # Run the LSTM time series prediction
    predictor, metrics = main() 