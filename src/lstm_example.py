#!/usr/bin/env python3
"""
Simple example of using the LSTM time series prediction model.
"""

import torch
import numpy as np
import pandas as pd
from lstm_timeseries import TimeSeriesPredictor, LSTMModel, TimeSeriesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def create_sample_data(n_samples=1000, n_features=3):
    """
    Create sample time series data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of time steps
    n_features : int
        Number of features
        
    Returns:
    --------
    pandas.DataFrame
        Sample time series data
    """
    print("📊 Creating sample time series data...")
    
    # Generate synthetic time series with some patterns
    np.random.seed(42)
    
    # Create base trends
    time = np.arange(n_samples)
    trend1 = 0.01 * time + 10 * np.sin(0.1 * time)  # Trend with seasonality
    trend2 = -0.005 * time + 5 * np.cos(0.15 * time)  # Different trend
    trend3 = 0.02 * time + 2 * np.sin(0.05 * time)  # Another trend
    
    # Add noise
    noise1 = np.random.normal(0, 0.5, n_samples)
    noise2 = np.random.normal(0, 0.3, n_samples)
    noise3 = np.random.normal(0, 0.7, n_samples)
    
    # Combine trends and noise
    feature1 = trend1 + noise1
    feature2 = trend2 + noise2
    feature3 = trend3 + noise3
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature_1': feature1,
        'Feature_2': feature2,
        'Feature_3': feature3
    })
    
    print(f"✅ Created sample data: {df.shape}")
    return df

def simple_lstm_example():
    """
    Simple example of LSTM time series prediction.
    """
    print("🚀 Simple LSTM Time Series Example")
    print("="*50)
    
    # Create sample data
    df = create_sample_data(n_samples=500, n_features=3)
    
    # Initialize predictor with smaller parameters for quick training
    predictor = TimeSeriesPredictor(
        sequence_length=10,  # 10 time steps of history
        hidden_size=32,      # Smaller hidden size
        num_layers=1,        # Single layer
        learning_rate=0.01,  # Higher learning rate
        batch_size=16,       # Smaller batch size
        epochs=20            # Fewer epochs for quick demo
    )
    
    # Prepare data
    train_loader, test_loader, data_scaled = predictor.prepare_data(
        df, 
        target_column='Feature_1',  # Predict first feature
        train_split=0.8
    )
    
    # Build model
    input_size = df.shape[1]
    output_size = 1
    model = predictor.build_model(input_size, output_size)
    
    # Train model
    print("\n🔧 Training model...")
    train_losses, test_losses = predictor.train_model(train_loader, test_loader)
    
    # Make predictions
    print("\n📊 Making predictions...")
    predictions, actuals = predictor.predict(test_loader)
    
    # Evaluate model
    metrics = predictor.evaluate_model(predictions, actuals)
    
    # Plot results
    predictor.plot_training_history(train_losses, test_losses)
    predictor.plot_predictions(predictions, actuals)
    
    return predictor, metrics

def multivariate_prediction_example():
    """
    Example of multivariate time series prediction.
    """
    print("\n🚀 Multivariate LSTM Example")
    print("="*50)
    
    # Load the cleaned macroeconomic data
    df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
    
    # Initialize predictor
    predictor = TimeSeriesPredictor(
        sequence_length=12,  # 12 months of history
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        epochs=30  # Moderate number of epochs
    )
    
    # Prepare data - predict all variables
    train_loader, test_loader, data_scaled = predictor.prepare_data(
        df, 
        target_column=None,  # Predict all columns
        train_split=0.8
    )
    
    # Build model for multivariate prediction
    input_size = df.shape[1]
    output_size = df.shape[1]  # Predict all features
    model = predictor.build_model(input_size, output_size)
    
    # Train model
    print("\n🔧 Training multivariate model...")
    train_losses, test_losses = predictor.train_model(train_loader, test_loader)
    
    # Make predictions
    print("\n📊 Making multivariate predictions...")
    predictions, actuals = predictor.predict(test_loader)
    
    # Evaluate model
    metrics = predictor.evaluate_model(predictions, actuals)
    
    # Plot results for each variable
    plot_multivariate_results(predictions, actuals, df.columns)
    
    return predictor, metrics

def plot_multivariate_results(predictions, actuals, column_names):
    """
    Plot results for multivariate prediction.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Model predictions
    actuals : numpy.ndarray
        Actual values
    column_names : list
        Names of the columns
    """
    print("\n📈 Creating multivariate results plots...")
    
    n_features = len(column_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3*n_features))
    fig.suptitle('Multivariate Time Series Predictions', fontsize=16, fontweight='bold')
    
    for i, col_name in enumerate(column_names):
        ax = axes[i] if n_features > 1 else axes
        
        # Plot actual vs predicted for this feature
        ax.plot(actuals[:, i], label='Actual', color='blue', alpha=0.7)
        ax.plot(predictions[:, i], label='Predicted', color='red', alpha=0.7)
        ax.set_title(f'{col_name}', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multivariate_predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Multivariate predictions plot saved as 'multivariate_predictions.png'")
    plt.show()

def custom_sequence_example():
    """
    Example of creating custom sequences for LSTM input.
    """
    print("\n🚀 Custom Sequence Example")
    print("="*50)
    
    # Create sample data
    data = np.random.randn(100, 3)  # 100 time steps, 3 features
    
    # Create custom dataset
    sequence_length = 5
    dataset = TimeSeriesDataset(data, sequence_length, target_column=0)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Input shape: {dataset[0][0].shape}")
    print(f"Target shape: {dataset[0][1].shape}")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get a batch
    for batch_x, batch_y in dataloader:
        print(f"\nBatch input shape: {batch_x.shape}")
        print(f"Batch target shape: {batch_y.shape}")
        print(f"Input sequence example:\n{batch_x[0]}")
        print(f"Target example: {batch_y[0]}")
        break

def model_architecture_example():
    """
    Example of different LSTM architectures.
    """
    print("\n🚀 LSTM Architecture Examples")
    print("="*50)
    
    # Example 1: Simple LSTM
    print("1. Simple LSTM (1 layer, 32 hidden units):")
    model1 = LSTMModel(input_size=3, hidden_size=32, num_layers=1, output_size=1)
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Example 2: Deep LSTM
    print("2. Deep LSTM (3 layers, 64 hidden units):")
    model2 = LSTMModel(input_size=3, hidden_size=64, num_layers=3, output_size=1)
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    # Example 3: Wide LSTM
    print("3. Wide LSTM (2 layers, 128 hidden units):")
    model3 = LSTMModel(input_size=3, hidden_size=128, num_layers=2, output_size=1)
    print(f"   Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    
    # Test forward pass
    print("\n4. Testing forward pass:")
    x = torch.randn(4, 10, 3)  # batch_size=4, seq_len=10, features=3
    output = model1(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

def main():
    """
    Main function to run all examples.
    """
    print("🚀 LSTM Time Series Examples")
    print("="*60)
    
    # Run examples
    print("\n" + "="*30)
    print("Example 1: Simple LSTM")
    print("="*30)
    simple_lstm_example()
    
    print("\n" + "="*30)
    print("Example 2: Custom Sequences")
    print("="*30)
    custom_sequence_example()
    
    print("\n" + "="*30)
    print("Example 3: Model Architectures")
    print("="*30)
    model_architecture_example()
    
    # Try multivariate prediction if data exists
    try:
        print("\n" + "="*30)
        print("Example 4: Multivariate Prediction")
        print("="*30)
        multivariate_prediction_example()
    except FileNotFoundError:
        print("Skipping multivariate example - cleaned data not found")
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60)

if __name__ == "__main__":
    main() 