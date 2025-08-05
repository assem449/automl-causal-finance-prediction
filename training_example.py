#!/usr/bin/env python3
"""
Simple example of using the LSTM training loop.
"""

import torch
import numpy as np
import pandas as pd
from train_lstm import LSTMTrainer, prepare_data_for_training
from lstm_timeseries import LSTMModel
import matplotlib.pyplot as plt

def simple_training_example():
    """
    Simple example of training an LSTM model.
    """
    print("🚀 Simple LSTM Training Example")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 300
    time = np.arange(n_samples)
    
    # Create synthetic time series with trend and seasonality
    trend = 0.01 * time
    seasonality = 5 * np.sin(0.1 * time)
    noise = np.random.normal(0, 0.3, n_samples)
    
    data = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'value': data,
        'trend': trend,
        'seasonality': seasonality
    })
    
    print(f"✅ Created sample data: {df.shape}")
    
    # Training parameters
    sequence_length = 10
    hidden_size = 32
    num_layers = 1
    learning_rate = 0.01
    batch_size = 16
    epochs = 30
    
    # Prepare data
    train_loader, val_loader, scaler, data_info = prepare_data_for_training(
        df, 
        sequence_length=sequence_length,
        target_column='value',
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
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = LSTMTrainer(model)
    trainer.setup_training(learning_rate=learning_rate)
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=10
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model('simple_lstm_model.pth')
    
    return trainer, history

def custom_training_example():
    """
    Example with custom training parameters.
    """
    print("\n🚀 Custom Training Example")
    print("="*50)
    
    # Load real data if available
    try:
        df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded macroeconomic data: {df.shape}")
        target_col = 'Personal Loan Delinquency Rate'
    except FileNotFoundError:
        print("⚠️  Using sample data")
        # Create multivariate sample data
        np.random.seed(42)
        n_samples = 400
        time = np.arange(n_samples)
        
        df = pd.DataFrame({
            'Feature_1': 0.01 * time + 10 * np.sin(0.1 * time) + np.random.normal(0, 0.5, n_samples),
            'Feature_2': -0.005 * time + 5 * np.cos(0.15 * time) + np.random.normal(0, 0.3, n_samples),
            'Feature_3': 0.02 * time + 2 * np.sin(0.05 * time) + np.random.normal(0, 0.7, n_samples)
        })
        target_col = 'Feature_1'
    
    # Custom training parameters
    sequence_length = 15
    hidden_size = 128
    num_layers = 3
    learning_rate = 0.0005
    batch_size = 64
    epochs = 40
    
    # Prepare data
    train_loader, val_loader, scaler, data_info = prepare_data_for_training(
        df, 
        sequence_length=sequence_length,
        target_column=target_col,
        train_split=0.75,
        batch_size=batch_size
    )
    
    # Build model
    model = LSTMModel(
        input_size=data_info['input_size'],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=data_info['output_size'],
        dropout=0.3  # Higher dropout for deeper model
    )
    
    print(f"🔧 Custom model built:")
    print(f"  Input size: {data_info['input_size']}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {num_layers}")
    print(f"  Dropout: 0.3")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer with custom settings
    trainer = LSTMTrainer(model)
    trainer.setup_training(
        learning_rate=learning_rate,
        weight_decay=1e-4  # Higher weight decay
    )
    
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
    trainer.save_model('custom_lstm_model.pth')
    
    return trainer, history

def compare_models_example():
    """
    Example comparing different model configurations.
    """
    print("\n🚀 Model Comparison Example")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    time = np.arange(n_samples)
    data = 0.01 * time + 5 * np.sin(0.1 * time) + np.random.normal(0, 0.3, n_samples)
    df = pd.DataFrame({'value': data})
    
    # Model configurations to compare
    configs = [
        {'name': 'Small', 'hidden_size': 16, 'num_layers': 1, 'dropout': 0.1},
        {'name': 'Medium', 'hidden_size': 32, 'num_layers': 2, 'dropout': 0.2},
        {'name': 'Large', 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔧 Training {config['name']} model...")
        
        # Prepare data
        train_loader, val_loader, scaler, data_info = prepare_data_for_training(
            df, 
            sequence_length=8,
            target_column='value',
            train_split=0.8,
            batch_size=16
        )
        
        # Build model
        model = LSTMModel(
            input_size=data_info['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=data_info['output_size'],
            dropout=config['dropout']
        )
        
        # Initialize trainer
        trainer = LSTMTrainer(model)
        trainer.setup_training(learning_rate=0.01)
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,
            early_stopping_patience=8
        )
        
        # Store results
        results[config['name']] = {
            'best_val_loss': trainer.best_val_loss,
            'final_val_r2': trainer.val_metrics[-1]['R2'],
            'final_val_rmse': trainer.val_metrics[-1]['RMSE'],
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"  Best Val Loss: {trainer.best_val_loss:.6f}")
        print(f"  Final Val R²: {trainer.val_metrics[-1]['R2']:.4f}")
        print(f"  Parameters: {results[config['name']]['parameters']:,}")
    
    # Print comparison
    print("\n" + "="*50)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*50)
    print(f"{'Model':<10} {'Parameters':<12} {'Val Loss':<10} {'Val R²':<8} {'Val RMSE':<10}")
    print("-" * 50)
    
    for name, result in results.items():
        print(f"{name:<10} {result['parameters']:<12,} {result['best_val_loss']:<10.6f} "
              f"{result['final_val_r2']:<8.4f} {result['final_val_rmse']:<10.4f}")
    
    return results

def main():
    """
    Main function to run all examples.
    """
    print("🚀 LSTM Training Examples")
    print("="*60)
    
    # Run simple example
    print("\n" + "="*30)
    print("Example 1: Simple Training")
    print("="*30)
    simple_training_example()
    
    # Run custom example
    print("\n" + "="*30)
    print("Example 2: Custom Training")
    print("="*30)
    custom_training_example()
    
    # Run comparison example
    print("\n" + "="*30)
    print("Example 3: Model Comparison")
    print("="*30)
    compare_models_example()
    
    print("\n" + "="*60)
    print("✅ All training examples completed!")
    print("="*60)

if __name__ == "__main__":
    main() 