#!/usr/bin/env python3
"""
Example of LSTM interpretability analysis using SHAP, LIME, and other methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lstm_interpretability import LSTMInterpretability
from lstm_timeseries import LSTMModel
from train_lstm import LSTMTrainer, prepare_data_for_training
import torch
import warnings
warnings.filterwarnings('ignore')

def demonstrate_interpretability():
    """
    Demonstrate LSTM interpretability with a simple example.
    """
    print("🚀 LSTM Interpretability Demonstration")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded data: {df.shape}")
    except FileNotFoundError:
        print("❌ Data file not found. Please ensure cleaned_macroeconomic_data.csv exists.")
        return
    
    # Define features
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    
    # Train a simple model for demonstration
    print("\n🔧 Training a simple LSTM model for interpretability analysis...")
    
    # Prepare data
    train_loader, val_loader, scaler, data_info = prepare_data_for_training(
        df[feature_columns],
        target_column='Personal Loan Delinquency Rate',
        sequence_length=12,
        train_split=0.8,
        batch_size=32
    )
    
    # Create and train model
    model = LSTMModel(
        input_size=data_info['input_size'],
        hidden_size=64,
        num_layers=2,
        output_size=data_info['output_size']
    )
    
    trainer = LSTMTrainer(model)
    trainer.setup_training(learning_rate=0.001)
    
    # Train for a few epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,  # Quick training for demo
        early_stopping_patience=5
    )
    
    print("✅ Model training completed!")
    
    # Initialize interpretability system
    interpreter = LSTMInterpretability(
        model=model,
        scaler=scaler,
        feature_names=feature_columns,
        target_name='Personal Loan Delinquency Rate'
    )
    
    # Demonstrate different interpretation methods
    print("\n🔍 Demonstrating different interpretation methods...")
    
    # 1. Gradient-based importance
    print("\n📊 1. Gradient-based Importance Analysis")
    print("-" * 40)
    
    # Prepare data for interpretation
    X_sequences, y_targets, scaled_data = interpreter.prepare_interpretation_data(
        df[feature_columns], sequence_length=12
    )
    
    # Compute gradient importance
    gradient_results = interpreter.gradient_based_importance(X_sequences, method='gradients')
    
    print("Gradient-based feature importance:")
    for i, feature in enumerate(feature_columns):
        importance = gradient_results['feature_importance'].mean(axis=0)[i]
        print(f"  {feature}: {importance:.4f}")
    
    # Plot gradient importance
    interpreter.plot_feature_importance(gradient_results, 'Gradients')
    
    # 2. SHAP importance
    print("\n📊 2. SHAP Importance Analysis")
    print("-" * 40)
    
    try:
        shap_results = interpreter.shap_importance(X_sequences[:50])  # Use subset for speed
        
        print("SHAP feature importance:")
        for i, feature in enumerate(feature_columns):
            importance = shap_results['feature_importance'].mean(axis=0)[i]
            print(f"  {feature}: {importance:.4f}")
        
        # Plot SHAP importance
        interpreter.plot_feature_importance(shap_results, 'SHAP')
        
    except Exception as e:
        print(f"  ⚠️  SHAP analysis failed: {e}")
    
    # 3. LIME importance
    print("\n📊 3. LIME Importance Analysis")
    print("-" * 40)
    
    try:
        lime_results = interpreter.lime_importance(X_sequences[:20])  # Use subset for speed
        
        print("LIME feature importance:")
        for i, feature in enumerate(feature_columns):
            importance = lime_results['feature_importance'][i]
            print(f"  {feature}: {importance:.4f}")
        
        # Plot LIME importance
        interpreter.plot_feature_importance(lime_results, 'LIME')
        
    except Exception as e:
        print(f"  ⚠️  LIME analysis failed: {e}")
    
    # 4. Attention analysis
    print("\n📊 4. Attention Analysis")
    print("-" * 40)
    
    try:
        attention_results = interpreter.attention_analysis(X_sequences[:50])
        
        print("Attention analysis completed")
        print(f"  Attention weights shape: {attention_results['attention_weights'].shape}")
        print(f"  Hidden states shape: {attention_results['hidden_states'].shape}")
        
        # Plot attention analysis
        interpreter.plot_attention_analysis(attention_results)
        
    except Exception as e:
        print(f"  ⚠️  Attention analysis failed: {e}")
    
    return interpreter, model, scaler

def compare_interpretation_methods():
    """
    Compare different interpretation methods.
    """
    print("\n🔍 Comparing Interpretation Methods")
    print("="*50)
    
    # Load data
    df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    
    # Train model
    train_loader, val_loader, scaler, data_info = prepare_data_for_training(
        df[feature_columns],
        target_column='Personal Loan Delinquency Rate',
        sequence_length=12,
        train_split=0.8,
        batch_size=32
    )
    
    model = LSTMModel(
        input_size=data_info['input_size'],
        hidden_size=64,
        num_layers=2,
        output_size=data_info['output_size']
    )
    
    trainer = LSTMTrainer(model)
    trainer.setup_training(learning_rate=0.001)
    trainer.train(train_loader, val_loader, epochs=10, early_stopping_patience=3)
    
    # Initialize interpreter
    interpreter = LSTMInterpretability(
        model=model,
        scaler=scaler,
        feature_names=feature_columns,
        target_name='Personal Loan Delinquency Rate'
    )
    
    # Prepare data
    X_sequences, y_targets, scaled_data = interpreter.prepare_interpretation_data(
        df[feature_columns], sequence_length=12
    )
    
    # Compare methods
    methods = {}
    
    # Gradient methods
    for method in ['gradients', 'integrated_gradients']:
        try:
            results = interpreter.gradient_based_importance(X_sequences, method)
            methods[method] = results['feature_importance'].mean(axis=0)
        except Exception as e:
            print(f"  ⚠️  {method} failed: {e}")
    
    # SHAP
    try:
        shap_results = interpreter.shap_importance(X_sequences[:30])
        methods['shap'] = shap_results['feature_importance'].mean(axis=0)
    except Exception as e:
        print(f"  ⚠️  SHAP failed: {e}")
    
    # LIME
    try:
        lime_results = interpreter.lime_importance(X_sequences[:10])
        methods['lime'] = lime_results['feature_importance']
    except Exception as e:
        print(f"  ⚠️  LIME failed: {e}")
    
    # Create comparison plot
    if methods:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods_list = list(methods.keys())
        feature_importance_matrix = np.array([methods[method] for method in methods_list])
        
        im = ax.imshow(feature_importance_matrix, aspect='auto', cmap='viridis')
        ax.set_title('Feature Importance Comparison Across Methods', fontsize=14, fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Interpretation Methods')
        ax.set_xticks(range(len(feature_columns)))
        ax.set_xticklabels(feature_columns, rotation=45)
        ax.set_yticks(range(len(methods_list)))
        ax.set_yticklabels(methods_list)
        
        # Add value annotations
        for i in range(len(methods_list)):
            for j in range(len(feature_columns)):
                text = ax.text(j, i, f'{feature_importance_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Importance Score')
        plt.tight_layout()
        plt.savefig('interpretation_methods_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Interpretation methods comparison plot saved as 'interpretation_methods_comparison.png'")
        plt.show()
    
    return methods

def analyze_temporal_importance():
    """
    Analyze how feature importance changes over time.
    """
    print("\n🔍 Temporal Feature Importance Analysis")
    print("="*50)
    
    # Load data
    df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    
    # Train model
    train_loader, val_loader, scaler, data_info = prepare_data_for_training(
        df[feature_columns],
        target_column='Personal Loan Delinquency Rate',
        sequence_length=12,
        train_split=0.8,
        batch_size=32
    )
    
    model = LSTMModel(
        input_size=data_info['input_size'],
        hidden_size=64,
        num_layers=2,
        output_size=data_info['output_size']
    )
    
    trainer = LSTMTrainer(model)
    trainer.setup_training(learning_rate=0.001)
    trainer.train(train_loader, val_loader, epochs=10, early_stopping_patience=3)
    
    # Initialize interpreter
    interpreter = LSTMInterpretability(
        model=model,
        scaler=scaler,
        feature_names=feature_columns,
        target_name='Personal Loan Delinquency Rate'
    )
    
    # Prepare data
    X_sequences, y_targets, scaled_data = interpreter.prepare_interpretation_data(
        df[feature_columns], sequence_length=12
    )
    
    # Compute temporal importance
    try:
        gradient_results = interpreter.gradient_based_importance(X_sequences, method='gradients')
        temporal_importance = gradient_results['feature_importance']  # Shape: (time_steps, features)
        
        # Create temporal analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Temporal Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Feature importance over time steps
        for i, feature in enumerate(feature_columns):
            axes[0, 0].plot(range(temporal_importance.shape[0]), temporal_importance[:, i], 
                           label=feature, marker='o', alpha=0.7)
        axes[0, 0].set_title('Feature Importance Over Time Steps')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Importance Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Heatmap of temporal importance
        im = axes[0, 1].imshow(temporal_importance.T, aspect='auto', cmap='viridis')
        axes[0, 1].set_title('Temporal Importance Heatmap')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Feature')
        axes[0, 1].set_yticks(range(len(feature_columns)))
        axes[0, 1].set_yticklabels(feature_columns)
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot 3: Average importance by feature
        avg_importance = temporal_importance.mean(axis=0)
        axes[1, 0].bar(feature_columns, avg_importance, alpha=0.7)
        axes[1, 0].set_title('Average Feature Importance')
        axes[1, 0].set_ylabel('Average Importance Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Importance variance by feature
        importance_variance = temporal_importance.var(axis=0)
        axes[1, 1].bar(feature_columns, importance_variance, alpha=0.7, color='orange')
        axes[1, 1].set_title('Feature Importance Variance')
        axes[1, 1].set_ylabel('Importance Variance')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('temporal_importance_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Temporal importance analysis plot saved as 'temporal_importance_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"  ⚠️  Temporal analysis failed: {e}")

def main():
    """
    Main function to run interpretability examples.
    """
    print("🚀 LSTM Interpretability Examples")
    print("="*60)
    
    # Demonstrate basic interpretability
    interpreter, model, scaler = demonstrate_interpretability()
    
    # Compare interpretation methods
    methods = compare_interpretation_methods()
    
    # Analyze temporal importance
    analyze_temporal_importance()
    
    print("\n" + "="*60)
    print("✅ LSTM interpretability examples completed!")
    print("="*60)
    print("📁 Files created:")
    print("  - feature_importance_*.png")
    print("  - attention_analysis.png")
    print("  - interpretation_methods_comparison.png")
    print("  - temporal_importance_analysis.png")
    print("  - interpretation_results.json")

if __name__ == "__main__":
    main() 