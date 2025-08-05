#!/usr/bin/env python3
"""
Example of fairness evaluation for LSTM models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fairness_evaluation import FairnessEvaluator
from lstm_timeseries import LSTMModel
from train_lstm import LSTMTrainer, prepare_data_for_training
import torch
import warnings
warnings.filterwarnings('ignore')

def demonstrate_fairness_evaluation():
    """
    Demonstrate fairness evaluation with a simple example.
    """
    print("🚀 Fairness Evaluation Demonstration")
    print("="*50)
    
    # Load data with fairness groups
    try:
        df = pd.read_csv('macroeconomic_data_with_fairness_groups.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded dataset with fairness groups: {df.shape}")
    except FileNotFoundError:
        print("❌ Dataset with fairness groups not found. Please run create_fairness_dataset.py first.")
        return
    
    # Show group distributions
    print("\n📊 Group Distributions:")
    group_columns = ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']
    
    for group_col in group_columns:
        print(f"\n{group_col}:")
        counts = df[group_col].value_counts()
        for group, count in counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {group}: {count} samples ({percentage:.1f}%)")
    
    # Train a simple model for demonstration
    print("\n🔧 Training a simple LSTM model for fairness evaluation...")
    
    # Prepare data
    train_loader, val_loader, scaler, data_info = prepare_data_for_training(
        df[['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']],
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
        epochs=10,  # Quick training for demo
        early_stopping_patience=5
    )
    
    print("✅ Model training completed!")
    
    # Initialize fairness evaluator
    evaluator = FairnessEvaluator()
    
    # Evaluate fairness for one group type
    print("\n🔍 Evaluating fairness across Economic Groups...")
    group_results = evaluator.evaluate_model_by_group(
        model, scaler, group_column='Economic_Group'
    )
    
    # Calculate fairness metrics
    fairness_metrics = evaluator.calculate_fairness_metrics(group_results)
    
    # Print results
    print("\n📋 Fairness Evaluation Results:")
    print("="*40)
    
    for group, metrics in group_results.items():
        print(f"\n{group}:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Samples: {metrics['n_samples']}")
    
    if fairness_metrics:
        print(f"\n📊 Fairness Metrics:")
        print(f"  R² Range: {fairness_metrics['r2_range']:.4f}")
        print(f"  R² Coefficient of Variation: {fairness_metrics['r2_cv']:.4f}")
        print(f"  RMSE Range: {fairness_metrics['rmse_range']:.4f}")
        print(f"  RMSE Coefficient of Variation: {fairness_metrics['rmse_cv']:.4f}")
    
    # Plot results
    evaluator.plot_fairness_results(group_results, 'Economic_Group')
    
    return model, scaler, evaluator

def analyze_group_characteristics():
    """
    Analyze the characteristics of different groups.
    """
    print("\n🔍 Group Characteristics Analysis")
    print("="*50)
    
    # Load data
    df = pd.read_csv('macroeconomic_data_with_fairness_groups.csv', index_col=0, parse_dates=True)
    
    # Analyze each group type
    group_columns = ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    
    for group_col in group_columns:
        print(f"\n📊 {group_col} Analysis:")
        print("-" * 40)
        
        # Group statistics
        group_stats = df.groupby(group_col)[feature_columns].agg(['mean', 'std']).round(4)
        
        for group in df[group_col].unique():
            print(f"\n{group}:")
            group_data = df[df[group_col] == group]
            
            for feature in feature_columns:
                mean_val = group_data[feature].mean()
                std_val = group_data[feature].std()
                print(f"  {feature}: {mean_val:.4f} ± {std_val:.4f}")

def create_fairness_visualization():
    """
    Create a comprehensive fairness visualization.
    """
    print("\n📈 Creating fairness visualization...")
    
    # Load data
    df = pd.read_csv('macroeconomic_data_with_fairness_groups.csv', index_col=0, parse_dates=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fairness Groups Analysis', fontsize=16, fontweight='bold')
    
    group_columns = ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']
    
    for i, group_col in enumerate(group_columns):
        ax = axes[i // 3, i % 3]
        
        # Create box plot for delinquency rate by group
        groups = df[group_col].unique()
        data_to_plot = [df[df[group_col] == group]['Personal Loan Delinquency Rate'].values for group in groups]
        
        bp = ax.boxplot(data_to_plot, labels=groups, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(groups)]):
            patch.set_facecolor(color)
        
        ax.set_title(f'Delinquency Rate by {group_col}')
        ax.set_ylabel('Personal Loan Delinquency Rate (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Remove the last subplot if not needed
    if len(group_columns) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig('fairness_groups_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Fairness groups analysis plot saved as 'fairness_groups_analysis.png'")
    plt.show()

def main():
    """
    Main function to run fairness evaluation examples.
    """
    print("🚀 Fairness Evaluation Examples")
    print("="*60)
    
    # Demonstrate fairness evaluation
    model, scaler, evaluator = demonstrate_fairness_evaluation()
    
    # Analyze group characteristics
    analyze_group_characteristics()
    
    # Create visualization
    create_fairness_visualization()
    
    print("\n" + "="*60)
    print("✅ Fairness evaluation examples completed!")
    print("="*60)
    print("📁 Files created:")
    print("  - fairness_evaluation_economic_group.png")
    print("  - fairness_groups_analysis.png")
    print("  - Various fairness evaluation plots")

if __name__ == "__main__":
    main() 