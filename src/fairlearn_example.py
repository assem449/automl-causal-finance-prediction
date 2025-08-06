#!/usr/bin/env python3
"""
Example of fairlearn fairness evaluation for LSTM models.
Demonstrates demographic parity and equalized odds metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fairlearn_evaluation import FairlearnEvaluator
from lstm_timeseries import LSTMModel
from train_lstm import LSTMTrainer, prepare_data_for_training
import torch
import warnings
warnings.filterwarnings('ignore')

def demonstrate_fairlearn_evaluation():
    """
    Demonstrate fairlearn evaluation with a simple example.
    """
    print("🚀 Fairlearn Fairness Evaluation Demonstration")
    print("="*60)
    
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
    print("\n🔧 Training a simple LSTM model for fairlearn evaluation...")
    
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
    
    # Initialize fairlearn evaluator
    evaluator = FairlearnEvaluator()
    
    # Evaluate fairness for Economic Groups (Group A vs Group B)
    print("\n🔍 Evaluating fairness across Economic Groups (Group A vs Group B)...")
    
    # Prepare binary classification data
    y_true, y_pred, sensitive_features, group_data = evaluator.prepare_binary_classification_data(
        model, scaler, group_column='Economic_Group'
    )
    
    # Compute fairness metrics
    fairness_metrics = evaluator.compute_fairness_metrics(
        y_true, y_pred, sensitive_features, 'Economic_Group'
    )
    
    # Print detailed results
    print("\n📋 Fairlearn Fairness Results:")
    print("="*50)
    
    print(f"\n📊 Demographic Parity Metrics:")
    print(f"  Difference: {fairness_metrics['demographic_parity_difference']:.4f}")
    print(f"  Ratio: {fairness_metrics['demographic_parity_ratio']:.4f}")
    
    print(f"\n📊 Equalized Odds Metrics:")
    print(f"  Difference: {fairness_metrics['equalized_odds_difference']:.4f}")
    print(f"  Ratio: {fairness_metrics['equalized_odds_ratio']:.4f}")
    
    print(f"\n📊 Group Performance Metrics:")
    group_metrics = fairness_metrics['group_metrics']
    for group, metrics in group_metrics.items():
        print(f"\n  {group}:")
        if 'selection_rate' in metrics:
            print(f"    Selection Rate: {metrics['selection_rate']:.4f}")
        if 'true_positive_rate' in metrics:
            print(f"    True Positive Rate: {metrics['true_positive_rate']:.4f}")
        if 'false_positive_rate' in metrics:
            print(f"    False Positive Rate: {metrics['false_positive_rate']:.4f}")
        if 'accuracy' in metrics:
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
        
        # Print all available metrics
        print(f"    Available metrics: {list(metrics.keys())}")
    
    # Plot results
    evaluator.plot_fairness_metrics(fairness_metrics, 'Economic_Group')
    
    return model, scaler, evaluator, fairness_metrics

def compare_group_a_vs_group_b():
    """
    Compare performance between Group A and Group B specifically.
    """
    print("\n🔍 Detailed Group A vs Group B Comparison")
    print("="*50)
    
    # Load data
    df = pd.read_csv('macroeconomic_data_with_fairness_groups.csv', index_col=0, parse_dates=True)
    
    # Focus on Economic Groups
    group_a_data = df[df['Economic_Group'] == 'Group A (High Rates)']
    group_b_data = df[df['Economic_Group'] == 'Group B (Low Rates)']
    
    print(f"\n📊 Data Distribution:")
    print(f"  Group A (High Rates): {len(group_a_data)} samples")
    print(f"  Group B (Low Rates): {len(group_b_data)} samples")
    
    # Compare characteristics
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    
    print(f"\n📊 Feature Comparison:")
    for feature in feature_columns:
        group_a_mean = group_a_data[feature].mean()
        group_b_mean = group_b_data[feature].mean()
        group_a_std = group_a_data[feature].std()
        group_b_std = group_b_data[feature].std()
        
        print(f"\n  {feature}:")
        print(f"    Group A: {group_a_mean:.4f} ± {group_a_std:.4f}")
        print(f"    Group B: {group_b_mean:.4f} ± {group_b_std:.4f}")
        print(f"    Difference: {abs(group_a_mean - group_b_mean):.4f}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Group A vs Group B: Feature Comparison', fontsize=14, fontweight='bold')
    
    for i, feature in enumerate(feature_columns):
        ax = axes[i]
        
        # Create box plots
        data_to_plot = [
            group_a_data[feature].values,
            group_b_data[feature].values
        ]
        
        bp = ax.boxplot(data_to_plot, labels=['Group A\n(High Rates)', 'Group B\n(Low Rates)'], 
                       patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{feature}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('group_a_vs_group_b_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Group comparison plot saved as 'group_a_vs_group_b_comparison.png'")
    plt.show()

def interpret_fairness_metrics(fairness_metrics):
    """
    Interpret the fairness metrics and provide recommendations.
    """
    print("\n📋 Fairness Metrics Interpretation")
    print("="*50)
    
    # Demographic Parity
    dp_diff = fairness_metrics['demographic_parity_difference']
    dp_ratio = fairness_metrics['demographic_parity_ratio']
    
    print(f"\n📊 Demographic Parity Analysis:")
    print(f"  Difference: {dp_diff:.4f}")
    print(f"  Ratio: {dp_ratio:.4f}")
    
    if abs(dp_diff) < 0.1:
        print("  ✅ Good demographic parity - selection rates are similar across groups")
    elif abs(dp_diff) < 0.2:
        print("  ⚠️  Moderate demographic parity issues - some bias in selection rates")
    else:
        print("  ❌ Significant demographic parity issues - strong bias in selection rates")
    
    # Equalized Odds
    eo_diff = fairness_metrics['equalized_odds_difference']
    eo_ratio = fairness_metrics['equalized_odds_ratio']
    
    print(f"\n📊 Equalized Odds Analysis:")
    print(f"  Difference: {eo_diff:.4f}")
    print(f"  Ratio: {eo_ratio:.4f}")
    
    if abs(eo_diff) < 0.1:
        print("  ✅ Good equalized odds - similar true/false positive rates across groups")
    elif abs(eo_diff) < 0.2:
        print("  ⚠️  Moderate equalized odds issues - some bias in prediction accuracy")
    else:
        print("  ❌ Significant equalized odds issues - strong bias in prediction accuracy")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if abs(dp_diff) > 0.1 or abs(eo_diff) > 0.1:
        print("  🔧 Consider fairness-aware training techniques:")
        print("    - Use fairlearn.reductions for training")
        print("    - Apply post-processing with ThresholdOptimizer")
        print("    - Collect more balanced training data")
        print("    - Use different thresholds for different groups")
    else:
        print("  ✅ Model shows good fairness properties")
        print("  📊 Continue monitoring fairness metrics")

def main():
    """
    Main function to run fairlearn evaluation examples.
    """
    print("🚀 Fairlearn Fairness Evaluation Examples")
    print("="*60)
    
    # Demonstrate fairlearn evaluation
    model, scaler, evaluator, fairness_metrics = demonstrate_fairlearn_evaluation()
    
    # Compare Group A vs Group B
    compare_group_a_vs_group_b()
    
    # Interpret fairness metrics
    interpret_fairness_metrics(fairness_metrics)
    
    print("\n" + "="*60)
    print("✅ Fairlearn evaluation examples completed!")
    print("="*60)
    print("📁 Files created:")
    print("  - fairlearn_fairness_economic_group.png")
    print("  - group_a_vs_group_b_comparison.png")
    print("  - Various fairness evaluation plots")

if __name__ == "__main__":
    main() 