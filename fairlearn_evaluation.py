#!/usr/bin/env python3
"""
Fairlearn-based fairness evaluation for LSTM models.
Computes demographic parity, equalized odds, and other fairness metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from lstm_timeseries import LSTMModel
from train_lstm import LSTMTrainer, prepare_data_for_training
import warnings
warnings.filterwarnings('ignore')

# Fairlearn imports
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    MetricFrame
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

class FairlearnEvaluator:
    """
    Comprehensive fairness evaluation using fairlearn.
    """
    
    def __init__(self, data_path='macroeconomic_data_with_fairness_groups.csv'):
        """
        Initialize the fairlearn evaluator.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset with fairness groups
        """
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.group_columns = ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']
        print(f"✅ Loaded fairness dataset: {self.data.shape}")
        print(f"📊 Available groups: {self.group_columns}")
    
    def prepare_binary_classification_data(self, model, scaler, group_column='Economic_Group', 
                                         threshold_method='median', sequence_length=12):
        """
        Prepare data for binary classification fairness evaluation.
        
        Parameters:
        -----------
        model : LSTMModel
            Trained LSTM model
        scaler : MinMaxScaler
            Fitted scaler
        group_column : str
            Group column to evaluate
        threshold_method : str
            Method to create binary labels ('median', 'mean', 'custom')
        sequence_length : int
            Length of input sequences
            
        Returns:
        --------
        tuple
            (y_true_binary, y_pred_binary, sensitive_features, group_data)
        """
        print(f"🔧 Preparing binary classification data for {group_column}...")
        
        # Get unique groups
        groups = self.data[group_column].unique()
        all_predictions = []
        all_targets = []
        all_sensitive_features = []
        all_group_data = []
        
        for group in groups:
            print(f"  📊 Processing {group}...")
            
            # Filter data for this group
            group_data = self.data[self.data[group_column] == group].copy()
            
            if len(group_data) < sequence_length + 1:
                print(f"    ⚠️  Skipping {group} - insufficient data")
                continue
            
            # Prepare sequences
            feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
            scaled_data = scaler.transform(group_data[feature_columns])
            
            sequences = []
            targets = []
            
            for i in range(len(scaled_data) - sequence_length):
                seq = scaled_data[i:i + sequence_length]
                target = scaled_data[i + sequence_length, 2]  # Target column
                sequences.append(seq)
                targets.append(target)
            
            if len(sequences) == 0:
                continue
            
            # Make predictions
            X = torch.FloatTensor(sequences)
            model.eval()
            with torch.no_grad():
                predictions = model(X).squeeze()
            
            # Convert back to original scale
            predictions_original = scaler.inverse_transform(
                np.zeros((len(predictions), 3))
            )
            predictions_original[:, 2] = predictions.numpy()
            predictions_original = scaler.inverse_transform(predictions_original)[:, 2]
            
            targets_original = scaler.inverse_transform(
                np.zeros((len(targets), 3))
            )
            targets_original[:, 2] = targets
            targets_original = scaler.inverse_transform(targets_original)[:, 2]
            
            # Create binary labels based on threshold
            if threshold_method == 'median':
                threshold = np.median(targets_original)
            elif threshold_method == 'mean':
                threshold = np.mean(targets_original)
            else:
                threshold = 3.0  # Custom threshold for delinquency rate
            
            y_true_binary = (targets_original > threshold).astype(int)
            y_pred_binary = (predictions_original > threshold).astype(int)
            
            # Store results
            all_predictions.extend(y_pred_binary)
            all_targets.extend(y_true_binary)
            all_sensitive_features.extend([group] * len(y_pred_binary))
            all_group_data.extend(group_data.iloc[sequence_length:].values.tolist())
            
            print(f"    ✅ {group}: {len(y_pred_binary)} samples, threshold={threshold:.2f}")
        
        return (np.array(all_targets), np.array(all_predictions), 
                np.array(all_sensitive_features), np.array(all_group_data))
    
    def compute_fairness_metrics(self, y_true, y_pred, sensitive_features, group_column):
        """
        Compute comprehensive fairness metrics using fairlearn.
        
        Parameters:
        -----------
        y_true : np.array
            True binary labels
        y_pred : np.array
            Predicted binary labels
        sensitive_features : np.array
            Sensitive feature values (group labels)
        group_column : str
            Name of the group column
            
        Returns:
        --------
        dict
            Comprehensive fairness metrics
        """
        print(f"📊 Computing fairness metrics for {group_column}...")
        
        # Create MetricFrame for comprehensive analysis
        metrics = {
            'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean(),
            'selection_rate': selection_rate,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
        }
        
        # Handle case where some metrics might not be available
        try:
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features
            )
        except Exception as e:
            print(f"    ⚠️  MetricFrame creation failed: {e}")
            # Fallback to basic metrics
            metric_frame = None
        
        # Compute specific fairness metrics
        fairness_metrics = {
            'demographic_parity_difference': demographic_parity_difference(
                y_true, y_pred, sensitive_features=sensitive_features
            ),
            'demographic_parity_ratio': demographic_parity_ratio(
                y_true, y_pred, sensitive_features=sensitive_features
            ),
            'equalized_odds_difference': equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive_features
            ),
            'equalized_odds_ratio': equalized_odds_ratio(
                y_true, y_pred, sensitive_features=sensitive_features
            ),
            'metric_frame': metric_frame,
            'group_metrics': metric_frame.by_group.to_dict() if metric_frame is not None else {}
        }
        
        return fairness_metrics
    
    def plot_fairness_metrics(self, fairness_metrics, group_column, save_plot=True):
        """
        Plot fairness metrics and group performance.
        
        Parameters:
        -----------
        fairness_metrics : dict
            Fairness metrics from compute_fairness_metrics
        group_column : str
            Group column name
        save_plot : bool
            Whether to save the plot
        """
        print(f"📈 Creating fairness visualization for {group_column}...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Fairlearn Fairness Evaluation: {group_column}', fontsize=16, fontweight='bold')
        
        # Plot 1: Demographic Parity
        dp_diff = fairness_metrics['demographic_parity_difference']
        dp_ratio = fairness_metrics['demographic_parity_ratio']
        
        axes[0, 0].bar(['Demographic Parity\nDifference', 'Demographic Parity\nRatio'], 
                       [dp_diff, dp_ratio], color=['red', 'blue'], alpha=0.7)
        axes[0, 0].set_title('Demographic Parity Metrics')
        axes[0, 0].set_ylabel('Metric Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        axes[0, 0].text(0, dp_diff + 0.01, f'{dp_diff:.4f}', ha='center', va='bottom')
        axes[0, 0].text(1, dp_ratio + 0.01, f'{dp_ratio:.4f}', ha='center', va='bottom')
        
        # Plot 2: Equalized Odds
        eo_diff = fairness_metrics['equalized_odds_difference']
        eo_ratio = fairness_metrics['equalized_odds_ratio']
        
        axes[0, 1].bar(['Equalized Odds\nDifference', 'Equalized Odds\nRatio'], 
                       [eo_diff, eo_ratio], color=['green', 'orange'], alpha=0.7)
        axes[0, 1].set_title('Equalized Odds Metrics')
        axes[0, 1].set_ylabel('Metric Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        axes[0, 1].text(0, eo_diff + 0.01, f'{eo_diff:.4f}', ha='center', va='bottom')
        axes[0, 1].text(1, eo_ratio + 0.01, f'{eo_ratio:.4f}', ha='center', va='bottom')
        
        # Plot 3: Selection Rate by Group
        group_metrics = fairness_metrics['group_metrics']
        groups = list(group_metrics['selection_rate'].keys())
        selection_rates = list(group_metrics['selection_rate'].values())
        
        axes[0, 2].bar(groups, selection_rates, alpha=0.7, color=['red', 'blue', 'green', 'orange'][:len(groups)])
        axes[0, 2].set_title('Selection Rate by Group')
        axes[0, 2].set_ylabel('Selection Rate')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: True Positive Rate by Group
        tpr_rates = list(group_metrics['true_positive_rate'].values())
        axes[1, 0].bar(groups, tpr_rates, alpha=0.7, color=['red', 'blue', 'green', 'orange'][:len(groups)])
        axes[1, 0].set_title('True Positive Rate by Group')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: False Positive Rate by Group
        fpr_rates = list(group_metrics['false_positive_rate'].values())
        axes[1, 1].bar(groups, fpr_rates, alpha=0.7, color=['red', 'blue', 'green', 'orange'][:len(groups)])
        axes[1, 1].set_title('False Positive Rate by Group')
        axes[1, 1].set_ylabel('False Positive Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Accuracy by Group
        accuracy_rates = list(group_metrics['accuracy'].values())
        axes[1, 2].bar(groups, accuracy_rates, alpha=0.7, color=['red', 'blue', 'green', 'orange'][:len(groups)])
        axes[1, 2].set_title('Accuracy by Group')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'fairlearn_fairness_{group_column.replace(" ", "_").lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Fairlearn fairness plot saved as '{filename}'")
        
        plt.show()
    
    def comprehensive_fairlearn_evaluation(self, model, scaler, target_column='Personal Loan Delinquency Rate'):
        """
        Perform comprehensive fairlearn evaluation across all groups.
        
        Parameters:
        -----------
        model : LSTMModel
            Trained LSTM model
        scaler : MinMaxScaler
            Fitted scaler
        target_column : str
            Target column name
            
        Returns:
        --------
        dict
            Comprehensive fairlearn results
        """
        print("🚀 Comprehensive Fairlearn Fairness Evaluation")
        print("="*60)
        
        all_results = {}
        
        for group_column in self.group_columns:
            print(f"\n📊 Evaluating {group_column}...")
            
            try:
                # Prepare binary classification data
                y_true, y_pred, sensitive_features, group_data = self.prepare_binary_classification_data(
                    model, scaler, group_column=group_column
                )
                
                if len(y_true) == 0:
                    print(f"  ⚠️  No valid data for {group_column}")
                    continue
                
                # Compute fairness metrics
                fairness_metrics = self.compute_fairness_metrics(
                    y_true, y_pred, sensitive_features, group_column
                )
                
                # Store results
                all_results[group_column] = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'sensitive_features': sensitive_features,
                    'fairness_metrics': fairness_metrics,
                    'n_samples': len(y_true)
                }
                
                # Plot results
                self.plot_fairness_metrics(fairness_metrics, group_column)
                
                # Print summary
                print(f"  📋 Fairness Summary for {group_column}:")
                print(f"    Demographic Parity Difference: {fairness_metrics['demographic_parity_difference']:.4f}")
                print(f"    Demographic Parity Ratio: {fairness_metrics['demographic_parity_ratio']:.4f}")
                print(f"    Equalized Odds Difference: {fairness_metrics['equalized_odds_difference']:.4f}")
                print(f"    Equalized Odds Ratio: {fairness_metrics['equalized_odds_ratio']:.4f}")
                print(f"    Total Samples: {len(y_true)}")
                
            except Exception as e:
                print(f"  ❌ Error evaluating {group_column}: {e}")
                continue
        
        return all_results
    
    def save_fairlearn_report(self, all_results, filename='fairlearn_evaluation_report.json'):
        """
        Save comprehensive fairlearn evaluation report.
        
        Parameters:
        -----------
        all_results : dict
            Results from comprehensive_fairlearn_evaluation
        filename : str
            Output filename
        """
        import json
        
        # Prepare report data
        report = {
            'evaluation_date': str(pd.Timestamp.now()),
            'dataset_shape': self.data.shape,
            'group_columns': self.group_columns,
            'results': {}
        }
        
        for group_column, results in all_results.items():
            # Extract key metrics for JSON serialization
            fairness_metrics = results['fairness_metrics']
            report['results'][group_column] = {
                'n_samples': results['n_samples'],
                'demographic_parity_difference': fairness_metrics['demographic_parity_difference'],
                'demographic_parity_ratio': fairness_metrics['demographic_parity_ratio'],
                'equalized_odds_difference': fairness_metrics['equalized_odds_difference'],
                'equalized_odds_ratio': fairness_metrics['equalized_odds_ratio'],
                'group_metrics': fairness_metrics['group_metrics']
            }
        
        # Save report
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Fairlearn report saved as '{filename}'")
    
    def print_fairness_interpretation(self, all_results):
        """
        Print interpretation of fairness metrics.
        
        Parameters:
        -----------
        all_results : dict
            Results from comprehensive_fairlearn_evaluation
        """
        print("\n" + "="*60)
        print("📋 FAIRNESS METRICS INTERPRETATION")
        print("="*60)
        
        for group_column, results in all_results.items():
            metrics = results['fairness_metrics']
            
            print(f"\n🔍 {group_column}:")
            print("-" * 40)
            
            # Demographic Parity
            dp_diff = metrics['demographic_parity_difference']
            dp_ratio = metrics['demographic_parity_ratio']
            
            print(f"📊 Demographic Parity:")
            print(f"  Difference: {dp_diff:.4f}")
            print(f"  Ratio: {dp_ratio:.4f}")
            
            if abs(dp_diff) < 0.1:
                print("  ✅ Good demographic parity (difference < 0.1)")
            else:
                print("  ⚠️  Potential demographic parity issues")
            
            # Equalized Odds
            eo_diff = metrics['equalized_odds_difference']
            eo_ratio = metrics['equalized_odds_ratio']
            
            print(f"\n📊 Equalized Odds:")
            print(f"  Difference: {eo_diff:.4f}")
            print(f"  Ratio: {eo_ratio:.4f}")
            
            if abs(eo_diff) < 0.1:
                print("  ✅ Good equalized odds (difference < 0.1)")
            else:
                print("  ⚠️  Potential equalized odds issues")
            
            # Group-specific metrics
            group_metrics = metrics['group_metrics']
            print(f"\n📊 Group Performance:")
            for group, metrics_dict in group_metrics.items():
                print(f"  {group}:")
                print(f"    Selection Rate: {metrics_dict['selection_rate']:.4f}")
                print(f"    True Positive Rate: {metrics_dict['true_positive_rate']:.4f}")
                print(f"    False Positive Rate: {metrics_dict['false_positive_rate']:.4f}")
                print(f"    Accuracy: {metrics_dict['accuracy']:.4f}")

def load_trained_model(model_path='best_lstm_model.pth', data_info=None):
    """
    Load a trained LSTM model.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    data_info : dict
        Model architecture information
        
    Returns:
    --------
    tuple
        (model, data_info)
    """
    try:
        # Load model
        if data_info is None:
            # Default architecture
            data_info = {
                'input_size': 3,
                'output_size': 1,
                'hidden_size': 128,
                'num_layers': 2
            }
        
        model = LSTMModel(
            input_size=data_info['input_size'],
            hidden_size=data_info['hidden_size'],
            num_layers=data_info['num_layers'],
            output_size=data_info['output_size']
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        print(f"✅ Loaded trained model from {model_path}")
        return model, data_info
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def main():
    """
    Main function to run fairlearn evaluation.
    """
    print("🚀 Fairlearn LSTM Model Fairness Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = FairlearnEvaluator()
    
    # Load trained model (you can modify this path)
    model, data_info = load_trained_model()
    
    if model is None:
        print("⚠️  No trained model found. Please train a model first or provide the correct path.")
        return
    
    # Prepare data for scaling
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    scaler = MinMaxScaler()
    scaler.fit(evaluator.data[feature_columns])
    
    # Run comprehensive fairlearn evaluation
    all_results = evaluator.comprehensive_fairlearn_evaluation(model, scaler)
    
    # Save report
    evaluator.save_fairlearn_report(all_results)
    
    # Print interpretation
    evaluator.print_fairness_interpretation(all_results)
    
    print("\n" + "="*60)
    print("✅ Fairlearn fairness evaluation completed!")
    print("="*60)

if __name__ == "__main__":
    main() 