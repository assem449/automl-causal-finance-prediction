#!/usr/bin/env python3
"""
Fairness evaluation for LSTM models across different economic groups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from lstm_timeseries import LSTMModel, TimeSeriesDataset
from train_lstm import LSTMTrainer, prepare_data_for_training
import warnings
warnings.filterwarnings('ignore')

class FairnessEvaluator:
    """
    Evaluates model fairness across different groups.
    """
    
    def __init__(self, data_path='macroeconomic_data_with_fairness_groups.csv'):
        """
        Initialize the fairness evaluator.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset with fairness groups
        """
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.group_columns = ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']
        print(f"✅ Loaded fairness dataset: {self.data.shape}")
        print(f"📊 Available groups: {self.group_columns}")
    
    def evaluate_model_by_group(self, model, scaler, target_column='Personal Loan Delinquency Rate', 
                               sequence_length=12, group_column='Economic_Group'):
        """
        Evaluate model performance for each group.
        
        Parameters:
        -----------
        model : LSTMModel
            Trained LSTM model
        scaler : MinMaxScaler
            Fitted scaler
        target_column : str
            Target column name
        sequence_length : int
            Length of input sequences
        group_column : str
            Group column to evaluate
            
        Returns:
        --------
        dict
            Performance metrics for each group
        """
        print(f"🔍 Evaluating model fairness across {group_column}...")
        
        # Get unique groups
        groups = self.data[group_column].unique()
        results = {}
        
        for group in groups:
            print(f"  📊 Evaluating {group}...")
            
            # Filter data for this group
            group_data = self.data[self.data[group_column] == group].copy()
            
            if len(group_data) < sequence_length + 1:
                print(f"    ⚠️  Skipping {group} - insufficient data ({len(group_data)} samples)")
                continue
            
            # Prepare data for this group
            try:
                # Scale the data
                scaled_data = scaler.transform(group_data[['Federal Funds Rate', 'Inflation Rate (CPI)', target_column]])
                
                # Create sequences
                sequences = []
                targets = []
                
                for i in range(len(scaled_data) - sequence_length):
                    seq = scaled_data[i:i + sequence_length]
                    target = scaled_data[i + sequence_length, 2]  # Target column index
                    sequences.append(seq)
                    targets.append(target)
                
                if len(sequences) == 0:
                    print(f"    ⚠️  Skipping {group} - no valid sequences")
                    continue
                
                # Convert to tensors
                X = torch.FloatTensor(sequences)
                y = torch.FloatTensor(targets)
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    predictions = model(X)
                    predictions = predictions.squeeze()
                
                # Convert back to original scale
                predictions_original = scaler.inverse_transform(
                    np.zeros((len(predictions), 3))
                )
                predictions_original[:, 2] = predictions.numpy()
                predictions_original = scaler.inverse_transform(predictions_original)[:, 2]
                
                targets_original = scaler.inverse_transform(
                    np.zeros((len(targets), 3))
                )
                targets_original[:, 2] = y.numpy()
                targets_original = scaler.inverse_transform(targets_original)[:, 2]
                
                # Calculate metrics
                mse = mean_squared_error(targets_original, predictions_original)
                mae = mean_absolute_error(targets_original, predictions_original)
                r2 = r2_score(targets_original, predictions_original)
                rmse = np.sqrt(mse)
                
                results[group] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'n_samples': len(sequences),
                    'mean_target': np.mean(targets_original),
                    'mean_prediction': np.mean(predictions_original)
                }
                
                print(f"    ✅ {group}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error evaluating {group}: {e}")
                continue
        
        return results
    
    def calculate_fairness_metrics(self, group_results):
        """
        Calculate fairness metrics across groups.
        
        Parameters:
        -----------
        group_results : dict
            Results from evaluate_model_by_group
            
        Returns:
        --------
        dict
            Fairness metrics
        """
        if len(group_results) < 2:
            return {}
        
        # Extract metrics
        groups = list(group_results.keys())
        r2_scores = [group_results[g]['r2'] for g in groups]
        rmse_scores = [group_results[g]['rmse'] for g in groups]
        mae_scores = [group_results[g]['mae'] for g in groups]
        
        # Calculate fairness metrics
        fairness_metrics = {
            'r2_range': max(r2_scores) - min(r2_scores),
            'r2_std': np.std(r2_scores),
            'rmse_range': max(rmse_scores) - min(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_range': max(mae_scores) - min(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_cv': np.std(r2_scores) / np.mean(r2_scores) if np.mean(r2_scores) != 0 else float('inf'),
            'rmse_cv': np.std(rmse_scores) / np.mean(rmse_scores) if np.mean(rmse_scores) != 0 else float('inf'),
            'mae_cv': np.std(mae_scores) / np.mean(mae_scores) if np.mean(mae_scores) != 0 else float('inf')
        }
        
        return fairness_metrics
    
    def plot_fairness_results(self, group_results, group_column, save_plot=True):
        """
        Plot fairness evaluation results.
        
        Parameters:
        -----------
        group_results : dict
            Results from evaluate_model_by_group
        group_column : str
            Group column name
        save_plot : bool
            Whether to save the plot
        """
        if not group_results:
            print("⚠️  No results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Fairness Evaluation: {group_column}', fontsize=16, fontweight='bold')
        
        groups = list(group_results.keys())
        metrics = ['r2', 'rmse', 'mae', 'mse']
        metric_names = ['R² Score', 'RMSE', 'MAE', 'MSE']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 2, i % 2]
            values = [group_results[g][metric] for g in groups]
            
            bars = ax.bar(groups, values, alpha=0.7, color=['red', 'blue', 'green', 'orange'][:len(groups)])
            ax.set_title(f'{name} by Group')
            ax.set_ylabel(name)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'fairness_evaluation_{group_column.replace(" ", "_").lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Fairness plot saved as '{filename}'")
        
        plt.show()
    
    def comprehensive_fairness_evaluation(self, model, scaler, target_column='Personal Loan Delinquency Rate'):
        """
        Perform comprehensive fairness evaluation across all groups.
        
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
            Comprehensive fairness results
        """
        print("🚀 Comprehensive Fairness Evaluation")
        print("="*50)
        
        all_results = {}
        fairness_summary = {}
        
        for group_column in self.group_columns:
            print(f"\n📊 Evaluating {group_column}...")
            
            # Evaluate model for this group
            group_results = self.evaluate_model_by_group(
                model, scaler, target_column, group_column=group_column
            )
            
            # Calculate fairness metrics
            fairness_metrics = self.calculate_fairness_metrics(group_results)
            
            all_results[group_column] = {
                'group_results': group_results,
                'fairness_metrics': fairness_metrics
            }
            
            # Plot results
            self.plot_fairness_results(group_results, group_column)
            
            # Print summary
            if fairness_metrics:
                print(f"  📋 Fairness Summary for {group_column}:")
                print(f"    R² Range: {fairness_metrics['r2_range']:.4f}")
                print(f"    R² CV: {fairness_metrics['r2_cv']:.4f}")
                print(f"    RMSE Range: {fairness_metrics['rmse_range']:.4f}")
                print(f"    RMSE CV: {fairness_metrics['rmse_cv']:.4f}")
            
            fairness_summary[group_column] = fairness_metrics
        
        return all_results, fairness_summary
    
    def save_fairness_report(self, all_results, filename='fairness_evaluation_report.json'):
        """
        Save comprehensive fairness evaluation report.
        
        Parameters:
        -----------
        all_results : dict
            Results from comprehensive_fairness_evaluation
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
            report['results'][group_column] = {
                'group_results': results['group_results'],
                'fairness_metrics': results['fairness_metrics']
            }
        
        # Save report
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Fairness report saved as '{filename}'")

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
        (model, scaler, data_info)
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
    Main function to run fairness evaluation.
    """
    print("🚀 LSTM Model Fairness Evaluation")
    print("="*50)
    
    # Initialize evaluator
    evaluator = FairnessEvaluator()
    
    # Load trained model (you can modify this path)
    model, data_info = load_trained_model()
    
    if model is None:
        print("⚠️  No trained model found. Please train a model first or provide the correct path.")
        return
    
    # Prepare data for scaling
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    scaler.fit(evaluator.data[feature_columns])
    
    # Run comprehensive fairness evaluation
    all_results, fairness_summary = evaluator.comprehensive_fairness_evaluation(
        model, scaler
    )
    
    # Save report
    evaluator.save_fairness_report(all_results)
    
    print("\n" + "="*50)
    print("✅ Fairness evaluation completed!")
    print("="*50)
    
    # Print overall fairness summary
    print("\n📋 Overall Fairness Summary:")
    for group_column, metrics in fairness_summary.items():
        if metrics:
            print(f"\n{group_column}:")
            print(f"  R² Coefficient of Variation: {metrics['r2_cv']:.4f}")
            print(f"  RMSE Coefficient of Variation: {metrics['rmse_cv']:.4f}")
            print(f"  R² Range: {metrics['r2_range']:.4f}")
            print(f"  RMSE Range: {metrics['rmse_range']:.4f}")

if __name__ == "__main__":
    main() 