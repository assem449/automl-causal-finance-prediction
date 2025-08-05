#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for LSTM time series prediction.
Tunes hidden size, number of layers, and learning rate.
"""

import optuna
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
import json
from lstm_timeseries import TimeSeriesDataset, LSTMModel
from train_lstm import LSTMTrainer, prepare_data_for_training
import warnings
warnings.filterwarnings('ignore')

class OptunaLSTMTuner:
    """
    Optuna-based hyperparameter tuner for LSTM models.
    """
    
    def __init__(self, data, target_column=None, sequence_length=12, 
                 n_trials=50, timeout=3600, study_name='lstm_optimization'):
        """
        Initialize the Optuna tuner.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Time series data
        target_column : str, optional
            Target column name
        sequence_length : int
            Length of input sequences
        n_trials : int
            Number of optimization trials
        timeout : int
            Timeout in seconds
        study_name : str
            Name for the Optuna study
        """
        self.data = data
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        
        # Prepare data once
        self.train_loader, self.val_loader, self.scaler, self.data_info = self._prepare_data()
        
        # Study results
        self.best_params = None
        self.best_value = None
        self.trial_results = []
        
    def _prepare_data(self):
        """
        Prepare data for training.
        
        Returns:
        --------
        tuple
            (train_loader, val_loader, scaler, data_info)
        """
        print("📊 Preparing data for hyperparameter optimization...")
        
        return prepare_data_for_training(
            self.data,
            sequence_length=self.sequence_length,
            target_column=self.target_column,
            train_split=0.8,
            batch_size=32
        )
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
            
        Returns:
        --------
        float
            Validation loss (objective to minimize)
        """
        # Suggest hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Additional hyperparameters
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        
        try:
            # Build model
            model = LSTMModel(
                input_size=self.data_info['input_size'],
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=self.data_info['output_size'],
                dropout=dropout
            )
            
            # Recreate data loaders with new batch size
            train_loader, val_loader, _, _ = prepare_data_for_training(
                self.data,
                sequence_length=self.sequence_length,
                target_column=self.target_column,
                train_split=0.8,
                batch_size=batch_size
            )
            
            # Initialize trainer
            trainer = LSTMTrainer(model)
            
            # Setup training with suggested parameters
            if optimizer_name == 'adam':
                trainer.setup_training(learning_rate=learning_rate, weight_decay=weight_decay)
            else:  # adamw
                trainer.setup_training(learning_rate=learning_rate, weight_decay=weight_decay)
            
            # Train model with early stopping
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=30,  # Reduced epochs for faster optimization
                early_stopping_patience=8
            )
            
            # Get best validation loss
            best_val_loss = trainer.best_val_loss
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'dropout': dropout,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'optimizer': optimizer_name,
                'best_val_loss': best_val_loss,
                'final_val_r2': trainer.val_metrics[-1]['R2'],
                'final_val_rmse': trainer.val_metrics[-1]['RMSE'],
                'parameters': sum(p.numel() for p in model.parameters())
            }
            
            self.trial_results.append(trial_result)
            
            # Report intermediate value
            trial.report(best_val_loss, step=len(trainer.val_losses))
            
            return best_val_loss
            
        except Exception as e:
            print(f"⚠️  Trial {trial.number} failed: {e}")
            return float('inf')  # Return high loss for failed trials
    
    def optimize(self):
        """
        Run the hyperparameter optimization.
        
        Returns:
        --------
        optuna.Study
            Optimized study object
        """
        print(f"🚀 Starting Optuna optimization...")
        print(f"  Trials: {self.n_trials}")
        print(f"  Timeout: {self.timeout} seconds")
        print(f"  Study name: {self.study_name}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Store best results
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        print(f"✅ Optimization completed!")
        print(f"Best validation loss: {self.best_value:.6f}")
        print(f"Best parameters: {self.best_params}")
        
        return study
    
    def train_best_model(self, epochs=100):
        """
        Train the best model found during optimization.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
            
        Returns:
        --------
        tuple
            (trainer, model, history)
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")
        
        print("🔧 Training best model with optimized parameters...")
        
        # Build best model
        model = LSTMModel(
            input_size=self.data_info['input_size'],
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            output_size=self.data_info['output_size'],
            dropout=self.best_params['dropout']
        )
        
        # Prepare data with best batch size
        train_loader, val_loader, scaler, data_info = prepare_data_for_training(
            self.data,
            sequence_length=self.sequence_length,
            target_column=self.target_column,
            train_split=0.8,
            batch_size=self.best_params['batch_size']
        )
        
        # Initialize trainer
        trainer = LSTMTrainer(model)
        trainer.setup_training(
            learning_rate=self.best_params['learning_rate'],
            weight_decay=self.best_params['weight_decay']
        )
        
        # Train best model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            early_stopping_patience=20
        )
        
        # Save best model
        trainer.save_model('best_optimized_lstm_model.pth')
        
        return trainer, model, history
    
    def plot_optimization_history(self, study, save_plots=True):
        """
        Plot optimization history and results.
        
        Parameters:
        -----------
        study : optuna.Study
            Optimized study object
        save_plots : bool
            Whether to save plots
        """
        print("📈 Creating optimization visualization...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Optuna LSTM Hyperparameter Optimization Results', fontsize=16, fontweight='bold')
        
        # 1. Optimization history
        axes[0, 0].plot(study.trials_dataframe()['value'])
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Validation Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            param_names = list(importance.keys())
            importance_values = list(importance.values())
            
            axes[0, 1].barh(param_names, importance_values)
            axes[0, 1].set_title('Parameter Importance')
            axes[0, 1].set_xlabel('Importance')
        except:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Parameter Importance')
        
        # 3. Hidden size vs loss
        hidden_sizes = [trial.params.get('hidden_size', 0) for trial in study.trials]
        losses = [trial.value for trial in study.trials if trial.value is not None]
        if len(hidden_sizes) == len(losses):
            axes[0, 2].scatter(hidden_sizes, losses, alpha=0.6)
            axes[0, 2].set_title('Hidden Size vs Loss')
            axes[0, 2].set_xlabel('Hidden Size')
            axes[0, 2].set_ylabel('Validation Loss')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Learning rate vs loss
        lrs = [trial.params.get('learning_rate', 0) for trial in study.trials]
        if len(lrs) == len(losses):
            axes[1, 0].scatter(lrs, losses, alpha=0.6)
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_title('Learning Rate vs Loss')
            axes[1, 0].set_xlabel('Learning Rate')
            axes[1, 0].set_ylabel('Validation Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Number of layers vs loss
        layers = [trial.params.get('num_layers', 0) for trial in study.trials]
        if len(layers) == len(losses):
            axes[1, 1].scatter(layers, losses, alpha=0.6)
            axes[1, 1].set_title('Number of Layers vs Loss')
            axes[1, 1].set_xlabel('Number of Layers')
            axes[1, 1].set_ylabel('Validation Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Dropout vs loss
        dropouts = [trial.params.get('dropout', 0) for trial in study.trials]
        if len(dropouts) == len(losses):
            axes[1, 2].scatter(dropouts, losses, alpha=0.6)
            axes[1, 2].set_title('Dropout vs Loss')
            axes[1, 2].set_xlabel('Dropout Rate')
            axes[1, 2].set_ylabel('Validation Loss')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('optuna_optimization_results.png', dpi=300, bbox_inches='tight')
            print("✅ Optimization results plot saved as 'optuna_optimization_results.png'")
        
        plt.show()
    
    def plot_parameter_relationships(self, study, save_plots=True):
        """
        Plot relationships between different parameters.
        
        Parameters:
        -----------
        study : optuna.Study
            Optimized study object
        save_plots : bool
            Whether to save plots
        """
        print("📊 Creating parameter relationship plots...")
        
        # Extract trial data
        trials_df = study.trials_dataframe()
        
        # Create correlation heatmap
        param_columns = ['params_hidden_size', 'params_num_layers', 'params_learning_rate', 
                        'params_dropout', 'params_weight_decay', 'params_batch_size']
        
        # Filter available columns
        available_columns = [col for col in param_columns if col in trials_df.columns]
        
        if len(available_columns) > 1:
            corr_matrix = trials_df[available_columns].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, square=True)
            plt.title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('parameter_correlations.png', dpi=300, bbox_inches='tight')
                print("✅ Parameter correlations plot saved as 'parameter_correlations.png'")
            
            plt.show()
    
    def save_optimization_results(self, study, filename='optimization_results.json'):
        """
        Save optimization results to JSON file.
        
        Parameters:
        -----------
        study : optuna.Study
            Optimized study object
        filename : str
            Output filename
        """
        results = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(study.trials),
            'trial_results': self.trial_results,
            'study_name': self.study_name
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Optimization results saved to '{filename}'")
    
    def print_optimization_summary(self, study):
        """
        Print a summary of the optimization results.
        
        Parameters:
        -----------
        study : optuna.Study
            Optimized study object
        """
        print("\n" + "="*60)
        print("📋 OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"Study name: {self.study_name}")
        print(f"Number of trials: {len(study.trials)}")
        print(f"Best validation loss: {self.best_value:.6f}")
        print(f"Best trial number: {study.best_trial.number}")
        
        print("\nBest parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Print top 5 trials
        print("\nTop 5 trials:")
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
        for i, trial in enumerate(sorted_trials[:5]):
            print(f"  {i+1}. Trial {trial.number}: {trial.value:.6f}")
            for param, value in trial.params.items():
                print(f"     {param}: {value}")
            print()

def main():
    """
    Main function to run Optuna hyperparameter optimization.
    """
    print("🚀 Optuna LSTM Hyperparameter Optimization")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded macroeconomic data: {df.shape}")
        target_column = 'Personal Loan Delinquency Rate'
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
        target_column = 'Feature_1'
        print(f"✅ Created sample data: {df.shape}")
    
    # Initialize tuner
    tuner = OptunaLSTMTuner(
        data=df,
        target_column=target_column,
        sequence_length=12,
        n_trials=20,  # Reduced for demonstration
        timeout=1800,  # 30 minutes
        study_name='lstm_macroeconomic_optimization'
    )
    
    # Run optimization
    study = tuner.optimize()
    
    # Plot results
    tuner.plot_optimization_history(study)
    tuner.plot_parameter_relationships(study)
    
    # Save results
    tuner.save_optimization_results(study)
    
    # Print summary
    tuner.print_optimization_summary(study)
    
    # Train best model
    print("\n" + "="*60)
    print("🔧 TRAINING BEST MODEL")
    print("="*60)
    
    trainer, model, history = tuner.train_best_model(epochs=50)
    
    # Final evaluation
    print("\n" + "="*60)
    print("📋 FINAL RESULTS")
    print("="*60)
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Final validation R²: {trainer.val_metrics[-1]['R2']:.4f}")
    print(f"Final validation RMSE: {trainer.val_metrics[-1]['RMSE']:.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return tuner, study, trainer

if __name__ == "__main__":
    # Run the optimization
    tuner, study, trainer = main() 