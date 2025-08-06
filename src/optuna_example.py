#!/usr/bin/env python3
"""
Simple example of Optuna hyperparameter optimization for LSTM.
"""

import optuna
import torch
import numpy as np
import pandas as pd
from optuna_lstm_tuning import OptunaLSTMTuner
import matplotlib.pyplot as plt

def quick_optimization_example():
    """
    Quick optimization example with few trials.
    """
    print("🚀 Quick Optuna Optimization Example")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 300
    time = np.arange(n_samples)
    
    # Create synthetic time series
    trend = 0.01 * time
    seasonality = 5 * np.sin(0.1 * time)
    noise = np.random.normal(0, 0.3, n_samples)
    data = trend + seasonality + noise
    
    df = pd.DataFrame({
        'value': data,
        'trend': trend,
        'seasonality': seasonality
    })
    
    print(f"✅ Created sample data: {df.shape}")
    
    # Initialize tuner with few trials for quick demo
    tuner = OptunaLSTMTuner(
        data=df,
        target_column='value',
        sequence_length=8,
        n_trials=5,  # Very few trials for quick demo
        timeout=600,  # 10 minutes
        study_name='quick_lstm_optimization'
    )
    
    # Run optimization
    study = tuner.optimize()
    
    # Print summary
    tuner.print_optimization_summary(study)
    
    return tuner, study

def parameter_importance_example():
    """
    Example showing parameter importance analysis.
    """
    print("\n🚀 Parameter Importance Example")
    print("="*50)
    
    # Create more complex sample data
    np.random.seed(42)
    n_samples = 400
    time = np.arange(n_samples)
    
    # Create multivariate time series
    df = pd.DataFrame({
        'Feature_1': 0.01 * time + 10 * np.sin(0.1 * time) + np.random.normal(0, 0.5, n_samples),
        'Feature_2': -0.005 * time + 5 * np.cos(0.15 * time) + np.random.normal(0, 0.3, n_samples),
        'Feature_3': 0.02 * time + 2 * np.sin(0.05 * time) + np.random.normal(0, 0.7, n_samples)
    })
    
    # Initialize tuner
    tuner = OptunaLSTMTuner(
        data=df,
        target_column='Feature_1',
        sequence_length=10,
        n_trials=10,  # More trials for better importance analysis
        timeout=1200,  # 20 minutes
        study_name='importance_analysis'
    )
    
    # Run optimization
    study = tuner.optimize()
    
    # Plot results
    tuner.plot_optimization_history(study)
    tuner.plot_parameter_relationships(study)
    
    # Print parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\n📊 Parameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"⚠️  Could not calculate parameter importance: {e}")
    
    return tuner, study

def custom_search_space_example():
    """
    Example with custom search space.
    """
    print("\n🚀 Custom Search Space Example")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 250
    time = np.arange(n_samples)
    data = 0.01 * time + 5 * np.sin(0.1 * time) + np.random.normal(0, 0.3, n_samples)
    df = pd.DataFrame({'value': data})
    
    # Create custom tuner with modified objective
    class CustomOptunaTuner(OptunaLSTMTuner):
        def objective(self, trial):
            # Custom search space
            hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.4)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            
            try:
                # Build model
                model = self._build_model(hidden_size, num_layers, dropout)
                
                # Train and evaluate
                trainer = self._train_model(model, learning_rate, batch_size)
                
                return trainer.best_val_loss
                
            except Exception as e:
                print(f"⚠️  Trial {trial.number} failed: {e}")
                return float('inf')
        
        def _build_model(self, hidden_size, num_layers, dropout):
            from lstm_timeseries import LSTMModel
            return LSTMModel(
                input_size=self.data_info['input_size'],
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=self.data_info['output_size'],
                dropout=dropout
            )
        
        def _train_model(self, model, learning_rate, batch_size):
            from train_lstm import LSTMTrainer, prepare_data_for_training
            
            # Prepare data
            train_loader, val_loader, _, _ = prepare_data_for_training(
                self.data,
                sequence_length=self.sequence_length,
                target_column=self.target_column,
                train_split=0.8,
                batch_size=batch_size
            )
            
            # Train
            trainer = LSTMTrainer(model)
            trainer.setup_training(learning_rate=learning_rate)
            trainer.train(train_loader, val_loader, epochs=15, early_stopping_patience=5)
            
            return trainer
    
    # Run custom optimization
    tuner = CustomOptunaTuner(
        data=df,
        target_column='value',
        sequence_length=6,
        n_trials=8,
        timeout=900,
        study_name='custom_search_space'
    )
    
    study = tuner.optimize()
    tuner.print_optimization_summary(study)
    
    return tuner, study

def compare_optimization_methods():
    """
    Compare different optimization samplers.
    """
    print("\n🚀 Optimization Methods Comparison")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    time = np.arange(n_samples)
    data = 0.01 * time + 3 * np.sin(0.1 * time) + np.random.normal(0, 0.2, n_samples)
    df = pd.DataFrame({'value': data})
    
    # Test different samplers
    samplers = {
        'TPE': optuna.samplers.TPESampler(seed=42),
        'Random': optuna.samplers.RandomSampler(seed=42),
        'Grid': optuna.samplers.GridSampler({
            'hidden_size': [32, 64],
            'num_layers': [1, 2],
            'learning_rate': [0.001, 0.01]
        })
    }
    
    results = {}
    
    for sampler_name, sampler in samplers.items():
        print(f"\n🔧 Testing {sampler_name} sampler...")
        
        # Create study with specific sampler
        study = optuna.create_study(
            direction='minimize',
            study_name=f'lstm_{sampler_name.lower()}',
            sampler=sampler
        )
        
        # Create tuner
        tuner = OptunaLSTMTuner(
            data=df,
            target_column='value',
            sequence_length=6,
            n_trials=6,  # Few trials for comparison
            timeout=600,
            study_name=f'comparison_{sampler_name.lower()}'
        )
        
        # Run optimization
        study = tuner.optimize()
        
        results[sampler_name] = {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'best_params': study.best_params
        }
        
        print(f"  Best value: {study.best_value:.6f}")
        print(f"  Trials: {len(study.trials)}")
    
    # Print comparison
    print("\n" + "="*50)
    print("📊 SAMPLER COMPARISON RESULTS")
    print("="*50)
    print(f"{'Sampler':<10} {'Best Value':<12} {'Trials':<8} {'Best Hidden Size':<18}")
    print("-" * 50)
    
    for sampler_name, result in results.items():
        best_hidden = result['best_params'].get('hidden_size', 'N/A')
        print(f"{sampler_name:<10} {result['best_value']:<12.6f} {result['n_trials']:<8} {best_hidden:<18}")
    
    return results

def main():
    """
    Main function to run all examples.
    """
    print("🚀 Optuna LSTM Optimization Examples")
    print("="*60)
    
    # Run quick example
    print("\n" + "="*30)
    print("Example 1: Quick Optimization")
    print("="*30)
    quick_optimization_example()
    
    # Run parameter importance example
    print("\n" + "="*30)
    print("Example 2: Parameter Importance")
    print("="*30)
    parameter_importance_example()
    
    # Run custom search space example
    print("\n" + "="*30)
    print("Example 3: Custom Search Space")
    print("="*30)
    custom_search_space_example()
    
    # Run comparison example
    print("\n" + "="*30)
    print("Example 4: Optimization Methods Comparison")
    print("="*30)
    compare_optimization_methods()
    
    print("\n" + "="*60)
    print("✅ All Optuna examples completed!")
    print("="*60)

if __name__ == "__main__":
    main() 