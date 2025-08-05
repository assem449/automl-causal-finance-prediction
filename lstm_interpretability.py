#!/usr/bin/env python3
"""
Comprehensive interpretability system for LSTM time series models.
Uses SHAP, LIME, and other methods to explain feature importance across time steps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import shap
import lime
import lime.lime_tabular
from lime import submodular_pick
import warnings
warnings.filterwarnings('ignore')

# Import our LSTM components
from lstm_timeseries import LSTMModel, TimeSeriesDataset
from train_lstm import LSTMTrainer, prepare_data_for_training

class LSTMInterpretability:
    """
    Comprehensive interpretability system for LSTM time series models.
    """
    
    def __init__(self, model, scaler, feature_names=None, target_name='target'):
        """
        Initialize the interpretability system.
        
        Parameters:
        -----------
        model : LSTMModel
            Trained LSTM model
        scaler : MinMaxScaler
            Fitted scaler
        feature_names : list
            Names of input features
        target_name : str
            Name of target variable
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(model.input_size)]
        self.target_name = target_name
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"✅ Initialized LSTM interpretability system")
        print(f"  Model input size: {model.input_size}")
        print(f"  Model hidden size: {model.hidden_size}")
        print(f"  Model layers: {model.num_layers}")
        print(f"  Features: {self.feature_names}")
    
    def prepare_interpretation_data(self, data, sequence_length=12):
        """
        Prepare data for interpretation methods.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Time series data
        sequence_length : int
            Length of input sequences
            
        Returns:
        --------
        tuple
            (X_sequences, y_targets, scaled_data)
        """
        print(f"🔧 Preparing data for interpretation...")
        
        # Scale the data
        scaled_data = self.scaler.transform(data)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - sequence_length):
            seq = scaled_data[i:i + sequence_length]
            target = scaled_data[i + sequence_length, -1]  # Last feature is target
            sequences.append(seq)
            targets.append(target)
        
        X_sequences = np.array(sequences)
        y_targets = np.array(targets)
        
        print(f"  Created {len(sequences)} sequences")
        print(f"  Sequence shape: {X_sequences.shape}")
        print(f"  Target shape: {y_targets.shape}")
        
        return X_sequences, y_targets, scaled_data
    
    def gradient_based_importance(self, X_sequences, method='integrated_gradients'):
        """
        Compute gradient-based feature importance.
        
        Parameters:
        -----------
        X_sequences : np.array
            Input sequences
        method : str
            Method to use ('gradients', 'integrated_gradients', 'smoothgrad')
            
        Returns:
        --------
        dict
            Feature importance scores
        """
        print(f"🔍 Computing gradient-based importance using {method}...")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sequences)
        X_tensor.requires_grad_(True)
        
        if method == 'gradients':
            importance_scores = self._compute_gradients(X_tensor)
        elif method == 'integrated_gradients':
            importance_scores = self._compute_integrated_gradients(X_tensor)
        elif method == 'smoothgrad':
            importance_scores = self._compute_smoothgrad(X_tensor)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return importance_scores
    
    def _compute_gradients(self, X_tensor):
        """
        Compute basic gradients for feature importance.
        """
        # Forward pass
        outputs = self.model(X_tensor)
        
        # Backward pass
        outputs.backward(torch.ones_like(outputs))
        
        # Get gradients
        gradients = X_tensor.grad.abs().mean(dim=0)  # Average across batch
        
        # Convert to numpy
        importance_scores = gradients.detach().numpy()
        
        return {
            'feature_importance': importance_scores,
            'time_step_importance': importance_scores.mean(axis=1),
            'overall_importance': importance_scores.mean()
        }
    
    def _compute_integrated_gradients(self, X_tensor, steps=50):
        """
        Compute integrated gradients for feature importance.
        """
        baseline = torch.zeros_like(X_tensor)
        importance_scores = torch.zeros_like(X_tensor)
        
        for step in range(steps):
            alpha = step / steps
            interpolated = baseline + alpha * (X_tensor - baseline)
            interpolated.requires_grad_(True)
            
            outputs = self.model(interpolated)
            outputs.backward(torch.ones_like(outputs))
            
            importance_scores += interpolated.grad.abs()
        
        importance_scores = importance_scores / steps
        importance_scores = importance_scores.detach().numpy()
        
        return {
            'feature_importance': importance_scores,
            'time_step_importance': importance_scores.mean(axis=1),
            'overall_importance': importance_scores.mean()
        }
    
    def _compute_smoothgrad(self, X_tensor, noise_std=0.1, num_samples=50):
        """
        Compute SmoothGrad for feature importance.
        """
        importance_scores = torch.zeros_like(X_tensor)
        
        for _ in range(num_samples):
            # Add noise
            noise = torch.randn_like(X_tensor) * noise_std
            noisy_input = X_tensor + noise
            noisy_input.requires_grad_(True)
            
            outputs = self.model(noisy_input)
            outputs.backward(torch.ones_like(outputs))
            
            importance_scores += noisy_input.grad.abs()
        
        importance_scores = importance_scores / num_samples
        importance_scores = importance_scores.detach().numpy()
        
        return {
            'feature_importance': importance_scores,
            'time_step_importance': importance_scores.mean(axis=1),
            'overall_importance': importance_scores.mean()
        }
    
    def shap_importance(self, X_sequences, background_samples=100):
        """
        Compute SHAP values for feature importance.
        
        Parameters:
        -----------
        X_sequences : np.array
            Input sequences
        background_samples : int
            Number of background samples for SHAP
            
        Returns:
        --------
        dict
            SHAP importance scores
        """
        print(f"🔍 Computing SHAP importance...")
        
        # Flatten sequences for SHAP
        X_flat = X_sequences.reshape(X_sequences.shape[0], -1)
        
        # Create background dataset
        background_indices = np.random.choice(len(X_flat), min(background_samples, len(X_flat)), replace=False)
        background = X_flat[background_indices]
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(self._shap_predict_function, background)
        
        # Compute SHAP values for a sample
        sample_indices = np.random.choice(len(X_flat), min(50, len(X_flat)), replace=False)
        shap_values = explainer.shap_values(X_flat[sample_indices])
        
        # Reshape back to sequence format
        shap_values_reshaped = np.array(shap_values).reshape(-1, X_sequences.shape[1], X_sequences.shape[2])
        
        return {
            'shap_values': shap_values_reshaped,
            'feature_importance': np.abs(shap_values_reshaped).mean(axis=0),
            'time_step_importance': np.abs(shap_values_reshaped).mean(axis=(0, 2)),
            'overall_importance': np.abs(shap_values_reshaped).mean()
        }
    
    def _shap_predict_function(self, X_flat):
        """
        Prediction function for SHAP explainer.
        """
        # Reshape to sequence format
        X_reshaped = X_flat.reshape(-1, self.model.lstm.input_size)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_reshaped)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor.unsqueeze(0))  # Add batch dimension
        
        return predictions.numpy()
    
    def lime_importance(self, X_sequences, num_samples=10):
        """
        Compute LIME importance for feature importance.
        
        Parameters:
        -----------
        X_sequences : np.array
            Input sequences
        num_samples : int
            Number of samples to explain
            
        Returns:
        --------
        dict
            LIME importance scores
        """
        print(f"🔍 Computing LIME importance...")
        
        # Flatten sequences for LIME
        X_flat = X_sequences.reshape(X_sequences.shape[0], -1)
        
        # Create feature names for flattened data
        flat_feature_names = []
        for t in range(X_sequences.shape[1]):
            for f in self.feature_names:
                flat_feature_names.append(f"{f}_t{t}")
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_flat,
            feature_names=flat_feature_names,
            class_names=[self.target_name],
            mode='regression'
        )
        
        # Compute LIME for sample instances
        lime_importance = np.zeros((X_sequences.shape[1], X_sequences.shape[2]))
        lime_counts = np.zeros((X_sequences.shape[1], X_sequences.shape[2]))
        
        sample_indices = np.random.choice(len(X_flat), min(num_samples, len(X_flat)), replace=False)
        
        for idx in sample_indices:
            exp = explainer.explain_instance(
                X_flat[idx],
                self._lime_predict_function,
                num_features=len(flat_feature_names)
            )
            
            # Parse LIME explanation
            for feature, importance in exp.as_list():
                # Extract time step and feature from feature name
                if '_t' in feature:
                    feature_name, time_step = feature.rsplit('_t', 1)
                    time_idx = int(time_step)
                    feature_idx = self.feature_names.index(feature_name)
                    
                    lime_importance[time_idx, feature_idx] += abs(importance)
                    lime_counts[time_idx, feature_idx] += 1
        
        # Average importance scores
        lime_importance = np.divide(lime_importance, lime_counts, where=lime_counts > 0)
        
        return {
            'lime_importance': lime_importance,
            'feature_importance': lime_importance.mean(axis=0),
            'time_step_importance': lime_importance.mean(axis=1),
            'overall_importance': lime_importance.mean()
        }
    
    def _lime_predict_function(self, X_flat):
        """
        Prediction function for LIME explainer.
        """
        # Reshape to sequence format
        X_reshaped = X_flat.reshape(-1, self.model.lstm.input_size)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_reshaped)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor.unsqueeze(0))
        
        return predictions.numpy()
    
    def attention_analysis(self, X_sequences):
        """
        Analyze attention patterns in the LSTM model.
        
        Parameters:
        -----------
        X_sequences : np.array
            Input sequences
            
        Returns:
        --------
        dict
            Attention analysis results
        """
        print(f"🔍 Computing attention analysis...")
        
        # This is a simplified attention analysis
        # For more sophisticated attention, you'd need to modify the LSTM model
        
        X_tensor = torch.FloatTensor(X_sequences)
        
        # Get hidden states
        with torch.no_grad():
            # Forward pass through LSTM
            batch_size = X_tensor.size(0)
            h0 = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size).to(X_tensor.device)
            c0 = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size).to(X_tensor.device)
            
            lstm_out, (hidden, cell) = self.model.lstm(X_tensor, (h0, c0))
            
            # Compute attention weights (simplified)
            attention_weights = torch.softmax(lstm_out, dim=1)
            
            # Average attention across batch
            avg_attention = attention_weights.mean(dim=0)
        
        return {
            'attention_weights': avg_attention.detach().numpy(),
            'hidden_states': hidden.detach().numpy(),
            'cell_states': cell.detach().numpy()
        }
    
    def plot_feature_importance(self, importance_scores, method_name, save_plot=True):
        """
        Plot feature importance across time steps.
        
        Parameters:
        -----------
        importance_scores : dict
            Importance scores from interpretation methods
        method_name : str
            Name of the interpretation method
        save_plot : bool
            Whether to save the plot
        """
        print(f"📈 Creating feature importance plots for {method_name}...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Feature Importance Analysis: {method_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Feature importance heatmap
        if 'feature_importance' in importance_scores:
            feature_imp = importance_scores['feature_importance']
            if len(feature_imp.shape) == 2:  # Time x Features
                im = axes[0, 0].imshow(feature_imp.T, aspect='auto', cmap='viridis')
                axes[0, 0].set_title('Feature Importance Heatmap')
                axes[0, 0].set_xlabel('Time Step')
                axes[0, 0].set_ylabel('Feature')
                axes[0, 0].set_yticks(range(len(self.feature_names)))
                axes[0, 0].set_yticklabels(self.feature_names)
                plt.colorbar(im, ax=axes[0, 0])
        
        # Plot 2: Overall feature importance
        if 'feature_importance' in importance_scores:
            overall_imp = importance_scores['feature_importance']
            if len(overall_imp.shape) == 2:
                overall_imp = overall_imp.mean(axis=0)  # Average across time
            
            axes[0, 1].bar(self.feature_names, overall_imp, alpha=0.7)
            axes[0, 1].set_title('Overall Feature Importance')
            axes[0, 1].set_ylabel('Importance Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Time step importance
        if 'time_step_importance' in importance_scores:
            time_imp = importance_scores['time_step_importance']
            axes[1, 0].plot(range(len(time_imp)), time_imp, marker='o')
            axes[1, 0].set_title('Time Step Importance')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Importance Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Feature importance over time
        if 'feature_importance' in importance_scores:
            feature_imp = importance_scores['feature_importance']
            if len(feature_imp.shape) == 2:
                for i, feature in enumerate(self.feature_names):
                    axes[1, 1].plot(range(feature_imp.shape[0]), feature_imp[:, i], 
                                   label=feature, marker='o', alpha=0.7)
                axes[1, 1].set_title('Feature Importance Over Time')
                axes[1, 1].set_xlabel('Time Step')
                axes[1, 1].set_ylabel('Importance Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'feature_importance_{method_name.lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance plot saved as '{filename}'")
        
        plt.show()
    
    def plot_attention_analysis(self, attention_results, save_plot=True):
        """
        Plot attention analysis results.
        
        Parameters:
        -----------
        attention_results : dict
            Attention analysis results
        save_plot : bool
            Whether to save the plot
        """
        print(f"📈 Creating attention analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LSTM Attention Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Attention weights heatmap
        if 'attention_weights' in attention_results:
            attention_weights = attention_results['attention_weights']
            im = axes[0, 0].imshow(attention_weights.T, aspect='auto', cmap='viridis')
            axes[0, 0].set_title('Attention Weights Heatmap')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('Hidden State')
            plt.colorbar(im, ax=axes[0, 0])
        
        # Plot 2: Hidden states over time
        if 'hidden_states' in attention_results:
            hidden_states = attention_results['hidden_states'][-1]  # Last layer
            axes[0, 1].plot(hidden_states.T)
            axes[0, 1].set_title('Hidden States Over Time')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Hidden State Value')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cell states over time
        if 'cell_states' in attention_results:
            cell_states = attention_results['cell_states'][-1]  # Last layer
            axes[1, 0].plot(cell_states.T)
            axes[1, 0].set_title('Cell States Over Time')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Cell State Value')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Attention distribution
        if 'attention_weights' in attention_results:
            attention_weights = attention_results['attention_weights']
            axes[1, 1].hist(attention_weights.flatten(), bins=50, alpha=0.7)
            axes[1, 1].set_title('Attention Weights Distribution')
            axes[1, 1].set_xlabel('Attention Weight')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Attention analysis plot saved as 'attention_analysis.png'")
        
        plt.show()
    
    def comprehensive_interpretation(self, data, sequence_length=12, save_results=True):
        """
        Perform comprehensive interpretation analysis.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Time series data
        sequence_length : int
            Length of input sequences
        save_results : bool
            Whether to save results
            
        Returns:
        --------
        dict
            Comprehensive interpretation results
        """
        print("🚀 Comprehensive LSTM Interpretation Analysis")
        print("="*60)
        
        # Prepare data
        X_sequences, y_targets, scaled_data = self.prepare_interpretation_data(data, sequence_length)
        
        # Perform different interpretation methods
        results = {}
        
        # 1. Gradient-based methods
        print("\n📊 Computing gradient-based importance...")
        for method in ['gradients', 'integrated_gradients', 'smoothgrad']:
            try:
                results[method] = self.gradient_based_importance(X_sequences, method)
                self.plot_feature_importance(results[method], method)
            except Exception as e:
                print(f"  ⚠️  {method} failed: {e}")
        
        # 2. SHAP importance
        print("\n📊 Computing SHAP importance...")
        try:
            results['shap'] = self.shap_importance(X_sequences)
            self.plot_feature_importance(results['shap'], 'SHAP')
        except Exception as e:
            print(f"  ⚠️  SHAP failed: {e}")
        
        # 3. LIME importance
        print("\n📊 Computing LIME importance...")
        try:
            results['lime'] = self.lime_importance(X_sequences)
            self.plot_feature_importance(results['lime'], 'LIME')
        except Exception as e:
            print(f"  ⚠️  LIME failed: {e}")
        
        # 4. Attention analysis
        print("\n📊 Computing attention analysis...")
        try:
            results['attention'] = self.attention_analysis(X_sequences)
            self.plot_attention_analysis(results['attention'])
        except Exception as e:
            print(f"  ⚠️  Attention analysis failed: {e}")
        
        # Save results
        if save_results:
            self.save_interpretation_results(results)
        
        return results
    
    def save_interpretation_results(self, results, filename='interpretation_results.json'):
        """
        Save interpretation results to JSON file.
        
        Parameters:
        -----------
        results : dict
            Interpretation results
        filename : str
            Output filename
        """
        import json
        
        # Prepare results for JSON serialization
        serializable_results = {}
        
        for method, result in results.items():
            serializable_results[method] = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_results[method][key] = value.tolist()
                else:
                    serializable_results[method][key] = value
        
        # Add metadata
        serializable_results['metadata'] = {
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_info': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Interpretation results saved as '{filename}'")

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
    Main function to run LSTM interpretability analysis.
    """
    print("🚀 LSTM Time Series Interpretability Analysis")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded data: {df.shape}")
    except FileNotFoundError:
        print("❌ Data file not found. Please ensure cleaned_macroeconomic_data.csv exists.")
        return
    
    # Load trained model
    model, data_info = load_trained_model()
    
    if model is None:
        print("⚠️  No trained model found. Please train a model first.")
        return
    
    # Prepare scaler
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    scaler = MinMaxScaler()
    scaler.fit(df[feature_columns])
    
    # Initialize interpretability system
    interpreter = LSTMInterpretability(
        model=model,
        scaler=scaler,
        feature_names=feature_columns,
        target_name='Personal Loan Delinquency Rate'
    )
    
    # Run comprehensive interpretation
    results = interpreter.comprehensive_interpretation(df[feature_columns])
    
    print("\n" + "="*60)
    print("✅ LSTM interpretability analysis completed!")
    print("="*60)

if __name__ == "__main__":
    main() 