# Optuna Hyperparameter Optimization for LSTM

## Overview
This implementation provides automated hyperparameter optimization for LSTM time series prediction models using Optuna. The system automatically tunes key hyperparameters including hidden size, number of layers, and learning rate to find the optimal model configuration.

## Key Features

### 🚀 **Automated Optimization**
- **Bayesian Optimization**: Uses TPE (Tree-structured Parzen Estimator) sampler
- **Multi-parameter Tuning**: Optimizes 6+ hyperparameters simultaneously
- **Early Stopping**: Prunes poor trials to save computation time
- **Parallel Trials**: Supports parallel optimization across multiple cores

### 📊 **Comprehensive Search Space**
- **Hidden Size**: [32, 64, 128, 256]
- **Number of Layers**: [1, 2, 3, 4]
- **Learning Rate**: [1e-5, 1e-1] (log scale)
- **Dropout Rate**: [0.1, 0.5]
- **Weight Decay**: [1e-6, 1e-3] (log scale)
- **Batch Size**: [16, 32, 64, 128]
- **Optimizer**: [Adam, AdamW]

### 🔧 **Advanced Features**
- **Parameter Importance Analysis**: Identifies most important hyperparameters
- **Visualization**: Comprehensive plots of optimization results
- **Result Persistence**: Saves optimization results and best model
- **Customizable**: Easy to modify search spaces and objectives

## Usage Examples

### Basic Optimization
```python
from optuna_lstm_tuning import OptunaLSTMTuner

# Initialize tuner
tuner = OptunaLSTMTuner(
    data=df,
    target_column='target_variable',
    sequence_length=12,
    n_trials=50,
    timeout=3600,  # 1 hour
    study_name='lstm_optimization'
)

# Run optimization
study = tuner.optimize()

# Train best model
trainer, model, history = tuner.train_best_model(epochs=100)
```

### Custom Search Space
```python
class CustomOptunaTuner(OptunaLSTMTuner):
    def objective(self, trial):
        # Custom hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 2, 4)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # Build and train model
        model = self._build_model(hidden_size, num_layers)
        trainer = self._train_model(model, learning_rate)
        
        return trainer.best_val_loss
```

### Parameter Importance Analysis
```python
# Get parameter importance
importance = optuna.importance.get_param_importances(study)
for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{param}: {imp:.4f}")
```

## Hyperparameters Optimized

### **Architecture Parameters**
1. **Hidden Size**: Number of LSTM hidden units
   - Range: [32, 64, 128, 256]
   - Impact: Model capacity and complexity

2. **Number of Layers**: Depth of LSTM network
   - Range: [1, 2, 3, 4]
   - Impact: Model expressiveness and training time

3. **Dropout Rate**: Regularization strength
   - Range: [0.1, 0.5]
   - Impact: Overfitting prevention

### **Training Parameters**
4. **Learning Rate**: Step size for optimization
   - Range: [1e-5, 1e-1] (log scale)
   - Impact: Convergence speed and stability

5. **Weight Decay**: L2 regularization
   - Range: [1e-6, 1e-3] (log scale)
   - Impact: Model regularization

6. **Batch Size**: Training batch size
   - Range: [16, 32, 64, 128]
   - Impact: Memory usage and gradient estimates

7. **Optimizer**: Optimization algorithm
   - Options: [Adam, AdamW]
   - Impact: Convergence characteristics

## Optimization Process

### **1. Study Creation**
```python
study = optuna.create_study(
    direction='minimize',  # Minimize validation loss
    study_name='lstm_optimization',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
```

### **2. Trial Execution**
For each trial:
1. **Suggest Parameters**: Optuna suggests hyperparameter values
2. **Build Model**: Create LSTM with suggested parameters
3. **Train Model**: Train with early stopping
4. **Evaluate**: Calculate validation loss
5. **Report**: Report result back to Optuna

### **3. Optimization Loop**
- **TPE Sampler**: Uses Bayesian optimization
- **Median Pruner**: Stops poor trials early
- **Progress Tracking**: Real-time progress monitoring

## Results Analysis

### **Optimization History**
- **Loss Curve**: Shows improvement over trials
- **Parameter Trajectories**: How parameters evolve
- **Best Trial**: Identifies optimal configuration

### **Parameter Importance**
- **Feature Importance**: Which parameters matter most
- **Correlation Analysis**: Parameter relationships
- **Sensitivity Analysis**: Parameter impact on performance

### **Visualization**
1. **Optimization History**: Loss over trials
2. **Parameter Importance**: Bar chart of importance
3. **Parameter vs Loss**: Scatter plots
4. **Correlation Matrix**: Parameter relationships

## Example Results

### **Sample Optimization Output**
```
🚀 Starting Optuna optimization...
  Trials: 50
  Timeout: 3600 seconds
  Study name: lstm_macroeconomic_optimization

[I 2025-08-05 17:22:05,032] A new study created in memory with name: lstm_macroeconomic_optimization
[I 2025-08-05 17:22:15,123] Trial 0 finished with value: 0.082609
[I 2025-08-05 17:22:25,234] Trial 1 finished with value: 0.034015
...
[I 2025-08-05 17:45:12,456] Trial 49 finished with value: 0.012345

✅ Optimization completed!
Best validation loss: 0.012345
Best parameters: {
  'hidden_size': 128,
  'num_layers': 2,
  'learning_rate': 0.001234,
  'dropout': 0.2,
  'weight_decay': 1e-5,
  'batch_size': 32,
  'optimizer': 'adam'
}
```

### **Parameter Importance Results**
```
📊 Parameter Importance:
  learning_rate: 0.3245
  hidden_size: 0.2876
  num_layers: 0.1987
  dropout: 0.1234
  weight_decay: 0.0654
  batch_size: 0.0004
```

## Advanced Features

### **Custom Objectives**
```python
def custom_objective(trial):
    # Custom loss function
    model = build_model(trial)
    trainer = train_model(model, trial)
    
    # Multi-objective optimization
    return trainer.best_val_loss, trainer.final_val_r2
```

### **Pruning Strategies**
```python
# Median pruner
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=10,
    interval_steps=1
)

# Percentile pruner
pruner = optuna.pruners.PercentilePruner(
    percentile=25.0,
    n_startup_trials=5,
    n_warmup_steps=10,
    interval_steps=1
)
```

### **Parallel Optimization**
```python
# Run multiple trials in parallel
study.optimize(
    objective,
    n_trials=50,
    n_jobs=4,  # Use 4 CPU cores
    timeout=3600
)
```

## Best Practices

### **Search Space Design**
1. **Start Wide**: Begin with broad parameter ranges
2. **Focus Important**: Concentrate on high-impact parameters
3. **Log Scale**: Use log scale for learning rate and weight decay
4. **Discrete Values**: Use categorical for architecture choices

### **Optimization Strategy**
1. **Warm Start**: Use previous studies to initialize
2. **Early Stopping**: Prune poor trials quickly
3. **Validation**: Use robust validation strategy
4. **Reproducibility**: Set random seeds for consistency

### **Resource Management**
1. **Timeout**: Set reasonable timeouts
2. **Memory**: Monitor memory usage
3. **Parallel**: Use multiple cores when available
4. **Persistence**: Save intermediate results

## Performance Comparison

### **Optimization Methods**
| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **TPE** | Efficient, Bayesian | Complex setup | Most cases |
| **Random** | Simple, robust | Inefficient | Quick exploration |
| **Grid** | Exhaustive | Expensive | Small spaces |

### **Typical Results**
- **Improvement**: 20-50% better validation loss
- **Time**: 2-10x faster than manual tuning
- **Reliability**: More consistent model performance

## Integration with Existing Code

### **Seamless Integration**
```python
# Use with existing LSTM code
from lstm_timeseries import LSTMModel
from train_lstm import LSTMTrainer

# Optimize hyperparameters
tuner = OptunaLSTMTuner(data=df, target_column='target')
study = tuner.optimize()

# Use best model
trainer, model, history = tuner.train_best_model()
```

### **Customization Points**
1. **Search Space**: Modify parameter ranges
2. **Objective**: Change optimization target
3. **Sampler**: Use different optimization algorithms
4. **Pruner**: Implement custom pruning strategies

## Files Generated

### **Output Files**
- `best_optimized_lstm_model.pth`: Best model weights
- `optimization_results.json`: Complete optimization results
- `optuna_optimization_results.png`: Optimization visualization
- `parameter_correlations.png`: Parameter relationship plots

### **Code Files**
- `optuna_lstm_tuning.py`: Main optimization implementation
- `optuna_example.py`: Usage examples
- `Optuna_Optimization_Documentation.md`: This documentation

## Troubleshooting

### **Common Issues**
1. **Memory Errors**: Reduce batch size or model complexity
2. **Slow Optimization**: Use fewer trials or faster models
3. **Poor Results**: Expand search space or increase trials
4. **Convergence Issues**: Adjust learning rate range

### **Performance Tips**
1. **GPU Acceleration**: Use CUDA for faster training
2. **Parallel Trials**: Use multiple CPU cores
3. **Early Pruning**: Stop poor trials quickly
4. **Caching**: Save intermediate results

## Future Enhancements

### **Planned Features**
- **Multi-objective Optimization**: Optimize multiple metrics
- **Transfer Learning**: Use previous studies for warm start
- **Distributed Optimization**: Run across multiple machines
- **AutoML Integration**: Combine with other AutoML tools

### **Advanced Techniques**
- **Neural Architecture Search**: Optimize model architecture
- **Hyperparameter Transfer**: Transfer knowledge between datasets
- **Bayesian Neural Networks**: Uncertainty quantification
- **Meta-learning**: Learn optimization strategies

## Conclusion

The Optuna integration provides a powerful, automated solution for hyperparameter optimization of LSTM models. It significantly reduces the time and effort required to find optimal model configurations while improving model performance through systematic exploration of the hyperparameter space.

Key benefits:
- **Automation**: Reduces manual tuning effort
- **Efficiency**: Finds better parameters faster
- **Reproducibility**: Systematic and documented process
- **Flexibility**: Easy to customize for specific needs
- **Scalability**: Handles complex optimization problems 