# AutoML Causal Finance Prediction

A comprehensive machine learning project for macroeconomic time series prediction using LSTM models with causal inference, fairness evaluation, and interpretability analysis.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
python src/setup_env.py
```

### 2. Data Fetching
```bash
# Fetch macroeconomic data from FRED
python src/fetch_fred_data.py
```

### 3. Data Preprocessing
```bash
# Clean and align time series data
python src/preprocess_data.py
```

### 4. Model Training
```bash
# Train LSTM model
python src/train_lstm.py
```

### 5. Hyperparameter Optimization
```bash
# Run Optuna optimization
python src/optuna_lstm_tuning.py
```

### 6. Fairness Evaluation
```bash
# Create fairness groups
python src/create_fairness_dataset.py

# Evaluate fairness
python src/fairness_evaluation.py
```

### 7. Interpretability Analysis
```bash
# Run interpretability analysis
python src/lstm_interpretability.py
```

## 📊 Key Features

### **Data Management**
- **FRED API Integration**: Fetch macroeconomic indicators
- **Time Series Preprocessing**: Clean and align data
- **Fairness Groups**: Synthetic groups for bias evaluation

### **Model Development**
- **LSTM Architecture**: Time series prediction models
- **Optuna Optimization**: Automated hyperparameter tuning
- **Training Pipeline**: Comprehensive training with validation

### **Causal Analysis**
- **Causal DAG**: Directed Acyclic Graph creation
- **Causal Inference**: Understand variable relationships
- **Economic Interpretation**: Domain-specific analysis

### **Fairness Evaluation**
- **Fairlearn Integration**: Demographic parity and equalized odds
- **Group Analysis**: Performance across economic groups
- **Bias Detection**: Identify and mitigate model bias

### **Interpretability**
- **SHAP Analysis**: Game theory-based feature importance
- **LIME Explanations**: Local interpretable explanations
- **Gradient Analysis**: Model-specific importance
- **Attention Analysis**: LSTM attention patterns

## 🔧 Core Components

### **LSTM Time Series Model**
```python
from src.lstm_timeseries import LSTMModel, TimeSeriesDataset

# Create model
model = LSTMModel(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    output_size=1
)
```

### **Optuna Hyperparameter Optimization**
```python
from src.optuna_lstm_tuning import OptunaLSTMTuner

# Initialize tuner
tuner = OptunaLSTMTuner(
    data=df,
    n_trials=50,
    timeout=3600
)

# Run optimization
study = tuner.optimize()
```

### **Fairness Evaluation**
```python
from src.fairlearn_evaluation import FairlearnEvaluator

# Initialize evaluator
evaluator = FairlearnEvaluator()

# Evaluate fairness
results = evaluator.comprehensive_fairlearn_evaluation(model, scaler)
```

### **Interpretability Analysis**
```python
from src.lstm_interpretability import LSTMInterpretability

# Initialize interpreter
interpreter = LSTMInterpretability(model, scaler, feature_names)

# Run comprehensive analysis
results = interpreter.comprehensive_interpretation(data)
```

## 📈 Output Files

### **Visualizations** (`outputs/`)
- `macroeconomic_indicators.png`: Time series plots
- `lstm_training_history.png`: Training progress
- `optuna_optimization_results.png`: Optimization results
- `fairness_analysis.png`: Fairness evaluation plots
- `feature_importance_*.png`: Interpretability plots

### **Results** (`outputs/`)
- `fairness_groups_summary.json`: Fairness group definitions
- `interpretation_results.json`: Interpretability analysis
- `optimization_results.json`: Optuna optimization results

### **Data** (`data/`)
- `cleaned_macroeconomic_data.csv`: Preprocessed data
- `macroeconomic_data_with_fairness_groups.csv`: Data with fairness groups
- `group_statistics.csv`: Group analysis statistics

## 🎯 Use Cases

### **Financial Risk Modeling**
- Predict loan delinquency rates
- Understand economic factor relationships
- Ensure fair lending practices

### **Economic Forecasting**
- Forecast macroeconomic indicators
- Analyze causal relationships
- Validate model interpretability

### **Regulatory Compliance**
- Fairness evaluation for lending models
- Interpretability for regulatory reporting
- Bias detection and mitigation

## 📚 Documentation

- **LSTM_Documentation.md**: Complete LSTM implementation guide
- **Optuna_Optimization_Documentation.md**: Hyperparameter optimization guide
- **Fairlearn_Documentation.md**: Fairness evaluation guide
- **LSTM_Interpretability_Documentation.md**: Interpretability analysis guide

## 🔍 Examples

### **Basic Usage**
```python
# Train and evaluate LSTM
python src/training_example.py

# Run hyperparameter optimization
python src/optuna_example.py

# Evaluate fairness
python src/fairness_example.py

# Analyze interpretability
python src/interpretability_example.py
```

### **Advanced Analysis**
```python
# Comprehensive causal analysis
python src/causal_dag.py

# Full fairness evaluation
python src/fairlearn_evaluation.py

# Complete interpretability analysis
python src/lstm_interpretability.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FRED API**: Federal Reserve Economic Data
- **Optuna**: Hyperparameter optimization framework
- **Fairlearn**: Fairness evaluation library
- **SHAP**: Model interpretability library
- **PyTorch**: Deep learning framework

---

**Note**: This project demonstrates advanced machine learning techniques for time series prediction with a focus on fairness, interpretability, and causal inference. It's designed for educational and research purposes in financial modeling and economic forecasting. 