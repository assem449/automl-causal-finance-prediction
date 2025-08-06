# Causal DAG Summary: Macroeconomic Relationships

## Overview
This document summarizes the causal Directed Acyclic Graph (DAG) created for the macroeconomic variables in our dataset.

## Causal Graph Structure

### Variables
- **Federal Funds Rate** (Interest Rate)
- **Unemployment Rate** 
- **Inflation Rate (CPI)**
- **Personal Loan Delinquency Rate**

### Causal Relationships

#### Direct Effects
1. **Federal Funds Rate → Inflation Rate (CPI)**
   - Higher interest rates typically reduce inflation by slowing economic activity
   - Monetary policy tool used by central banks

2. **Federal Funds Rate → Personal Loan Delinquency Rate**
   - Higher interest rates increase borrowing costs
   - Can lead to higher delinquency rates as borrowers struggle with payments

3. **Unemployment Rate → Inflation Rate (CPI)**
   - Higher unemployment typically reduces inflationary pressures
   - Phillips curve relationship (inverse relationship)

4. **Unemployment Rate → Personal Loan Delinquency Rate**
   - Higher unemployment reduces income
   - Increases likelihood of loan defaults

5. **Inflation Rate (CPI) → Personal Loan Delinquency Rate**
   - Higher inflation erodes purchasing power
   - Can increase financial stress and delinquency

#### Indirect Effects
1. **Federal Funds Rate → Inflation Rate (CPI) → Personal Loan Delinquency Rate**
   - Interest rates affect inflation, which then affects delinquency

2. **Unemployment Rate → Inflation Rate (CPI) → Personal Loan Delinquency Rate**
   - Unemployment affects inflation, which then affects delinquency

## Graph Properties

### Node Categories
- **Exogenous Variables** (light blue): Federal Funds Rate, Unemployment Rate
  - These variables are not caused by other variables in our model
  - They represent external factors or policy decisions

- **Intermediate Variables** (light green): Inflation Rate (CPI)
  - Affected by exogenous variables
  - Also affects other variables

- **Outcome Variables** (light coral): Personal Loan Delinquency Rate
  - The main outcome of interest
  - Affected by all other variables

### Graph Validation
- ✅ **Acyclic**: No cycles in the graph (valid DAG)
- ✅ **Directed**: All relationships have clear direction
- ✅ **Complete**: All hypothesized relationships included

## Economic Theory Basis

### Monetary Policy Channel
- Central banks use interest rates to control inflation
- Higher rates → lower inflation → potentially lower delinquency

### Labor Market Channel
- Unemployment affects both inflation and financial stress
- Higher unemployment → lower inflation but higher delinquency risk

### Financial Stress Channel
- Inflation affects real income and purchasing power
- Higher inflation → increased financial stress → higher delinquency

## Data Coverage
- **Time Period**: 1990-01-01 to 2025-03-01
- **Frequency**: Monthly
- **Observations**: 423 complete observations
- **Missing Values**: 0 (fully cleaned dataset)

## Generated Files
- `causal_dag_visualization.png`: Visual representation of the causal graph
- `causal_graph_info.txt`: Detailed graph information and properties
- `correlation_heatmap.png`: Correlation matrix visualization
- `time_series_plot.png`: Time series plots of all variables

## Next Steps for Causal Analysis
1. **Identification**: Verify that causal effects are identifiable
2. **Estimation**: Use appropriate methods (regression, instrumental variables, etc.)
3. **Validation**: Test assumptions and robustness
4. **Interpretation**: Interpret results in economic context

## Assumptions
- **Causal Markov Condition**: Variables are independent of their non-descendants given their parents
- **Faithfulness**: All conditional independencies in the data are due to the graph structure
- **No Unmeasured Confounding**: All relevant variables are included in the model
- **Linear Relationships**: For simplicity, we assume linear causal effects

## Limitations
- **Simplified Model**: Real economic relationships are more complex
- **Time Lags**: Immediate effects may not capture delayed responses
- **External Factors**: Other variables not included may be important
- **Structural Changes**: Relationships may change over time

## Usage
This causal DAG can be used for:
- **Causal Inference**: Estimating causal effects between variables
- **Policy Analysis**: Understanding impact of policy changes
- **Risk Assessment**: Predicting delinquency based on economic conditions
- **Scenario Analysis**: Simulating different economic scenarios 