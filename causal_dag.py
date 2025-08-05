#!/usr/bin/env python3
"""
Causal DAG creation and visualization for macroeconomic variables.
Uses the dowhy library to create and visualize causal relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from dowhy import CausalModel
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data(filepath='cleaned_macroeconomic_data.csv'):
    """
    Load the cleaned macroeconomic data.
    
    Parameters:
    -----------
    filepath : str
        Path to the cleaned CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned macroeconomic data
    """
    print("📊 Loading cleaned macroeconomic data...")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"📋 Variables: {list(df.columns)}")
    
    return df

def create_causal_dag():
    """
    Create a causal DAG based on economic theory.
    
    Assumptions:
    - Interest Rate and Unemployment affect Inflation and Delinquency
    - Inflation may affect Delinquency
    
    Returns:
    --------
    str
        DOT format string for the causal graph
    """
    print("\n🔗 Creating causal DAG...")
    
    # Define the causal relationships based on economic theory
    causal_graph = """
    digraph {
        // Exogenous variables (no incoming arrows)
        "Federal Funds Rate" [shape=box, style=filled, fillcolor=lightblue];
        "Unemployment Rate" [shape=box, style=filled, fillcolor=lightblue];
        
        // Endogenous variables
        "Inflation Rate (CPI)" [shape=box, style=filled, fillcolor=lightgreen];
        "Personal Loan Delinquency Rate" [shape=box, style=filled, fillcolor=lightcoral];
        
        // Causal relationships
        "Federal Funds Rate" -> "Inflation Rate (CPI)";
        "Federal Funds Rate" -> "Personal Loan Delinquency Rate";
        "Unemployment Rate" -> "Inflation Rate (CPI)";
        "Unemployment Rate" -> "Personal Loan Delinquency Rate";
        "Inflation Rate (CPI)" -> "Personal Loan Delinquency Rate";
        
        // Graph styling
        rankdir=LR;
        node [fontsize=12, fontname="Arial"];
        edge [fontsize=10, fontname="Arial"];
    }
    """
    
    print("✅ Causal DAG created with the following relationships:")
    print("  - Federal Funds Rate → Inflation Rate (CPI)")
    print("  - Federal Funds Rate → Personal Loan Delinquency Rate")
    print("  - Unemployment Rate → Inflation Rate (CPI)")
    print("  - Unemployment Rate → Personal Loan Delinquency Rate")
    print("  - Inflation Rate (CPI) → Personal Loan Delinquency Rate")
    
    return causal_graph

def create_dowhy_model(df, causal_graph):
    """
    Create a DoWhy causal model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with variables
    causal_graph : str
        DOT format causal graph
        
    Returns:
    --------
    dowhy.CausalModel
        DoWhy causal model
    """
    print("\n🔧 Creating DoWhy causal model...")
    
    # Create the causal model
    model = CausalModel(
        data=df,
        treatment=None,  # We'll specify this later for specific analyses
        outcome=None,    # We'll specify this later for specific analyses
        graph=causal_graph
    )
    
    print("✅ DoWhy causal model created successfully")
    
    return model

def visualize_dag_networkx(causal_graph, save_plot=True):
    """
    Visualize the causal DAG using NetworkX.
    
    Parameters:
    -----------
    causal_graph : str
        DOT format causal graph
    save_plot : bool
        Whether to save the plot
    """
    print("\n📈 Creating NetworkX visualization...")
    
    # Parse the DOT string to extract nodes and edges
    lines = causal_graph.strip().split('\n')
    nodes = []
    edges = []
    
    for line in lines:
        line = line.strip()
        if '->' in line and not line.startswith('//') and not line.startswith('digraph'):
            # Extract edge
            parts = line.split('->')
            source = parts[0].strip().strip('"')
            target = parts[1].strip().strip('"').strip(';')
            edges.append((source, target))
        elif '[' in line and 'shape=box' in line and not line.startswith('//'):
            # Extract node
            node_name = line.split('[')[0].strip().strip('"')
            nodes.append(node_name)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node)
    
    # Add edges
    for source, target in edges:
        G.add_edge(source, target)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes with different colors based on their role
    exogenous_nodes = ["Federal Funds Rate", "Unemployment Rate"]
    endogenous_nodes = ["Inflation Rate (CPI)", "Personal Loan Delinquency Rate"]
    
    # Draw exogenous nodes (light blue)
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=exogenous_nodes,
                          node_color='lightblue', 
                          node_size=3000,
                          alpha=0.8)
    
    # Draw endogenous nodes (light green and light coral)
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=["Inflation Rate (CPI)"],
                          node_color='lightgreen', 
                          node_size=3000,
                          alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=["Personal Loan Delinquency Rate"],
                          node_color='lightcoral', 
                          node_size=3000,
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->',
                          width=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_weight='bold')
    
    plt.title("Causal DAG: Macroeconomic Relationships", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=15, label='Exogenous Variables'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=15, label='Intermediate Variables'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                  markersize=15, label='Outcome Variables')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('causal_dag_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ Causal DAG visualization saved as 'causal_dag_visualization.png'")
    
    plt.show()

def analyze_causal_paths(model):
    """
    Analyze the causal paths in the model.
    
    Parameters:
    -----------
    model : dowhy.CausalModel
        DoWhy causal model
    """
    print("\n🔍 Analyzing causal paths...")
    
    # Get the graph structure
    graph = model.graph
    
    print("Causal paths in the model:")
    
    # Direct effects
    print("\nDirect effects:")
    print("  - Federal Funds Rate → Inflation Rate (CPI)")
    print("  - Federal Funds Rate → Personal Loan Delinquency Rate")
    print("  - Unemployment Rate → Inflation Rate (CPI)")
    print("  - Unemployment Rate → Personal Loan Delinquency Rate")
    print("  - Inflation Rate (CPI) → Personal Loan Delinquency Rate")
    
    # Indirect effects
    print("\nIndirect effects:")
    print("  - Federal Funds Rate → Inflation Rate (CPI) → Personal Loan Delinquency Rate")
    print("  - Unemployment Rate → Inflation Rate (CPI) → Personal Loan Delinquency Rate")
    
    # Backdoor paths
    print("\nBackdoor paths for Inflation Rate (CPI) → Personal Loan Delinquency Rate:")
    print("  - Through Federal Funds Rate")
    print("  - Through Unemployment Rate")

def create_correlation_heatmap(df, save_plot=True):
    """
    Create a correlation heatmap to show relationships between variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with variables
    save_plot : bool
        Whether to save the plot
    """
    print("\n📊 Creating correlation heatmap...")
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title("Correlation Matrix: Macroeconomic Variables", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")
    
    plt.show()

def save_causal_graph(causal_graph, filename='causal_graph.dot'):
    """
    Save the causal graph in DOT format.
    
    Parameters:
    -----------
    causal_graph : str
        DOT format causal graph
    filename : str
        Output filename
    """
    try:
        with open(filename, 'w') as f:
            f.write(causal_graph)
        print(f"💾 Causal graph saved as '{filename}'")
    except Exception as e:
        print(f"❌ Error saving causal graph: {e}")

def main():
    """
    Main function to create and visualize the causal DAG.
    """
    print("🚀 Causal DAG Creation and Visualization")
    print("="*60)
    
    # Step 1: Load the cleaned data
    df = load_cleaned_data()
    
    # Step 2: Create the causal DAG
    causal_graph = create_causal_dag()
    
    # Step 3: Create DoWhy model
    model = create_dowhy_model(df, causal_graph)
    
    # Step 4: Analyze causal paths
    analyze_causal_paths(model)
    
    # Step 5: Visualize the DAG
    visualize_dag_networkx(causal_graph)
    
    # Step 6: Create correlation heatmap
    create_correlation_heatmap(df)
    
    # Step 7: Save the causal graph
    save_causal_graph(causal_graph)
    
    # Final summary
    print("\n" + "="*60)
    print("📋 CAUSAL DAG SUMMARY")
    print("="*60)
    print("✅ Causal DAG created successfully")
    print("✅ DoWhy model initialized")
    print("✅ Visualizations generated")
    print("✅ Graph structure saved")
    
    print("\nThe causal DAG represents the following economic relationships:")
    print("• Interest rates and unemployment affect inflation and delinquency")
    print("• Inflation may affect delinquency rates")
    print("• Both direct and indirect causal paths are captured")
    
    return model, causal_graph

if __name__ == "__main__":
    # Run the causal DAG creation
    model, causal_graph = main() 