#!/usr/bin/env python3
"""
Simplified causal DAG creation and visualization for macroeconomic variables.
Uses NetworkX for graph creation and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
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

def create_causal_graph():
    """
    Create a causal graph using NetworkX.
    
    Assumptions:
    - Interest Rate and Unemployment affect Inflation and Delinquency
    - Inflation may affect Delinquency
    
    Returns:
    --------
    networkx.DiGraph
        Directed graph representing causal relationships
    """
    print("\n🔗 Creating causal graph...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    nodes = [
        "Federal Funds Rate",
        "Unemployment Rate", 
        "Inflation Rate (CPI)",
        "Personal Loan Delinquency Rate"
    ]
    
    for node in nodes:
        G.add_node(node)
    
    # Add edges (causal relationships)
    edges = [
        ("Federal Funds Rate", "Inflation Rate (CPI)"),
        ("Federal Funds Rate", "Personal Loan Delinquency Rate"),
        ("Unemployment Rate", "Inflation Rate (CPI)"),
        ("Unemployment Rate", "Personal Loan Delinquency Rate"),
        ("Inflation Rate (CPI)", "Personal Loan Delinquency Rate")
    ]
    
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    print("✅ Causal graph created with the following relationships:")
    for source, target in edges:
        print(f"  - {source} → {target}")
    
    return G

def visualize_causal_dag(G, save_plot=True):
    """
    Visualize the causal DAG using NetworkX.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Causal graph
    save_plot : bool
        Whether to save the plot
    """
    print("\n📈 Creating causal DAG visualization...")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Define node categories and colors
    exogenous_nodes = ["Federal Funds Rate", "Unemployment Rate"]
    intermediate_nodes = ["Inflation Rate (CPI)"]
    outcome_nodes = ["Personal Loan Delinquency Rate"]
    
    # Draw nodes with different colors based on their role
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=exogenous_nodes,
                          node_color='lightblue', 
                          node_size=3000,
                          alpha=0.8,
                          label='Exogenous Variables')
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=intermediate_nodes,
                          node_color='lightgreen', 
                          node_size=3000,
                          alpha=0.8,
                          label='Intermediate Variables')
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=outcome_nodes,
                          node_color='lightcoral', 
                          node_size=3000,
                          alpha=0.8,
                          label='Outcome Variables')
    
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
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('causal_dag_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ Causal DAG visualization saved as 'causal_dag_visualization.png'")
    
    plt.show()

def analyze_causal_paths(G):
    """
    Analyze the causal paths in the graph.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Causal graph
    """
    print("\n🔍 Analyzing causal paths...")
    
    # Get all simple paths
    all_paths = []
    for source in ["Federal Funds Rate", "Unemployment Rate"]:
        for target in ["Inflation Rate (CPI)", "Personal Loan Delinquency Rate"]:
            paths = list(nx.all_simple_paths(G, source, target))
            all_paths.extend(paths)
    
    print("All causal paths in the model:")
    for i, path in enumerate(all_paths, 1):
        path_str = " → ".join(path)
        print(f"  {i}. {path_str}")
    
    # Analyze direct vs indirect effects
    print("\nDirect effects (length 2):")
    direct_paths = [path for path in all_paths if len(path) == 2]
    for path in direct_paths:
        print(f"  - {path[0]} → {path[1]}")
    
    print("\nIndirect effects (length > 2):")
    indirect_paths = [path for path in all_paths if len(path) > 2]
    for path in indirect_paths:
        path_str = " → ".join(path)
        print(f"  - {path_str}")
    
    # Check for cycles (should be none in a DAG)
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"\n⚠️  Warning: {len(cycles)} cycles found in the graph")
        else:
            print("\n✅ No cycles found - graph is a valid DAG")
    except:
        print("\n✅ Graph structure validated")

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

def create_time_series_plot(df, save_plot=True):
    """
    Create a time series plot showing all variables over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with variables
    save_plot : bool
        Whether to save the plot
    """
    print("\n📈 Creating time series plot...")
    
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each variable
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Macroeconomic Variables Over Time', fontsize=16, fontweight='bold')
    
    variables = list(df.columns)
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[i//2, i%2]
        ax.plot(df.index, df[var], color=color, linewidth=2)
        ax.set_title(var, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('time_series_plot.png', dpi=300, bbox_inches='tight')
        print("✅ Time series plot saved as 'time_series_plot.png'")
    
    plt.show()

def save_graph_info(G, filename='causal_graph_info.txt'):
    """
    Save information about the causal graph.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Causal graph
    filename : str
        Output filename
    """
    try:
        with open(filename, 'w') as f:
            f.write("Causal Graph Information\n")
            f.write("="*50 + "\n\n")
            
            f.write("Nodes:\n")
            for node in G.nodes():
                f.write(f"  - {node}\n")
            
            f.write("\nEdges (Causal Relationships):\n")
            for edge in G.edges():
                f.write(f"  - {edge[0]} → {edge[1]}\n")
            
            f.write(f"\nGraph Properties:\n")
            f.write(f"  - Number of nodes: {G.number_of_nodes()}\n")
            f.write(f"  - Number of edges: {G.number_of_edges()}\n")
            f.write(f"  - Is DAG: {nx.is_directed_acyclic_graph(G)}\n")
            
            # Calculate in-degree and out-degree for each node
            f.write("\nNode Degrees:\n")
            for node in G.nodes():
                in_degree = G.in_degree(node)
                out_degree = G.out_degree(node)
                f.write(f"  - {node}: in-degree={in_degree}, out-degree={out_degree}\n")
        
        print(f"💾 Graph information saved as '{filename}'")
    except Exception as e:
        print(f"❌ Error saving graph info: {e}")

def main():
    """
    Main function to create and visualize the causal DAG.
    """
    print("🚀 Causal DAG Creation and Visualization")
    print("="*60)
    
    # Step 1: Load the cleaned data
    df = load_cleaned_data()
    
    # Step 2: Create the causal graph
    G = create_causal_graph()
    
    # Step 3: Analyze causal paths
    analyze_causal_paths(G)
    
    # Step 4: Visualize the DAG
    visualize_causal_dag(G)
    
    # Step 5: Create correlation heatmap
    create_correlation_heatmap(df)
    
    # Step 6: Create time series plot
    create_time_series_plot(df)
    
    # Step 7: Save graph information
    save_graph_info(G)
    
    # Final summary
    print("\n" + "="*60)
    print("📋 CAUSAL DAG SUMMARY")
    print("="*60)
    print("✅ Causal DAG created successfully")
    print("✅ Graph structure analyzed")
    print("✅ Visualizations generated")
    print("✅ Graph information saved")
    
    print("\nThe causal DAG represents the following economic relationships:")
    print("• Interest rates and unemployment affect inflation and delinquency")
    print("• Inflation may affect delinquency rates")
    print("• Both direct and indirect causal paths are captured")
    
    return G

if __name__ == "__main__":
    # Run the causal DAG creation
    causal_graph = main() 