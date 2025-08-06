#!/usr/bin/env python3
"""
Add synthetic group columns to macroeconomic dataset for fairness evaluation.
Creates meaningful groups based on economic indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_macroeconomic_data():
    """
    Load the cleaned macroeconomic data.
    
    Returns:
    --------
    pandas.DataFrame
        Macroeconomic data with Date index
    """
    try:
        df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded macroeconomic data: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Error: cleaned_macroeconomic_data.csv not found")
        return None

def create_economic_groups(df):
    """
    Create synthetic group columns based on economic indicators.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Macroeconomic data
        
    Returns:
    --------
    pandas.DataFrame
        Data with added group columns
    """
    print("🔧 Creating synthetic group columns...")
    
    # Create a copy to avoid modifying original
    df_with_groups = df.copy()
    
    # Group 1: Economic Condition Groups (based on Federal Funds Rate)
    # High rates = restrictive monetary policy, Low rates = accommodative
    fed_funds_median = df['Federal Funds Rate'].median()
    df_with_groups['Economic_Group'] = df['Federal Funds Rate'].apply(
        lambda x: 'Group A (High Rates)' if x > fed_funds_median else 'Group B (Low Rates)'
    )
    
    # Group 2: Inflation Pressure Groups (based on Inflation Rate)
    # High inflation = economic pressure, Low inflation = stable conditions
    inflation_median = df['Inflation Rate (CPI)'].median()
    df_with_groups['Inflation_Group'] = df['Inflation Rate (CPI)'].apply(
        lambda x: 'Group A (High Inflation)' if x > inflation_median else 'Group B (Low Inflation)'
    )
    
    # Group 3: Combined Economic Health Groups
    # Combine both indicators for a more nuanced grouping
    def create_combined_group(row):
        fed_high = row['Federal Funds Rate'] > fed_funds_median
        inflation_high = row['Inflation Rate (CPI)'] > inflation_median
        
        if fed_high and inflation_high:
            return 'Group A (High Rates + High Inflation)'
        elif fed_high and not inflation_high:
            return 'Group B (High Rates + Low Inflation)'
        elif not fed_high and inflation_high:
            return 'Group C (Low Rates + High Inflation)'
        else:
            return 'Group D (Low Rates + Low Inflation)'
    
    df_with_groups['Combined_Group'] = df_with_groups.apply(create_combined_group, axis=1)
    
    # Group 4: Time-based Groups (different economic eras)
    def create_time_group(date):
        year = date.year
        if year < 2000:
            return 'Group A (Pre-2000 Era)'
        elif year < 2008:
            return 'Group B (2000-2007 Era)'
        elif year < 2015:
            return 'Group C (2008-2014 Era)'
        else:
            return 'Group D (Post-2015 Era)'
    
    df_with_groups['Time_Group'] = df_with_groups.index.map(create_time_group)
    
    # Group 5: Delinquency Risk Groups (based on delinquency rate)
    # This creates groups that might be more relevant for credit risk
    delinquency_median = df['Personal Loan Delinquency Rate'].median()
    df_with_groups['Risk_Group'] = df['Personal Loan Delinquency Rate'].apply(
        lambda x: 'Group A (High Risk)' if x > delinquency_median else 'Group B (Low Risk)'
    )
    
    print(f"✅ Created 5 different group columns:")
    print(f"  - Economic_Group: Based on Federal Funds Rate")
    print(f"  - Inflation_Group: Based on Inflation Rate")
    print(f"  - Combined_Group: Based on both economic indicators")
    print(f"  - Time_Group: Based on economic eras")
    print(f"  - Risk_Group: Based on delinquency rate")
    
    return df_with_groups

def analyze_group_distributions(df):
    """
    Analyze the distribution of groups and their characteristics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with group columns
    """
    print("\n📊 Group Distribution Analysis")
    print("="*50)
    
    group_columns = ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']
    
    for group_col in group_columns:
        print(f"\n🔍 {group_col}:")
        group_counts = df[group_col].value_counts()
        print(f"  Distribution:")
        for group, count in group_counts.items():
            percentage = (count / len(df)) * 100
            print(f"    {group}: {count} samples ({percentage:.1f}%)")
        
        # Calculate mean delinquency rate for each group
        group_means = df.groupby(group_col)['Personal Loan Delinquency Rate'].mean()
        print(f"  Mean Delinquency Rate by Group:")
        for group, mean_rate in group_means.items():
            print(f"    {group}: {mean_rate:.3f}%")

def create_fairness_visualizations(df):
    """
    Create visualizations for fairness analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with group columns
    """
    print("\n📈 Creating fairness visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Fairness Analysis Across Economic Groups', fontsize=16, fontweight='bold')
    
    group_columns = ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']
    
    # Plot 1: Delinquency Rate by Economic Group
    sns.boxplot(data=df, x='Economic_Group', y='Personal Loan Delinquency Rate', ax=axes[0, 0])
    axes[0, 0].set_title('Delinquency Rate by Economic Group')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Delinquency Rate by Inflation Group
    sns.boxplot(data=df, x='Inflation_Group', y='Personal Loan Delinquency Rate', ax=axes[0, 1])
    axes[0, 1].set_title('Delinquency Rate by Inflation Group')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Delinquency Rate by Combined Group
    sns.boxplot(data=df, x='Combined_Group', y='Personal Loan Delinquency Rate', ax=axes[0, 2])
    axes[0, 2].set_title('Delinquency Rate by Combined Group')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Delinquency Rate by Time Group
    sns.boxplot(data=df, x='Time_Group', y='Personal Loan Delinquency Rate', ax=axes[1, 0])
    axes[1, 0].set_title('Delinquency Rate by Time Group')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Delinquency Rate by Risk Group
    sns.boxplot(data=df, x='Risk_Group', y='Personal Loan Delinquency Rate', ax=axes[1, 1])
    axes[1, 1].set_title('Delinquency Rate by Risk Group')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Time series of delinquency rate with group coloring
    df_plot = df.reset_index()
    df_plot['Year'] = df_plot['Date'].dt.year
    
    # Color by Economic Group
    colors = {'Group A (High Rates)': 'red', 'Group B (Low Rates)': 'blue'}
    for group in df_plot['Economic_Group'].unique():
        group_data = df_plot[df_plot['Economic_Group'] == group]
        axes[1, 2].scatter(group_data['Year'], group_data['Personal Loan Delinquency Rate'], 
                           alpha=0.6, label=group, color=colors.get(group, 'gray'))
    
    axes[1, 2].set_title('Delinquency Rate Over Time by Economic Group')
    axes[1, 2].set_xlabel('Year')
    axes[1, 2].set_ylabel('Personal Loan Delinquency Rate (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fairness_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Fairness analysis plot saved as 'fairness_analysis.png'")
    plt.show()

def create_group_statistics(df):
    """
    Create detailed statistics for each group.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with group columns
        
    Returns:
    --------
    pandas.DataFrame
        Group statistics
    """
    print("\n📋 Creating group statistics...")
    
    group_columns = ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']
    feature_columns = ['Federal Funds Rate', 'Inflation Rate (CPI)', 'Personal Loan Delinquency Rate']
    
    stats_list = []
    
    for group_col in group_columns:
        group_stats = df.groupby(group_col)[feature_columns].agg(['mean', 'std', 'min', 'max']).round(4)
        
        # Flatten column names
        group_stats.columns = [f"{col[0]}_{col[1]}" for col in group_stats.columns]
        
        # Add group information
        group_stats['group_column'] = group_col
        group_stats['group_name'] = group_stats.index
        
        stats_list.append(group_stats.reset_index(drop=True))
    
    # Combine all statistics
    all_stats = pd.concat(stats_list, ignore_index=True)
    
    # Save statistics
    all_stats.to_csv('group_statistics.csv', index=False)
    print("✅ Group statistics saved as 'group_statistics.csv'")
    
    return all_stats

def save_fairness_dataset(df):
    """
    Save the dataset with fairness groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with group columns
    """
    # Save the main dataset with groups
    df.to_csv('macroeconomic_data_with_fairness_groups.csv')
    print("✅ Dataset with fairness groups saved as 'macroeconomic_data_with_fairness_groups.csv'")
    
    # Create a summary of the groups
    group_summary = {
        'Economic_Group': {
            'description': 'Groups based on Federal Funds Rate (monetary policy stance)',
            'group_a': 'High interest rates (restrictive monetary policy)',
            'group_b': 'Low interest rates (accommodative monetary policy)'
        },
        'Inflation_Group': {
            'description': 'Groups based on Inflation Rate (CPI)',
            'group_a': 'High inflation periods',
            'group_b': 'Low inflation periods'
        },
        'Combined_Group': {
            'description': 'Groups based on both Federal Funds Rate and Inflation Rate',
            'group_a': 'High rates + High inflation (stagflation-like conditions)',
            'group_b': 'High rates + Low inflation (tight monetary policy)',
            'group_c': 'Low rates + High inflation (loose monetary policy)',
            'group_d': 'Low rates + Low inflation (accommodative conditions)'
        },
        'Time_Group': {
            'description': 'Groups based on economic eras',
            'group_a': 'Pre-2000 (pre-dotcom bubble)',
            'group_b': '2000-2007 (dotcom bubble and housing boom)',
            'group_c': '2008-2014 (financial crisis and recovery)',
            'group_d': 'Post-2015 (recent economic conditions)'
        },
        'Risk_Group': {
            'description': 'Groups based on Personal Loan Delinquency Rate',
            'group_a': 'High delinquency risk periods',
            'group_b': 'Low delinquency risk periods'
        }
    }
    
    # Save group summary
    import json
    with open('fairness_groups_summary.json', 'w') as f:
        json.dump(group_summary, f, indent=2)
    
    print("✅ Group summary saved as 'fairness_groups_summary.json'")

def main():
    """
    Main function to add fairness groups to the dataset.
    """
    print("🚀 Adding Fairness Groups to Macroeconomic Dataset")
    print("="*60)
    
    # Load data
    df = load_macroeconomic_data()
    if df is None:
        return
    
    # Create groups
    df_with_groups = create_economic_groups(df)
    
    # Analyze distributions
    analyze_group_distributions(df_with_groups)
    
    # Create visualizations
    create_fairness_visualizations(df_with_groups)
    
    # Create statistics
    stats = create_group_statistics(df_with_groups)
    
    # Save dataset
    save_fairness_dataset(df_with_groups)
    
    print("\n" + "="*60)
    print("✅ Fairness groups successfully added!")
    print("="*60)
    print(f"📊 Dataset shape: {df_with_groups.shape}")
    print(f"📁 Files created:")
    print(f"  - macroeconomic_data_with_fairness_groups.csv")
    print(f"  - fairness_analysis.png")
    print(f"  - group_statistics.csv")
    print(f"  - fairness_groups_summary.json")
    
    return df_with_groups

if __name__ == "__main__":
    df_with_groups = main() 