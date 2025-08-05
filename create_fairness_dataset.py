#!/usr/bin/env python3
"""
Create dataset with fairness groups for model evaluation.
"""

import pandas as pd
import numpy as np
import json

def create_fairness_groups():
    """
    Create dataset with fairness groups.
    """
    print("🚀 Creating dataset with fairness groups...")
    
    # Load data
    df = pd.read_csv('cleaned_macroeconomic_data.csv', index_col=0, parse_dates=True)
    print(f"✅ Loaded data: {df.shape}")
    
    # Create groups
    df_with_groups = df.copy()
    
    # Economic Group (based on Federal Funds Rate)
    fed_funds_median = df['Federal Funds Rate'].median()
    df_with_groups['Economic_Group'] = df['Federal Funds Rate'].apply(
        lambda x: 'Group A (High Rates)' if x > fed_funds_median else 'Group B (Low Rates)'
    )
    
    # Inflation Group (based on Inflation Rate)
    inflation_median = df['Inflation Rate (CPI)'].median()
    df_with_groups['Inflation_Group'] = df['Inflation Rate (CPI)'].apply(
        lambda x: 'Group A (High Inflation)' if x > inflation_median else 'Group B (Low Inflation)'
    )
    
    # Combined Group (both indicators)
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
    
    # Time Group (economic eras)
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
    
    # Risk Group (based on delinquency rate)
    delinquency_median = df['Personal Loan Delinquency Rate'].median()
    df_with_groups['Risk_Group'] = df['Personal Loan Delinquency Rate'].apply(
        lambda x: 'Group A (High Risk)' if x > delinquency_median else 'Group B (Low Risk)'
    )
    
    # Save dataset
    df_with_groups.to_csv('macroeconomic_data_with_fairness_groups.csv')
    print("✅ Saved dataset with fairness groups")
    
    # Print group distributions
    print("\n📊 Group Distributions:")
    for group_col in ['Economic_Group', 'Inflation_Group', 'Combined_Group', 'Time_Group', 'Risk_Group']:
        print(f"\n{group_col}:")
        counts = df_with_groups[group_col].value_counts()
        for group, count in counts.items():
            percentage = (count / len(df_with_groups)) * 100
            print(f"  {group}: {count} samples ({percentage:.1f}%)")
    
    # Create group summary
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
    
    with open('fairness_groups_summary.json', 'w') as f:
        json.dump(group_summary, f, indent=2)
    
    print("\n✅ Created fairness groups successfully!")
    print(f"📊 Dataset shape: {df_with_groups.shape}")
    print(f"📁 Files created:")
    print(f"  - macroeconomic_data_with_fairness_groups.csv")
    print(f"  - fairness_groups_summary.json")
    
    return df_with_groups

if __name__ == "__main__":
    df_with_groups = create_fairness_groups() 