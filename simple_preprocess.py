#!/usr/bin/env python3
"""
Simplified data preprocessing script for macroeconomic time series data.
Focuses on core functionality: resampling, missing value imputation, and time alignment.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='macroeconomic_data.csv'):
    """Load the macroeconomic data."""
    print("📊 Loading data...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def resample_to_monthly(df):
    """Resample data to monthly frequency if not already monthly."""
    print("🔄 Checking frequency...")
    freq = pd.infer_freq(df.index)
    print(f"Current frequency: {freq}")
    
    if freq == 'M' or freq == 'MS':
        print("✅ Data is already monthly frequency")
        return df
    
    print("🔄 Resampling to monthly frequency...")
    df_monthly = df.resample('M').last()
    print(f"✅ Resampled: {df_monthly.shape[0]} rows")
    return df_monthly

def handle_missing_values(df, method='combined', limit=12):
    """Handle missing values using specified method."""
    print(f"🔧 Handling missing values using {method}...")
    
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")
    
    df_cleaned = df.copy()
    
    if method == 'combined':
        # Forward fill first, then backward fill
        df_cleaned = df_cleaned.fillna(method='ffill', limit=limit)
        df_cleaned = df_cleaned.fillna(method='bfill', limit=limit)
    elif method == 'forward_fill':
        df_cleaned = df_cleaned.fillna(method='ffill', limit=limit)
    elif method == 'interpolate':
        df_cleaned = df_cleaned.interpolate(method='linear', limit=limit)
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Missing values after: {missing_after}")
    print(f"✅ Handled {missing_before - missing_after} missing values")
    
    return df_cleaned

def align_time_ranges(df):
    """Align time ranges so all series have data for the same dates."""
    print("📅 Aligning time ranges...")
    
    # Find the common date range where all series have data
    non_null_ranges = {}
    
    for column in df.columns:
        non_null_data = df[column].dropna()
        if len(non_null_data) > 0:
            non_null_ranges[column] = {
                'start': non_null_data.index.min(),
                'end': non_null_data.index.max(),
                'count': len(non_null_data)
            }
    
    print("Date ranges for each series:")
    for col, range_info in non_null_ranges.items():
        print(f"  - {col}: {range_info['start']} to {range_info['end']} ({range_info['count']} observations)")
    
    # Find the intersection of all ranges
    if non_null_ranges:
        common_start = max([info['start'] for info in non_null_ranges.values()])
        common_end = min([info['end'] for info in non_null_ranges.values()])
        
        print(f"\nCommon date range: {common_start} to {common_end}")
        
        # Filter to common range
        df_aligned = df.loc[common_start:common_end]
        
        print(f"✅ Aligned data: {df_aligned.shape[0]} rows")
        
        return df_aligned
    else:
        print("❌ No valid data found")
        return df

def save_data(df, filename='cleaned_macroeconomic_data.csv'):
    """Save the cleaned data."""
    try:
        df.to_csv(filename)
        print(f"💾 Data saved to '{filename}'")
    except Exception as e:
        print(f"❌ Error saving data: {e}")

def main():
    """Main preprocessing pipeline."""
    print("🚀 Simple Data Preprocessing Pipeline")
    print("="*50)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Resample to monthly
    df_monthly = resample_to_monthly(df)
    
    # Step 3: Handle missing values
    df_cleaned = handle_missing_values(df_monthly, method='combined', limit=12)
    
    # Step 4: Align time ranges
    df_aligned = align_time_ranges(df_cleaned)
    
    # Step 5: Save cleaned data
    save_data(df_aligned)
    
    # Final summary
    print("\n" + "="*50)
    print("📋 FINAL SUMMARY")
    print("="*50)
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_aligned.shape}")
    print(f"Date range: {df_aligned.index.min()} to {df_aligned.index.max()}")
    print(f"Missing values: {df_aligned.isnull().sum().sum()}")
    
    print("\nFirst 5 rows:")
    print(df_aligned.head())
    
    print("\nLast 5 rows:")
    print(df_aligned.tail())
    
    return df_aligned

if __name__ == "__main__":
    cleaned_data = main() 