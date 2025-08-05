#!/usr/bin/env python3
"""
Data preprocessing script for macroeconomic time series data.
Handles resampling, missing value imputation, and time alignment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data(filepath='macroeconomic_data.csv'):
    """
    Load the macroeconomic data and provide initial inspection.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the data
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data with datetime index
    """
    print("📊 Loading and inspecting data...")
    
    # Load the data
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"✅ Data loaded successfully")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Display data info
    print("\n" + "="*60)
    print("📈 INITIAL DATA INSPECTION")
    print("="*60)
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    missing_data = df.isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  - {col}: {missing} missing values ({missing/len(df)*100:.1f}%)")
        else:
            print(f"  - {col}: No missing values")
    
    print(f"\nSummary statistics:")
    print(df.describe())
    
    return df

def resample_to_monthly(df):
    """
    Resample data to monthly frequency if not already monthly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with datetime index
        
    Returns:
    --------
    pandas.DataFrame
        Resampled data with monthly frequency
    """
    print("\n🔄 Resampling to monthly frequency...")
    
    # Check current frequency
    freq = pd.infer_freq(df.index)
    print(f"Current frequency: {freq}")
    
    if freq == 'M' or freq == 'MS':
        print("✅ Data is already monthly frequency")
        return df
    
    # Resample to monthly frequency (end of month)
    df_monthly = df.resample('M').last()
    
    print(f"✅ Resampled to monthly frequency")
    print(f"📊 New shape: {df_monthly.shape}")
    
    return df_monthly

def handle_missing_values(df, method='forward_fill', limit=None):
    """
    Handle missing values using specified method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str
        Method for handling missing values:
        - 'forward_fill': Forward fill (ffill)
        - 'backward_fill': Backward fill (bfill)
        - 'interpolate': Linear interpolation
        - 'combined': Forward fill then backward fill
    limit : int, optional
        Maximum number of consecutive fills
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing values handled
    """
    print(f"\n🔧 Handling missing values using {method}...")
    
    # Count missing values before
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")
    
    df_cleaned = df.copy()
    
    if method == 'forward_fill':
        df_cleaned = df_cleaned.fillna(method='ffill', limit=limit)
    elif method == 'backward_fill':
        df_cleaned = df_cleaned.fillna(method='bfill', limit=limit)
    elif method == 'interpolate':
        df_cleaned = df_cleaned.interpolate(method='linear', limit=limit)
    elif method == 'combined':
        # Forward fill first, then backward fill
        df_cleaned = df_cleaned.fillna(method='ffill', limit=limit)
        df_cleaned = df_cleaned.fillna(method='bfill', limit=limit)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Count missing values after
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Missing values after: {missing_after}")
    print(f"✅ Handled {missing_before - missing_after} missing values")
    
    return df_cleaned

def align_time_ranges(df):
    """
    Align time ranges so all series have data for the same dates.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        Aligned DataFrame with common date range
    """
    print("\n📅 Aligning time ranges...")
    
    # Find the common date range where all series have data
    non_null_ranges = {}
    
    for column in df.columns:
        # Find first and last non-null values for each series
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
        
        print(f"✅ Aligned data shape: {df_aligned.shape}")
        
        return df_aligned
    else:
        print("❌ No valid data found")
        return df

def analyze_data_quality(df, df_cleaned):
    """
    Analyze data quality before and after cleaning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame
    df_cleaned : pandas.DataFrame
        Cleaned DataFrame
    """
    print("\n" + "="*60)
    print("📊 DATA QUALITY ANALYSIS")
    print("="*60)
    
    print(f"\nOriginal data:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
    print(f"\nCleaned data:")
    print(f"  Shape: {df_cleaned.shape}")
    print(f"  Date range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
    print(f"  Missing values: {df_cleaned.isnull().sum().sum()}")
    
    print(f"\nData coverage by series:")
    for col in df_cleaned.columns:
        coverage = (df_cleaned[col].notna().sum() / len(df_cleaned)) * 100
        print(f"  - {col}: {coverage:.1f}% coverage")

def plot_data_comparison(df_original, df_cleaned, save_plots=True):
    """
    Create comparison plots of original vs cleaned data.
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        Original DataFrame
    df_cleaned : pandas.DataFrame
        Cleaned DataFrame
    save_plots : bool
        Whether to save plots to files
    """
    print("\n📈 Creating comparison plots...")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # Create subplots for each series
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality: Original vs Cleaned', fontsize=16, fontweight='bold')
    
    for i, column in enumerate(df_cleaned.columns):
        ax = axes[i//2, i%2]
        
        # Plot original data
        ax.plot(df_original.index, df_original[column], 
               alpha=0.5, label='Original', color='gray', linewidth=1)
        
        # Plot cleaned data
        ax.plot(df_cleaned.index, df_cleaned[column], 
               label='Cleaned', color='blue', linewidth=2)
        
        ax.set_title(f'{column}', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('data_quality_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Comparison plot saved as 'data_quality_comparison.png'")
    
    plt.show()

def save_cleaned_data(df, filename='cleaned_macroeconomic_data.csv'):
    """
    Save the cleaned data to a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned DataFrame
    filename : str
        Output filename
    """
    try:
        df.to_csv(filename)
        print(f"💾 Cleaned data saved to '{filename}'")
        
        # Also save as Excel if possible
        try:
            df.to_excel(filename.replace('.csv', '.xlsx'))
            print(f"💾 Cleaned data also saved as '{filename.replace('.csv', '.xlsx')}'")
        except ImportError:
            print("ℹ️  Excel export requires openpyxl package")
            
    except Exception as e:
        print(f"❌ Error saving data: {e}")

def main():
    """
    Main function to execute the data preprocessing pipeline.
    """
    print("🚀 Macroeconomic Data Preprocessing Pipeline")
    print("="*60)
    
    # Step 1: Load and inspect data
    df = load_and_inspect_data()
    
    # Step 2: Resample to monthly frequency
    df_monthly = resample_to_monthly(df)
    
    # Step 3: Handle missing values
    df_cleaned = handle_missing_values(df_monthly, method='combined', limit=12)
    
    # Step 4: Align time ranges
    df_aligned = align_time_ranges(df_cleaned)
    
    # Step 5: Analyze data quality
    analyze_data_quality(df, df_aligned)
    
    # Step 6: Create comparison plots
    plot_data_comparison(df, df_aligned)
    
    # Step 7: Save cleaned data
    save_cleaned_data(df_aligned)
    
    # Display final data info
    print("\n" + "="*60)
    print("📋 FINAL CLEANED DATA SUMMARY")
    print("="*60)
    print(f"Shape: {df_aligned.shape}")
    print(f"Date range: {df_aligned.index.min()} to {df_aligned.index.max()}")
    print(f"Missing values: {df_aligned.isnull().sum().sum()}")
    
    print("\nFirst 10 rows of cleaned data:")
    print(df_aligned.head(10))
    
    print("\nLast 10 rows of cleaned data:")
    print(df_aligned.tail(10))
    
    return df_aligned

if __name__ == "__main__":
    # Run the preprocessing pipeline
    cleaned_data = main() 