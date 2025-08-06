#!/usr/bin/env python3
"""
Example usage of the FRED macroeconomic data fetcher.

This script demonstrates how to use the individual functions
from fetch_fred_data.py for custom data analysis.
"""

from fetch_fred_data import fetch_fred_data, display_data_info, plot_data, save_data
import pandas as pd

def example_basic_usage():
    """Example of basic usage with default settings."""
    print("=== Basic Usage Example ===")
    
    # Fetch data from 2000 onwards
    df = fetch_fred_data(start_date='2000-01-01')
    
    if df is not None:
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print("\nFirst 5 rows:")
        print(df.head())
    
    return df

def example_custom_analysis():
    """Example of custom analysis on the fetched data."""
    print("\n=== Custom Analysis Example ===")
    
    # Fetch data
    df = fetch_fred_data(start_date='2010-01-01')
    
    if df is not None:
        # Calculate some basic statistics
        print("Recent statistics (last 5 years):")
        recent_data = df.tail(60)  # Last 5 years (60 months)
        
        for column in df.columns:
            print(f"\n{column}:")
            print(f"  Mean: {recent_data[column].mean():.2f}")
            print(f"  Std:  {recent_data[column].std():.2f}")
            print(f"  Min:  {recent_data[column].min():.2f}")
            print(f"  Max:  {recent_data[column].max():.2f}")
        
        # Calculate year-over-year changes
        print("\nYear-over-year changes (latest):")
        for column in df.columns:
            if len(df) >= 12:
                current = df[column].iloc[-1]
                year_ago = df[column].iloc[-13]  # 12 months ago
                if pd.notna(current) and pd.notna(year_ago):
                    change = ((current - year_ago) / year_ago) * 100
                    print(f"  {column}: {change:+.2f}%")
    
    return df

def example_data_export():
    """Example of exporting data in different formats."""
    print("\n=== Data Export Example ===")
    
    # Fetch data
    df = fetch_fred_data(start_date='2015-01-01')
    
    if df is not None:
        # Save as CSV
        save_data(df, 'recent_macro_data.csv')
        
        # Save as Excel (if openpyxl is available)
        try:
            df.to_excel('recent_macro_data.xlsx')
            print("✅ Data also saved as Excel file")
        except ImportError:
            print("ℹ️  Excel export requires openpyxl package")
        
        # Save as JSON
        df.to_json('recent_macro_data.json', date_format='iso')
        print("✅ Data also saved as JSON file")
    
    return df

def example_plotting():
    """Example of custom plotting."""
    print("\n=== Custom Plotting Example ===")
    
    # Fetch data
    df = fetch_fred_data(start_date='2015-01-01')
    
    if df is not None:
        # Create custom plots
        import matplotlib.pyplot as plt
        
        # Plot with custom styling
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for column in df.columns:
            ax.plot(df.index, df[column], label=column, linewidth=2)
        
        ax.set_title('Macroeconomic Indicators (2015-Present)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('custom_macro_plot.png', dpi=300, bbox_inches='tight')
        print("✅ Custom plot saved as 'custom_macro_plot.png'")
        plt.show()
    
    return df

if __name__ == "__main__":
    print("🚀 FRED Data Fetcher - Example Usage")
    print("="*50)
    
    # Run examples
    df1 = example_basic_usage()
    df2 = example_custom_analysis()
    df3 = example_data_export()
    df4 = example_plotting()
    
    print("\n✅ All examples completed!")
    print("\nTo run the full script with all features:")
    print("python fetch_fred_data.py") 