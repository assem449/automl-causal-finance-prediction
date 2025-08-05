import pandas as pd
import numpy as np
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings('ignore')

def fetch_fred_data(api_key=None, start_date='1990-01-01'):
    """
    Fetch monthly macroeconomic data from FRED using fredapi library.
    
    Parameters:
    -----------
    api_key : str, optional
        FRED API key. If None, will try to use environment variable FRED_API_KEY
    start_date : str
        Start date for data retrieval in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with monthly macroeconomic data indexed by date
    """
    
    # Initialize FRED API client
    try:
        fred = Fred(api_key=api_key)
        print("✅ Successfully connected to FRED API")
    except Exception as e:
        print(f"❌ Error connecting to FRED API: {e}")
        print("Please make sure you have a valid FRED API key.")
        print("You can get one for free at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    
    # Define the series to fetch
    series_info = {
        'FEDFUNDS': 'Federal Funds Rate',
        'UNRATE': 'Unemployment Rate',
        'CPALTT01USM657N': 'Inflation Rate (CPI)',
        'DRSFRMACBS': 'Personal Loan Delinquency Rate'
    }
    
    # Dictionary to store the data
    data_dict = {}
    
    print(f"\n📊 Fetching monthly macroeconomic data from {start_date} to present...")
    
    # Fetch each series
    for series_id, description in series_info.items():
        try:
            print(f"  🔄 Fetching {description} ({series_id})...")
            
            # Get the data
            series_data = fred.get_series(series_id, observation_start=start_date)
            
            # Convert to DataFrame
            df_temp = pd.DataFrame(series_data, columns=[description])
            df_temp.index.name = 'Date'
            
            data_dict[description] = df_temp
            
            print(f"    ✅ Retrieved {len(df_temp)} observations")
            
        except Exception as e:
            print(f"    ❌ Error fetching {series_id}: {e}")
            continue
    
    # Combine all series into a single DataFrame
    if data_dict:
        # Start with the first series
        combined_df = list(data_dict.values())[0]
        
        # Add other series
        for description, df_temp in list(data_dict.values())[1:]:
            combined_df = combined_df.join(df_temp, how='outer')
        
        print(f"\n✅ Successfully created combined dataset with {len(combined_df)} observations")
        print(f"📅 Date range: {combined_df.index.min().strftime('%Y-%m-%d')} to {combined_df.index.max().strftime('%Y-%m-%d')}")
        
        return combined_df
    else:
        print("❌ No data was successfully retrieved")
        return None

def display_data_info(df):
    """
    Display information about the fetched data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with macroeconomic data
    """
    if df is None:
        return
    
    print("\n" + "="*60)
    print("📈 DATA SUMMARY")
    print("="*60)
    
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    missing_data = df.isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  - {col}: {missing} missing values")
        else:
            print(f"  - {col}: No missing values")
    
    print("\nSummary statistics:")
    print(df.describe())

def plot_data(df, save_plot=True):
    """
    Create plots of the macroeconomic data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with macroeconomic data
    save_plot : bool
        Whether to save the plot to a file
    """
    if df is None:
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Monthly Macroeconomic Indicators', fontsize=16, fontweight='bold')
    
    # Plot each series
    for i, (col, ax) in enumerate(zip(df.columns, axes.flat)):
        df[col].plot(ax=ax, linewidth=2, color=f'C{i}')
        ax.set_title(col, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('macroeconomic_indicators.png', dpi=300, bbox_inches='tight')
        print("📊 Plot saved as 'macroeconomic_indicators.png'")
    
    plt.show()

def save_data(df, filename='macroeconomic_data.csv'):
    """
    Save the data to a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with macroeconomic data
    filename : str
        Name of the file to save
    """
    if df is None:
        return
    
    try:
        df.to_csv(filename)
        print(f"💾 Data saved to '{filename}'")
    except Exception as e:
        print(f"❌ Error saving data: {e}")

def main():
    """
    Main function to execute the data fetching process.
    """
    print("🚀 FRED Macroeconomic Data Fetcher")
    print("="*50)
    
    # Get API key from environment variable
    api_key = os.getenv('FRED_API_KEY')
    start_date = os.getenv('START_DATE', '1990-01-01')
    
    if not api_key:
        print("⚠️  Warning: FRED_API_KEY not found in environment variables.")
        print("Please set your API key in the .env file or as an environment variable.")
        print("You can get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    
    # Fetch the data
    df = fetch_fred_data(api_key=api_key, start_date=start_date)
    
    if df is not None:
        # Display information about the data
        display_data_info(df)
        
        # Save the data
        save_data(df)
        
        # Create plots
        plot_data(df)
        
        # Display first few rows
        print("\n" + "="*60)
        print("📋 FIRST 10 ROWS OF DATA")
        print("="*60)
        print(df.head(10))
        
        return df
    else:
        print("❌ Failed to fetch data. Please check your API key and try again.")
        return None

if __name__ == "__main__":
    # Run the main function
    data = main() 