# FRED Macroeconomic Data Fetcher

This Python script fetches monthly macroeconomic data from the Federal Reserve Economic Data (FRED) API using the `fredapi` library.

## Data Series Included

The script fetches the following monthly macroeconomic indicators:

- **Federal Funds Rate (FEDFUNDS)**: The interest rate at which depository institutions trade federal funds
- **Unemployment Rate (UNRATE)**: The percentage of the labor force that is unemployed
- **Inflation Rate (CPALTT01USM657N)**: Consumer Price Index for All Urban Consumers
- **Personal Loan Delinquency Rate (DRSFRMACBS)**: Delinquency rate on single-family residential mortgages

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a FRED API Key

1. Go to [FRED API Key Registration](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Sign up for a free account
3. Generate an API key

### 3. Set Your API Key

You have several options:

**Option A: Using .env file (Recommended)**
```bash
python setup_env.py
```
This will guide you through creating a `.env` file with your API key.

**Option B: Manual .env file creation**
Create a `.env` file in the project directory:
```bash
FRED_API_KEY=your_api_key_here
START_DATE=1990-01-01
```

**Option C: Environment Variable**
```bash
export FRED_API_KEY="your_api_key_here"
```

**Option D: Direct in Script**
Edit `fetch_fred_data.py` and modify the api_key variable.

## Usage

### Test Your API Key

Before running the main script, you can test if your API key is working:

```bash
python setup_env.py --test
```

### Run the Complete Script

```bash
python fetch_fred_data.py
```

This will:
- Fetch all macroeconomic data from 1990 to present
- Display data summary and statistics
- Save data to `macroeconomic_data.csv`
- Generate plots saved as `macroeconomic_indicators.png`
- Show the first 10 rows of data

### Use Individual Functions

You can also import and use individual functions:

```python
from fetch_fred_data import fetch_fred_data, display_data_info, plot_data, save_data

# Fetch data
df = fetch_fred_data(api_key="your_key", start_date='2000-01-01')

# Display information
display_data_info(df)

# Save to file
save_data(df, 'my_data.csv')

# Create plots
plot_data(df)
```

## Output Files

- `macroeconomic_data.csv`: CSV file containing all the fetched data
- `macroeconomic_indicators.png`: Visualization of all four indicators over time

## Data Structure

The resulting DataFrame has:
- **Index**: Date (monthly frequency)
- **Columns**: 
  - Federal Funds Rate
  - Unemployment Rate
  - Inflation Rate (CPI)
  - Personal Loan Delinquency Rate

## Example Output

```
🚀 FRED Macroeconomic Data Fetcher
==================================================
✅ Successfully connected to FRED API

📊 Fetching monthly macroeconomic data from 1990-01-01 to present...
  🔄 Fetching Federal Funds Rate (FEDFUNDS)...
    ✅ Retrieved 408 observations
  🔄 Fetching Unemployment Rate (UNRATE)...
    ✅ Retrieved 408 observations
  🔄 Fetching Inflation Rate (CPI) (CPALTT01USM657N)...
    ✅ Retrieved 408 observations
  🔄 Fetching Personal Loan Delinquency Rate (DRSFRMACBS)...
    ✅ Retrieved 408 observations

✅ Successfully created combined dataset with 408 observations
📅 Date range: 1990-01-01 to 2024-01-01
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure you have a valid FRED API key
2. **Network Issues**: Check your internet connection
3. **Rate Limits**: FRED has rate limits; if you hit them, wait a moment and try again

### Error Messages

- `Error connecting to FRED API`: Check your API key
- `Error fetching [SERIES_ID]`: The specific series might be temporarily unavailable

## Dependencies

- `fredapi`: FRED API client
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Plotting
- `seaborn`: Enhanced plotting

## License

This project is open source and available under the MIT License. 