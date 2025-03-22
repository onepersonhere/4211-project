import os
import sys
import subprocess
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    print("yfinance not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

def fetch_crypto_data(end_date=None, n_days=7, frequency="1h"):
    """
    Fetches cryptocurrency data for top cryptocurrencies from Yahoo Finance.

    Parameters:
    - end_date (str): The end date in 'YYYY-MM-DD' format (default: today).
    - n_days (int): The number of days before end_date to fetch (default: 7).
    - frequency (str): The interval of data (e.g., '1h', '4h', '1d').

    Returns:
    - DataFrame: Formatted crypto data with columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
    """
    
    # List of top cryptocurrency tickers (Yahoo Finance format)
    tickers = [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"
    ]

    # Set end date (default: today)
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Calculate start date
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n_days)).strftime('%Y-%m-%d')

    print(f"Fetching {frequency} data from {start_date} to {end_date} for {tickers}...")

    # Fetch cryptocurrency data
    try:
        crypto_data = yf.download(tickers, start=start_date, end=end_date, interval=frequency)
        
        # If data is empty, raise an exception
        if crypto_data.empty:
            raise ValueError(f"No data returned for interval {frequency}")
        
        # Reset multi-index and flatten column names - handle future_stack parameter correctly
        try:
            # Try with the new parameter (newer pandas versions)
            crypto_data = crypto_data.stack(level=1, future_stack=True).reset_index()
        except TypeError:
            # Fall back to old method for older pandas versions
            crypto_data = crypto_data.stack(level=1).reset_index()
        
        # Check if 'Date' or 'Datetime' column exists
        time_col = None
        for col in ['Date', 'Datetime', 'date', 'datetime']:
            if col in crypto_data.columns:
                time_col = col
                break
                
        if time_col is None:
            print(f"Warning: Could not find date/time column. Available columns: {crypto_data.columns.tolist()}")
            # If we can't find a date column, use the first column and assume it's the date
            time_col = crypto_data.columns[0]
        
        # Rename columns properly
        crypto_data.rename(columns={"level_1": "Ticker", time_col: "Date"}, inplace=True)
        
        # Ensure all required columns exist
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
        available_cols = crypto_data.columns.tolist()
        
        for col in required_cols:
            if col not in available_cols:
                print(f"Warning: Column '{col}' not found in data")
        
        # Filter to only columns that exist
        cols_to_use = [col for col in required_cols if col in available_cols]
        crypto_data = crypto_data[cols_to_use]
        
        # Save to CSV
        filename = f"crypto_prices_{frequency}"
        file_path = f'./data_raw/{filename}.csv'
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the DataFrame to the specified path
        crypto_data.to_csv(file_path, index=False)
        
        print(f"Cryptocurrency data successfully downloaded and saved as '{filename}.csv'.")
        
        return crypto_data
        
    except Exception as e:
        print(f"Error fetching data for interval {frequency}: {e}")
        return None

# Main function to run for multiple frequencies
def fetch_for_multiple_frequencies():
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Define frequencies and corresponding n_days
    # Only using frequencies supported by Yahoo Finance
    frequencies = {
        "1m": 7,      # 1 minute data for 1 week (last 7 days)
        "5m": 7,      # 5 minute data for 1 week
        "15m": 7,     # 15 minute data for 1 week
        "30m": 7,     # 30 minute data for 1 week
        "60m": 7,     # 60 minute data for 1 week
        "1h": 7,      # 1 hour data for 1 week (same as 60m)
        "1d": 90,     # Daily data for 3 months
        "1wk": 365,   # Weekly data for 1 year
        "1mo": 1095   # Monthly data for 3 years
    }

    # Fetch and save data for each frequency
    for frequency, n_days in frequencies.items():
        data = fetch_crypto_data(end_date=end_date, n_days=n_days, frequency=frequency)
        if data is None:
            print(f"Skipping failed frequency: {frequency}")

# Run for all frequencies
if __name__ == "__main__":
    fetch_for_multiple_frequencies()