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

def fetch_stock_data(end_date=None, n_days=7, frequency="1m"):
    """
    Fetches stock data for the MAG-7 tickers from Yahoo Finance.

    Parameters:
    - end_date (str): The end date in 'YYYY-MM-DD' format (default: today).
    - n_days (int): The number of days before end_date to fetch (default: 7).
    - frequency (str): The interval of data (e.g., '1m', '2m', '5m', '15m').

    Returns:
    - DataFrame: Formatted stock data with columns: ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
    """
    
    # List of MAG-7 tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META"]

    # Set end date (default: today)
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Calculate start date
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n_days)).strftime('%Y-%m-%d')

    print(f"Fetching {frequency} data from {start_date} to {end_date} for {tickers}...")

    # Fetch stock data
    stock_data = yf.download(tickers, start=start_date, end=end_date, interval=frequency)

    # Reset multi-index and flatten column names
    stock_data = stock_data.stack(level=1).reset_index()

    # Rename columns properly
    stock_data.rename(columns={"level_1": "Ticker", "Datetime": "Datetime"}, inplace=True)

    # Ensure correct column order
    stock_data = stock_data[["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

    # Save to CSV
    filename = f"stock_prices_{frequency}"
    file_path = f'./stock/data_raw/{filename}.csv'

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the DataFrame to the specified path
    stock_data.to_csv(file_path, index=False)

    print(f"Stock data successfully downloaded and saved as '{filename}.csv'.")
    
    return stock_data

# Main function to run for multiple frequencies
def fetch_for_multiple_frequencies():
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Define frequencies and corresponding n_days
    frequencies = {
        "1m": 7,
        "2m": 59,
        "5m": 59,
        "15m": 59
    }

    # Fetch and save data for each frequency
    for frequency, n_days in frequencies.items():
        fetch_stock_data(end_date=end_date, n_days=n_days, frequency=frequency)

# Run for all frequencies
if __name__ == "__main__":
    fetch_for_multiple_frequencies()
