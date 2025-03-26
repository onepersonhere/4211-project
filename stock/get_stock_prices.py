import os
import sys
import subprocess
from datetime import datetime
import pandas as pd

# Ensure yfinance is installed
try:
    import yfinance as yf
except ImportError:
    print("yfinance not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

def fetch_stock_data(start_date, end_date, frequency="1m", category="train"):
    """
    Fetches stock data for MAG-7 tickers from Yahoo Finance.

    Parameters:
    - start_date (str): 'YYYY-MM-DD' format
    - end_date (str): 'YYYY-MM-DD' format
    - frequency (str): Interval (e.g., '1m', '2m', '5m', '15m')
    - category (str): 'train' or 'test' (used for saving paths)

    Returns:
    - pd.DataFrame: Stock data with standardized columns
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META"]
    print(f"\nFetching {frequency} data from {start_date} to {end_date} for {tickers} ({category})...")

    # Download from Yahoo Finance
    stock_data = yf.download(tickers, start=start_date, end=end_date, interval=frequency, group_by='ticker', auto_adjust=False, threads=True, progress=False)

    # Check if data is empty
    if stock_data.empty:
        print(f"Warning: No data returned for {frequency} from {start_date} to {end_date}")
        return pd.DataFrame()

    # Flatten MultiIndex and reset
    stock_data = stock_data.stack(level=1).reset_index()

    # Fix column names
    stock_data.rename(columns={"level_0": "Datetime", "level_1": "Ticker"}, inplace=True)

    # Select and reorder useful columns
    stock_data = stock_data[["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

    # Create folder and save file
    filename = f"stock_prices_{frequency}_{category}.csv"
    folder_path = f'./stock/data_raw/{category}/'
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    stock_data.to_csv(file_path, index=False)

    print(f"Saved {category} data to '{file_path}'")
    return stock_data

def fetch_for_train_and_test():
    # Define train and test date ranges
    train_start = "2025-01-01"
    train_end = "2025-02-28"
    test_start = "2025-03-01"
    test_end = "2025-03-26"

    # Define frequencies
    frequencies = ["2m", "5m", "15m"]

    for freq in frequencies:
        fetch_stock_data(start_date=train_start, end_date=train_end, frequency=freq, category="train")
        fetch_stock_data(start_date=test_start, end_date=test_end, frequency=freq, category="test")

if __name__ == "__main__":
    fetch_for_train_and_test()
