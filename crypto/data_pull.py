import requests
import time
import pandas as pd
import os

def to_milliseconds(date_str):
    return int(time.mktime(time.strptime(date_str, "%Y-%m-%d %H:%M:%S"))) * 1000

# Define timeframe
start_time = to_milliseconds("2025-01-01 00:00:00")  # Start date
end_time = to_milliseconds("2025-03-01 00:00:00")    # End date

# Binance API URL
url = "https://api.binance.com/api/v3/klines"

# List of coins
coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# Create directory for saving data
os.makedirs("market_data", exist_ok=True)

# Maximum limit per request
max_limit = 1000

# Iterate over each coin
for coin in coins:
    all_data = []
    current_start_time = start_time
    
    while current_start_time < end_time:
        params = {
            "symbol": coin,
            "interval": "1m",  # 1-minute timeframe
            "startTime": current_start_time,
            "endTime": end_time,
            "limit": max_limit  # Max limit per request
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break  # Stop if no more data
            all_data.extend(data)
            
            # Move the start time forward based on the last received timestamp
            current_start_time = data[-1][0] + 1  # Add 1ms to avoid duplicates
        else:
            print(f"Error fetching {coin} data: {response.status_code}, {response.text}")
            break
    
    # Convert response to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
    ])
    
    # Convert timestamps to readable dates
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
    
    # Add coin column
    df["Coin"] = coin[:-4]  # Remove 'USDT' from symbol
    
    # Keep only relevant columns
    df = df[["Open Time", "Open", "High", "Low", "Close", "Volume", "Number of Trades", "Coin"]]
    
    output_dir = "market_data"
    os.makedirs(output_dir, exist_ok=True)
    # Save to CSV file in market_data folder
    filename = os.path.join(output_dir, f"{coin[:-4]}_1m_data.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {coin} data to {filename}")
