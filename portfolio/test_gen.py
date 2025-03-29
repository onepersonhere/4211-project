import pandas as pd
import numpy as np
import datetime

def generate_data(tickers, start_date, end_date, freq='D'):
    # Create a date range from start_date to end_date with the specified frequency
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    data_rows = []

    # For reproducible random numbers, you could uncomment the next line:
    # np.random.seed(42)

    for date in dates:
        for ticker in tickers:
            # Wider close price range => more variability
            close_price = np.random.uniform(50, 500)
            # Larger random returns (e.g., ±5%)
            actual_return = np.random.uniform(-0.20, 0.20)
            predicted_return = np.random.uniform(-0.20, 0.20)

            data_rows.append({
                "Date": date,
                "Ticker": ticker,
                "Close_Price": close_price,
                "Actual_Return": actual_return,
                "Predicted_Return": predicted_return
            })

    df = pd.DataFrame(data_rows)
    df.sort_values(["Date", "Ticker"], inplace=True)
    return df

def main():
    crypto_tickers = ["BTC", "ETH", "SOL"]
    stock_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]

    start_date = datetime.datetime(2025, 1, 1)
    end_date   = datetime.datetime(2025, 2, 28)

    # Generate crypto data with higher variance
    crypto_df = generate_data(crypto_tickers, start_date, end_date, freq='D')
    crypto_df.to_csv("test_crypto.csv", index=False)

    # Generate stock data with higher variance
    stock_df = generate_data(stock_tickers, start_date, end_date, freq='D')
    stock_df.to_csv("test_stocks.csv", index=False)

    print("Generated test_crypto.csv and test_stocks.csv with higher variance successfully.")

if __name__ == "__main__":
    main()