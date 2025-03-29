import pandas as pd
import numpy as np
import datetime


def generate_data(tickers, start_date, end_date, freq='D'):
    # Create a date range from start_date to end_date with the specified frequency (daily by default)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # We'll store each row in a list of dicts
    data_rows = []

    # For reproducible random numbers, you can uncomment the next line:
    # np.random.seed(42)

    for date in dates:
        for ticker in tickers:
            # Generate a random "Close_Price" (e.g. between 100 and 300)
            close_price = np.random.uniform(100, 300)
            # Generate an actual return (small random values, e.g. ~±1%)
            actual_return = np.random.uniform(-0.01, 0.01)
            # Generate a predicted return (another random small value)
            predicted_return = np.random.uniform(-0.01, 0.01)

            data_rows.append({
                "Date": date,
                "Ticker": ticker,
                "Close_Price": close_price,
                "Actual_Return": actual_return,
                "Predicted_Return": predicted_return
            })

    df = pd.DataFrame(data_rows)
    # Sort by Date, then Ticker for neatness
    df.sort_values(["Date", "Ticker"], inplace=True)
    return df


def main():
    # Define some example crypto and stock tickers
    crypto_tickers = ["BTC", "ETH", "SOL"]
    stock_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]

    # Define your date range
    start_date = datetime.datetime(2025, 1, 1)
    end_date = datetime.datetime(2025, 2, 28)

    # Generate crypto data
    crypto_df = generate_data(crypto_tickers, start_date, end_date, freq='D')
    # Write to CSV
    crypto_df.to_csv("test_crypto.csv", index=False)

    # Generate stock data
    stock_df = generate_data(stock_tickers, start_date, end_date, freq='D')
    # Write to CSV
    stock_df.to_csv("test_stocks.csv", index=False)

    print("Generated test_crypto.csv and test_stocks.csv successfully.")


if __name__ == "__main__":
    main()