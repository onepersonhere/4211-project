import pandas as pd
import numpy as np
import datetime


def generate_intraday_timestamps(start_date, end_date, freq="2min"):
    """
    Generate a DatetimeIndex of intraday timestamps for business days only,
    with timestamps from 9:30 to 16:00 at the specified frequency.
    """
    # Generate business days between start_date and end_date
    business_days = pd.date_range(start=start_date, end=end_date, freq="B")
    timestamps = []
    for day in business_days:
        day_str = day.strftime("%Y-%m-%d")
        # Create intraday timestamps from 9:30 to 16:00
        intraday = pd.date_range(start=f"{day_str} 09:30", end=f"{day_str} 16:00", freq=freq)
        timestamps.extend(intraday)
    return pd.DatetimeIndex(timestamps)


def generate_data(tickers, start_date, end_date, freq='2min'):
    """
    Generate realistic intraday test data for the given tickers.
    Only timestamps during market hours (9:30 to 16:00 on business days) are generated.
    Returns a DataFrame with columns: Date, Ticker, Close_Price, Actual_Return, Predicted_Return.

    For realistic intraday returns, a small standard deviation is used.
    """
    # Get timestamps for business days during market hours.
    timestamps = generate_intraday_timestamps(start_date, end_date, freq=freq)
    data_rows = []

    # For reproducible results, uncomment the next line if desired.
    np.random.seed(42)

    for ts in timestamps:
        for ticker in tickers:
            # Simulate a closing price between 50 and 500.
            close_price = np.random.uniform(1, 500)
            # Simulate realistic intraday returns (e.g., standard deviation of 0.1%).
            actual_return = np.random.normal(loc=0, scale=0.02)
            predicted_return = np.random.normal(loc=0, scale=0.005)
            data_rows.append({
                "Date": ts,
                "Ticker": ticker,
                "Close_Price": close_price,
                "Actual_Return": actual_return,
                "Predicted_Return": predicted_return
            })

    df = pd.DataFrame(data_rows)
    df.sort_values(["Date", "Ticker"], inplace=True)
    return df


def main():
    # Define tickers for crypto and stocks.
    crypto_tickers = ["BTC", "ETH", "SOL"]
    stock_tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]

    # Define the date range.
    start_date = datetime.datetime(2025, 3, 1)
    end_date = datetime.datetime(2025, 3, 24)

    # Generate intraday data (only during market hours) for both crypto and stocks.
    # Note: Even though crypto is normally traded 24/7, to have overlapping timestamps
    # for portfolio rebalancing, we simulate crypto data during the same active hours.
    crypto_df = generate_data(crypto_tickers, start_date, end_date, freq='2min')
    crypto_df.to_csv("test_crypto.csv", index=False)

    stock_df = generate_data(stock_tickers, start_date, end_date, freq='2min')
    stock_df.to_csv("test_stocks.csv", index=False)

    print("Generated test_crypto.csv and test_stocks.csv with realistic intraday market hours data.")


if __name__ == "__main__":
    main()