import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from scipy.optimize import minimize

def get_tbill_data(start, end):
    # Download 13-week T-bill (^IRX) data from yfinance
    if yf is None:
        raise ImportError("yfinance is required to fetch T-bill data.")
    tbill = yf.download("^IRX", start=start, end=end, progress=False)
    tbill.reset_index(inplace=True)  # flatten any multi-index
    if isinstance(tbill.columns, pd.MultiIndex):
        tbill.columns = tbill.columns.get_level_values(0)
    tbill.set_index("Date", inplace=True)
    tbill["Rate"] = tbill["Close"] / 100.0
    tbill["Daily_Rf"] = tbill["Rate"] / 252  # approximate daily risk-free
    return tbill[["Daily_Rf"]]

def load_asset_returns(csv_path):
    # Reads a CSV with columns [Date, Ticker, Actual_Return] and pivots by Date
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df.pivot(index="Date", columns="Ticker", values="Actual_Return")

def mean_cov_matrix(returns_df):
    mu = returns_df.mean()
    cov = returns_df.cov()
    return mu, cov

def find_tangency_portfolio(mu, cov, rf):
    # Maximize Sharpe ratio = (w^T mu - rf) / sqrt(w^T cov w), no shorting
    n = len(mu)
    init_w = np.ones(n) / n
    bounds = [(0,1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    def negative_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / (vol if vol > 1e-9 else 1e-9)
    res = minimize(negative_sharpe, init_w, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def find_global_min_variance(cov):
    # Find the portfolio with minimum variance subject to sum of weights=1, no shorting
    n = cov.shape[0]
    init_w = np.ones(n) / n
    bounds = [(0,1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)
    res = minimize(portfolio_vol, init_w, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def compute_portfolio_performance(weights, mu, cov, rf=0.0):
    ret = np.dot(weights, mu)
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / (vol if vol > 1e-9 else 1e-9)
    return vol, ret, sharpe

def plot_efficient_frontier(mu, cov, rf):
    # Plot the analytical Efficient Frontier, plus GMV and Tangency portfolios
    fig, ax = plt.subplots()
    one = np.ones(len(mu))
    inv_cov = np.linalg.inv(cov)
    A = one.T @ inv_cov @ one
    B = one.T @ inv_cov @ mu
    C = mu.T @ inv_cov @ mu
    Delta = A*C - B**2

    # Global Minimum Variance return
    r_min = B / A
    # r_max as an upper bound (1.5x the highest single-asset return)
    r_max = max(mu) * 1.5
    r_range = np.linspace(r_min, r_max, 100)
    # Frontier equation: sigma^2 = (A*r^2 - 2Br + C) / Delta
    frontier_vol = np.sqrt((A * r_range**2 - 2 * B * r_range + C) / Delta)
    ax.plot(frontier_vol, r_range, 'b-', label="Efficient Frontier")

    # Global Minimum Variance portfolio
    w_gmv = find_global_min_variance(cov)
    vol_gmv, ret_gmv, _ = compute_portfolio_performance(w_gmv, mu, cov, rf=0.0)
    ax.scatter([vol_gmv], [ret_gmv], c="green", marker="o", s=100, label="GMV Portfolio")

    print("\nGlobal Minimum Variance (GMV) Weights:")
    for asset, weight in zip(mu.index, w_gmv):
        print(f"  {asset}: {weight:.4f}")

    # Tangency Portfolio (analytical approach)
    w_tan = inv_cov @ (mu - rf * one)
    w_tan /= np.sum(w_tan)
    vol_tan, ret_tan, _ = compute_portfolio_performance(w_tan, mu, cov, rf)
    ax.scatter([vol_tan], [ret_tan], c="red", marker="*", s=200, label="Tangency Portfolio")

    print("\nTangency Portfolio Weights (Analytical):")
    for asset, weight in zip(mu.index, w_tan):
        print(f"  {asset}: {weight:.4f}")

    # Capital Market Line
    sharpe_tan = (ret_tan - rf) / vol_tan
    sigma_vals = np.linspace(0, vol_tan * 1.5, 100)
    cml = rf + sharpe_tan * sigma_vals
    ax.plot(sigma_vals, cml, 'k--', linewidth=2, label="Capital Market Line")

    ax.set_xlabel("Volatility (Std Dev)")
    ax.set_ylabel("Return")
    ax.set_title(f"Efficient Frontier (Analytical) with CAPM (Risk-Free = {rf:.2%})")
    ax.legend()
    plt.show()

def time_series_rebalance(crypto_csv, stock_csv, start_date, end_date, lookback=60):
    # 1) Load crypto and stock returns, combine into a single DataFrame
    crypto_df = load_asset_returns(crypto_csv)
    stock_df = load_asset_returns(stock_csv)
    asset_df = pd.concat([crypto_df, stock_df], axis=1)
    asset_df = asset_df[(asset_df.index >= start_date) & (asset_df.index <= end_date)]

    # 2) Load T-bill data for the same date range
    tbill_df = get_tbill_data(start_date, end_date)

    # 3) Join on Date index
    asset_df.index = pd.to_datetime(asset_df.index)
    tbill_df.index = pd.to_datetime(tbill_df.index)
    combined = asset_df.join(tbill_df, how="inner")
    dates = combined.index

    # 4) Time-series rolling rebalancing
    portfolio_val = 1.0
    portvals = []
    weights_dict = {}

    if len(dates) < lookback + 1:
        print("Not enough data to perform the rolling lookback.")
        return pd.DataFrame(), {}

    # Start logging portfolio from the first day we can rebalance
    portvals.append((dates[lookback], portfolio_val))

    for i in range(lookback, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        window_dates = dates[i - lookback : i]

        # 4a) Use the previous 'lookback' days to estimate mu, cov
        window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
        if window_returns.empty:
            continue
        mu_window, cov_window = mean_cov_matrix(window_returns)

        # 4b) Risk-free for the current day
        rf_daily = combined.loc[current_date, "Daily_Rf"]

        # 4c) Tangency portfolio for the next day
        w_tan = find_tangency_portfolio(mu_window, cov_window, rf_daily)
        weights_dict[current_date] = w_tan

        # 4d) Realized return from day i to i+1
        r_next = combined.loc[next_date, asset_df.columns].fillna(0.0)
        realized_ret = np.dot(w_tan, r_next)
        portfolio_val *= (1.0 + realized_ret)
        portvals.append((next_date, portfolio_val))

    # 5) Convert results to DataFrame, compute daily returns
    portvals_df = pd.DataFrame(portvals, columns=["Date", "PortfolioValue"]).set_index("Date")
    portvals_df["DailyRet"] = portvals_df["PortfolioValue"].pct_change().fillna(0.0)

    # 6) Performance stats
    daily_ret = portvals_df["DailyRet"]
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std(ddof=1)
    ann_sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    running_max = portvals_df["PortfolioValue"].cummax()
    max_drawdown = ((portvals_df["PortfolioValue"] / running_max) - 1).min()
    win_rate = (daily_ret > 0).sum() / (daily_ret != 0).sum()

    print("=== Final Portfolio Stats ===")
    print(f"Final Portfolio Value   : {portvals_df.iloc[-1]['PortfolioValue']:.4f}")
    print(f"Annualized Sharpe Ratio : {ann_sharpe:.4f}")
    print(f"Max Drawdown            : {max_drawdown:.2%}")
    print(f"Win Rate                : {win_rate:.2%}")

    # 7) Plot the analytical efficient frontier on the last rebalancing date
    if weights_dict:
        final_rebal_date = list(weights_dict.keys())[-1]
        window_dates = dates[dates <= final_rebal_date][-lookback:]
        window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
        mu_final, cov_final = mean_cov_matrix(window_returns)
        rf_final = combined.loc[final_rebal_date, "Daily_Rf"]
        plot_efficient_frontier(mu_final, cov_final, rf_final)
    else:
        print("No rebalancing steps were performed. Possibly not enough data after dropping NaNs.")

    return portvals_df, weights_dict

if __name__ == "__main__":
    start_date = datetime.datetime(2025, 1, 1)
    end_date = datetime.datetime(2025, 2, 28)
    portfolio_values, weights = time_series_rebalance("test_crypto.csv", "test_stocks.csv", start_date, end_date, lookback=32)