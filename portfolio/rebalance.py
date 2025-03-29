import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

def get_tbill_data(start, end):
    # Download T‑bill data from yfinance (ticker '^IRX')
    if yf is None:
        raise ImportError("yfinance is required to fetch T‑bill data.")
    tbill = yf.download("^IRX", start=start, end=end, progress=False)
    tbill.reset_index(inplace=True)  # ensure a flat index
    # Flatten columns if they are a MultiIndex (this removes extra levels)
    if isinstance(tbill.columns, pd.MultiIndex):
        tbill.columns = tbill.columns.get_level_values(0)
    tbill.set_index("Date", inplace=True)  # set Date as index to match asset data
    tbill["Rate"] = tbill["Close"] / 100.0  # convert percentage to decimal
    tbill["Daily_Rf"] = tbill["Rate"] / 252  # approximate daily risk‑free rate
    return tbill[["Daily_Rf"]]

def load_asset_returns(csv_path):
    # Read CSV with columns: Date, Ticker, Actual_Return and pivot so Date is index and Ticker columns
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df.pivot(index="Date", columns="Ticker", values="Actual_Return")

def mean_cov_matrix(returns_df):
    # Calculate mean returns and covariance matrix from historical data
    mu = returns_df.mean()
    cov = returns_df.cov()
    return mu, cov

def find_tangency_portfolio(mu, cov, rf):
    # Use SLSQP to find weights that maximize the Sharpe ratio relative to rf; no shorting allowed
    from scipy.optimize import minimize
    n = len(mu)
    init_w = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    def negative_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -(ret - rf) / (vol if vol > 1e-9 else 1e-9)
    res = minimize(negative_sharpe, init_w, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x

def compute_portfolio_performance(weights, mu, cov, rf=0.0):
    # Calculate portfolio return, volatility, and Sharpe ratio (using rf)
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    sharpe = (ret - rf) / (vol if vol > 1e-9 else 1e-9)
    return vol, ret, sharpe

def plot_efficient_frontier(mu, cov, rf, ax=None):
    # Compute the efficient frontier analytically via matrix manipulation
    if ax is None:
        fig, ax = plt.subplots()
    one = np.ones(len(mu))
    inv_cov = np.linalg.inv(cov)
    A = one.T @ inv_cov @ one
    B = one.T @ inv_cov @ mu
    C = mu.T @ inv_cov @ mu
    Delta = A * C - B**2
    r_min = B / A  # return of the global minimum variance portfolio
    r_max = max(mu) * 1.5  # set an upper bound for target returns
    r_range = np.linspace(r_min, r_max, 100)
    frontier_vol = np.sqrt((A * r_range**2 - 2 * B * r_range + C) / Delta)
    ax.plot(frontier_vol, r_range, 'b-', label="Efficient Frontier")
    # Compute tangency portfolio weights analytically
    w_tan = inv_cov @ (mu - rf * one)
    w_tan /= np.sum(w_tan)
    vol_tan, ret_tan, _ = compute_portfolio_performance(w_tan, mu, cov, rf)
    ax.scatter([vol_tan], [ret_tan], c="red", marker="*", s=200, label="Tangency Portfolio")
    sharpe_tan = (ret_tan - rf) / vol_tan
    sigma_vals = np.linspace(0, vol_tan * 1.5, 100)
    cml = rf + sharpe_tan * sigma_vals
    ax.plot(sigma_vals, cml, 'k--', linewidth=2, label="Capital Market Line")
    ax.set_xlabel("Volatility (Std Dev)")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier (Analytical) with CAPM (Risk-Free = {:.2%})".format(rf))
    ax.legend()
    plt.show()

def time_series_rebalance(crypto_csv, stock_csv, start_date, end_date, lookback=60):
    # Load crypto and stock returns from CSVs and combine them
    crypto_df = load_asset_returns(crypto_csv)
    stock_df = load_asset_returns(stock_csv)
    asset_df = pd.concat([crypto_df, stock_df], axis=1)
    asset_df = asset_df[(asset_df.index >= start_date) & (asset_df.index <= end_date)]
    tbill_df = get_tbill_data(start_date, end_date)  # get risk-free rates

    # Ensure both DataFrames have a single-level Date index for merging
    asset_df.index = pd.to_datetime(asset_df.index)
    tbill_df.index = pd.to_datetime(tbill_df.index)
    combined = asset_df.join(tbill_df, how="inner")  # join on Date index
    dates = combined.index

    portfolio_val = 1.0  # starting portfolio value
    portvals = [(dates[lookback], portfolio_val)]
    weights_dict = {}
    for i in range(lookback, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        window_dates = dates[i - lookback : i]
        window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
        if window_returns.empty:
            continue
        mu_window, cov_window = mean_cov_matrix(window_returns)
        rf_daily = combined.loc[current_date, "Daily_Rf"]
        w_tan = find_tangency_portfolio(mu_window, cov_window, rf_daily)
        weights_dict[current_date] = w_tan
        r_next = combined.loc[next_date, asset_df.columns]
        realized_ret = np.dot(w_tan, r_next.fillna(0))
        portfolio_val *= (1.0 + realized_ret)
        portvals.append((next_date, portfolio_val))
    portvals_df = pd.DataFrame(portvals, columns=["Date", "PortfolioValue"]).set_index("Date")
    portvals_df["DailyRet"] = portvals_df["PortfolioValue"].pct_change().fillna(0.0)
    daily_ret = portvals_df["DailyRet"]
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std(ddof=1)
    ann_sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    running_max = portvals_df["PortfolioValue"].cummax()
    max_drawdown = ((portvals_df["PortfolioValue"] / running_max) - 1).min()
    win_rate = (daily_ret > 0).sum() / (daily_ret != 0).sum()
    print("=== Portfolio Performance ===")
    print("Final Portfolio Value: {:.4f}".format(portvals_df.iloc[-1]["PortfolioValue"]))
    print("Annualized Sharpe Ratio: {:.4f}".format(ann_sharpe))
    print("Max Drawdown: {:.2%}".format(max_drawdown))
    print("Win Rate: {:.2%}".format(win_rate))
    final_rebal_date = list(weights_dict.keys())[-1]
    window_dates = dates[dates <= final_rebal_date][-lookback:]
    window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
    mu_final, cov_final = mean_cov_matrix(window_returns)
    rf_final = combined.loc[final_rebal_date, "Daily_Rf"]
    plot_efficient_frontier(mu_final, cov_final, rf_final)
    return portvals_df, weights_dict

if __name__ == "__main__":
    start_date = datetime.datetime(2025, 1, 1)
    end_date = datetime.datetime(2025, 2, 28)
    portfolio_values, weights = time_series_rebalance("../crypto/copy/returns.csv", "../stock/returns.csv", start_date, end_date, lookback=32)