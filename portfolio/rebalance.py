import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from scipy.optimize import minimize

def get_tbill_data(start, end):
    if yf is None:
        raise ImportError("yfinance is required to fetch T-bill data.")
    tbill = yf.download("^IRX", start=start, end=end, progress=False)
    tbill.reset_index(inplace=True)
    if isinstance(tbill.columns, pd.MultiIndex):
        tbill.columns = tbill.columns.get_level_values(0)
    tbill.set_index("Date", inplace=True)
    tbill["Rate"] = tbill["Close"] / 100.0
    tbill["Daily_Rf"] = tbill["Rate"] / 252
    return tbill[["Daily_Rf"]]

def load_asset_returns(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df.pivot(index="Date", columns="Ticker", values="Actual_Return")

def mean_cov_matrix(returns_df):
    mu = returns_df.mean()
    cov = returns_df.cov()
    return mu, cov

def find_tangency_portfolio(mu, cov, rf):
    """
    Maximize Sharpe ratio = (w^T mu - rf) / sqrt(w^T cov w)
    subject to:
      1) sum(w) = 1
      2) sum(|w|) <= 4  (25% maintenance margin)
      (no explicit bounds on w_i => can be negative, i.e. shorting allowed)
    """
    n = len(mu)
    init_w = np.ones(n) / n  # initial guess

    def negative_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / (vol if vol > 1e-9 else 1e-9)

    # sum(w) = 1
    cons = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        # sum of absolute weights <= 4
        {"type": "ineq", "fun": lambda w: 4.0 - np.sum(np.abs(w))}
    ]

    # No explicit bounds => shorting is allowed
    res = minimize(negative_sharpe, init_w, method="SLSQP", constraints=cons)
    return res.x

def find_global_min_variance(cov):
    """
    Minimize volatility = sqrt(w^T cov w)
    subject to:
      1) sum(w) = 1
      2) sum(|w|) <= 4  (25% maintenance margin)
    """
    n = cov.shape[0]
    init_w = np.ones(n) / n

    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)

    cons = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: 4.0 - np.sum(np.abs(w))}
    ]

    res = minimize(portfolio_vol, init_w, method="SLSQP", constraints=cons)
    return res.x

def compute_portfolio_performance(weights, mu, cov, rf=0.0):
    ret = np.dot(weights, mu)
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / (vol if vol > 1e-9 else 1e-9)
    return vol, ret, sharpe

def plot_efficient_frontier(mu, cov, rf):
    fig, ax = plt.subplots()
    one = np.ones(len(mu))
    inv_cov = np.linalg.inv(cov)
    A = one.T @ inv_cov @ one
    B = one.T @ inv_cov @ mu
    C = mu.T @ inv_cov @ mu
    Delta = A*C - B**2

    # r_min
    r_min = B / A
    r_max = max(mu) * 1.5
    r_range = np.linspace(r_min, r_max, 100)
    # standard formula for unconstrained frontier
    # but we also want to enforce sum(|w|) <= 4 => we can't just do a direct formula
    # so let's do a param-sweep approach with constraints, to reflect the no-limit shorting with margin
    # We'll create a series of target returns and solve for min vol subject to sum(|w|)<=4
    frontier_points = []
    for r_target in r_range:
        # Minimizing volatility subject to sum(w)=1, sum(|w|)<=4, and w^T mu = r_target
        def objective(w):
            return np.sqrt(w @ cov @ w)
        cons = [
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: 4.0 - np.sum(np.abs(w))},
            {"type": "eq",   "fun": lambda w: np.dot(w, mu) - r_target}
        ]
        n = len(mu)
        w0 = np.ones(n)/n
        res = minimize(objective, w0, constraints=cons, method="SLSQP")
        if res.success:
            vol_ = np.sqrt(res.x @ cov @ res.x)
            frontier_points.append((vol_, r_target))
    if frontier_points:
        frontier_points = np.array(frontier_points)
        ax.plot(frontier_points[:,0], frontier_points[:,1], 'b-', label="Efficient Frontier")
    else:
        print("No feasible frontier points found under margin constraints.")

    # Global Minimum Variance portfolio (with margin constraints)
    w_gmv = find_global_min_variance(cov)
    vol_gmv, ret_gmv, _ = compute_portfolio_performance(w_gmv, mu, cov, 0.0)
    ax.scatter([vol_gmv], [ret_gmv], c="green", marker="o", s=100, label="GMV Portfolio")
    print("\nGlobal Minimum Variance (GMV) Weights:")
    for asset, weight in zip(mu.index, w_gmv):
        print(f"  {asset}: {weight:.4f}")

    # Tangency Portfolio (with margin constraints)
    w_tan = find_tangency_portfolio(mu, cov, rf)
    vol_tan, ret_tan, _ = compute_portfolio_performance(w_tan, mu, cov, rf)
    ax.scatter([vol_tan], [ret_tan], c="red", marker="*", s=200, label="Tangency Portfolio")
    print("\nTangency Portfolio Weights (Margin-Constrained):")
    for asset, weight in zip(mu.index, w_tan):
        print(f"  {asset}: {weight:.4f}")

    # Capital Market Line (from (0, rf) to tangency)
    sharpe_tan = (ret_tan - rf)/(vol_tan if vol_tan>1e-9 else 1e-9)
    sigmas = np.linspace(0, vol_tan*1.5, 100)
    cml = rf + sharpe_tan*sigmas
    ax.plot(sigmas, cml, 'k--', linewidth=2, label="Capital Market Line")

    ax.set_xlabel("Volatility (Std Dev)")
    ax.set_ylabel("Return")
    ax.set_title(f"Efficient Frontier (Margin=25%) with CAPM (Risk-Free = {rf:.2%})")
    ax.legend()
    plt.show()

def time_series_rebalance(crypto_csv, stock_csv, start_date, end_date, lookback=60):
    # Load data
    crypto_df = load_asset_returns(crypto_csv)
    stock_df = load_asset_returns(stock_csv)
    asset_df = pd.concat([crypto_df, stock_df], axis=1)
    asset_df = asset_df[(asset_df.index >= start_date) & (asset_df.index <= end_date)]

    tbill_df = get_tbill_data(start_date, end_date)
    asset_df.index = pd.to_datetime(asset_df.index)
    tbill_df.index = pd.to_datetime(tbill_df.index)
    combined = asset_df.join(tbill_df, how="inner")
    dates = combined.index

    if len(dates) < lookback + 1:
        print("Not enough data to perform the rolling lookback.")
        return pd.DataFrame(), {}

    portfolio_val = 1.0
    portvals = []
    weights_dict = {}

    portvals.append((dates[lookback], portfolio_val))

    for i in range(lookback, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        window_dates = dates[i - lookback : i]
        window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
        if window_returns.empty:
            continue

        mu_window, cov_window = mean_cov_matrix(window_returns)
        rf_daily = combined.loc[current_date, "Daily_Rf"]
        # find tangency with margin constraint
        w_tan = find_tangency_portfolio(mu_window, cov_window, rf_daily)
        weights_dict[current_date] = w_tan

        r_next = combined.loc[next_date, asset_df.columns].fillna(0.0)
        realized_ret = np.dot(w_tan, r_next)
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
    win_rate = (daily_ret>0).sum() / (daily_ret!=0).sum()

    print("=== Final Portfolio Stats ===")
    print(f"Final Portfolio Value   : {portvals_df.iloc[-1]['PortfolioValue']:.4f}")
    print(f"Annualized Sharpe Ratio : {ann_sharpe:.4f}")
    print(f"Max Drawdown            : {max_drawdown:.2%}")
    print(f"Win Rate                : {win_rate:.2%}")

    # Plot the margin-constrained frontier on the last rebalance date
    if weights_dict:
        final_rebal_date = list(weights_dict.keys())[-1]
        window_dates = dates[dates <= final_rebal_date][-lookback:]
        window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
        mu_final, cov_final = mean_cov_matrix(window_returns)
        rf_final = combined.loc[final_rebal_date, "Daily_Rf"]
        plot_efficient_frontier(mu_final, cov_final, rf_final)
    else:
        print("No rebalancing steps were performed.")

    return portvals_df, weights_dict

if __name__ == "__main__":
    start_date = datetime.datetime(2025, 1, 1)
    end_date   = datetime.datetime(2025, 2, 28)
    portfolio_values, weights = time_series_rebalance(
        "test_crypto.csv",
        "test_stocks.csv",
        start_date,
        end_date,
        lookback=32
    )