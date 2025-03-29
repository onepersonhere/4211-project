import pandas as pd
import datetime
import yfinance as yf


def get_tbill_data(start, end):
    # Download 13-week T-bill (^IRX) data from yfinance
    tbill = yf.download("^IRX", start=start, end=end, progress=False)
    tbill.reset_index(inplace=True)  # flatten any multi-index
    if isinstance(tbill.columns, pd.MultiIndex):
        tbill.columns = tbill.columns.get_level_values(0)
    tbill.set_index("Date", inplace=True)
    # 'Close' is typically the annualized yield (e.g. 5.0 => 5% per year)
    tbill["Rate"] = tbill["Close"] / 100.0
    # Daily risk-free rate used for rebalancing
    tbill["Daily_Rf"] = tbill["Rate"] / 252
    return tbill[["Rate", "Daily_Rf"]]


def load_asset_returns(csv_path):
    # Reads a CSV with columns [Date, Ticker, Actual_Return] and pivots by Date
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df.pivot(index="Date", columns="Ticker", values="Actual_Return")


def mean_cov_matrix(returns_df):
    mu = returns_df.mean()
    cov = returns_df.cov()
    return mu, cov


def find_tangency_portfolio(mu, cov, rf, max_exposure):
    """
    Maximize Sharpe ratio = (w^T mu - rf) / sqrt(w^T cov w)
    subject to:
      1) sum(w) = 1  (fully invested)
      2) sum(|w|) <= max_exposure   (allows shorting, with maintenance margin = 1/max_exposure)
    """
    n = len(mu)
    init_w = np.ones(n) / n

    def negative_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -(ret - rf) / (vol if vol > 1e-9 else 1e-9)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    res = minimize(negative_sharpe, init_w, method="SLSQP", constraints=cons)
    return res.x


def find_global_min_variance(cov, max_exposure):
    """
    Minimize volatility = sqrt(w^T cov w)
    subject to:
      1) sum(w) = 1,
      2) sum(|w|) <= max_exposure
    """
    n = cov.shape[0]
    init_w = np.ones(n) / n

    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    res = minimize(portfolio_vol, init_w, method="SLSQP", constraints=cons)
    return res.x


def compute_portfolio_performance(weights, mu, cov, rf=0.0):
    ret = np.dot(weights, mu)
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / (vol if vol > 1e-9 else 1e-9)
    return vol, ret, sharpe



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def plot_efficient_frontier(mu, cov, rf, max_exposure, resolution=1000, spline_kind='cubic'):
    """
    Margin-constrained frontier with a spline that passes exactly through
    GMV and Tangency portfolios.
    """
    fig, ax = plt.subplots()

    # Param-sweep over target returns
    # Compute GMV first
    w_gmv = find_global_min_variance(cov, max_exposure)
    vol_gmv, ret_gmv, _ = compute_portfolio_performance(w_gmv, mu, cov, 0.0)

    r_min = ret_gmv  # or ret_gmv * 0.95, a bit below GMV
    r_max = max(mu) * 1.5
    r_vals = np.linspace(r_min, r_max, resolution)
    frontier_points = []

    def objective(w):
        return np.sqrt(w @ cov @ w)

    for r_target in r_vals:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq","fun": lambda w: max_exposure - np.sum(np.abs(w))},
            {"type": "eq", "fun": lambda w: w @ mu - r_target}
        ]
        w0 = np.ones(len(mu)) / len(mu)
        res = minimize(objective, w0, constraints=cons, method="SLSQP")
        if res.success:
            vol_ = np.sqrt(res.x @ cov @ res.x)
            frontier_points.append((vol_, r_target))

    frontier_points = np.array(frontier_points)
    if frontier_points.size == 0:
        print("No feasible frontier points found under margin constraints.")
        return None, None, None

    # Global Minimum Variance
    w_gmv = find_global_min_variance(cov, max_exposure)
    vol_gmv, ret_gmv, _ = compute_portfolio_performance(w_gmv, mu, cov, 0.0)
    ax.scatter([vol_gmv], [ret_gmv], c="green", marker="o", s=100, label="GMV Portfolio")

    # Tangency Portfolio
    w_tan = find_tangency_portfolio(mu, cov, rf, max_exposure)
    vol_tan, ret_tan, _ = compute_portfolio_performance(w_tan, mu, cov, rf)
    ax.scatter([vol_tan], [ret_tan], c="red", marker="*", s=200, label="Tangency Portfolio")

    # Subset frontier points to [vol_gmv, vol_tan]
    min_vol = min(vol_gmv, vol_tan)
    max_vol = max(vol_gmv, vol_tan)
    subset = frontier_points[
        (frontier_points[:,0] >= min_vol) &
        (frontier_points[:,0] <= max_vol)
    ]

    # Ensure GMV & Tangency are in the subset
    gmvtup = (vol_gmv, ret_gmv)
    tantup = (vol_tan, ret_tan)
    subset_list = subset.tolist()
    if gmvtup not in subset_list:
        subset_list.append(gmvtup)
    if tantup not in subset_list:
        subset_list.append(tantup)
    subset = np.array(subset_list)

    # Sort by volatility
    subset = subset[subset[:,0].argsort()]

    # Interpolate with a spline (linear, quadratic, or cubic)
    x = subset[:,0]  # vol
    y = subset[:,1]  # return
    f = interp1d(x, y, kind=spline_kind)

    # Plot the interpolated curve
    x_fit = np.linspace(x[0], x[-1], 200)
    y_fit = f(x_fit)
    ax.plot(x_fit, y_fit, 'b--', label="Interpolated Frontier")

    # Capital Market Line
    sharpe_tan = (ret_tan - rf)/(vol_tan if vol_tan>1e-9 else 1e-9)
    sigmas = np.linspace(0, vol_tan*1.5, 100)
    cml = rf + sharpe_tan*sigmas
    ax.plot(sigmas, cml, 'k--', linewidth=2, label="Capital Market Line")

    ax.set_xlabel("Volatility (Std Dev)")
    ax.set_ylabel("Return")
    ax.set_title(f"Frontier (Margin={1/max_exposure*100:.0f}%), Spline '{spline_kind}', R_f={rf:.2%}")
    ax.legend()
    plt.show()
    return w_tan, vol_tan, ret_tan


def complete_portfolio(tan_weights, ret_tan, rf_annual, target_return):
    """
    Given the tangency portfolio weights (risky assets), its annual return (ret_tan),
    and the annual risk-free rate (rf_annual), compute the fraction (x) to invest in the tangency portfolio
    such that the overall portfolio achieves the target_return:

      x = (target_return - rf_annual) / (ret_tan - rf_annual)

    The complete portfolio has:
      - Risky (tangency) portion: x * tan_weights
      - T-bills: 1 - x
    """
    if abs(ret_tan - rf_annual) < 1e-9:
        x = 0.0
    else:
        x = (target_return - rf_annual) / (ret_tan - rf_annual)
    return x, 1 - x


def time_series_rebalance(crypto_csv, stock_csv, start_date, end_date, lookback=60, margin=0.25, target_return=None):
    """
    Performs time-series rebalancing under margin constraints.

    Parameters:
      - margin: maintenance margin as a decimal (e.g., 0.25 means 25% margin, allowing total exposure up to 1/0.25 = 4)
      - target_return: desired annualized return for the complete portfolio. If not provided, it defaults to halfway
                       between the risk-free rate and the tangency portfolio's annual return at the last rebal date.

    The function prints out the portfolio performance stats as well as the complete portfolio weights (risky + T-bills).
    """
    # Calculate max exposure allowed from margin
    max_exposure = 1 / margin  # e.g., margin=0.25 -> max_exposure = 4

    # Load returns
    crypto_df = load_asset_returns(crypto_csv)
    stock_df = load_asset_returns(stock_csv)
    asset_df = pd.concat([crypto_df, stock_df], axis=1)
    asset_df = asset_df[(asset_df.index >= start_date) & (asset_df.index <= end_date)]

    # Load T-bill data
    tbill_df = get_tbill_data(start_date, end_date)
    asset_df.index = pd.to_datetime(asset_df.index)
    tbill_df.index = pd.to_datetime(tbill_df.index)
    combined = asset_df.join(tbill_df, how="inner")
    dates = combined.index

    if len(dates) < lookback + 1:
        print("Not enough data for the chosen lookback.")
        return pd.DataFrame(), {}

    portfolio_val = 1.0
    portvals = []
    weights_dict = {}
    portvals.append((dates[lookback], portfolio_val))

    for i in range(lookback, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        window_dates = dates[i - lookback: i]
        window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
        if window_returns.empty:
            continue
        mu_window, cov_window = mean_cov_matrix(window_returns)
        rf_daily = combined.loc[current_date, "Daily_Rf"]
        w_tan = find_tangency_portfolio(mu_window, cov_window, rf_daily, max_exposure)
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
    win_rate = (daily_ret > 0).sum() / (daily_ret != 0).sum()

    print("=== Final Portfolio Stats ===")
    print(f"Final Portfolio Value   : {portvals_df.iloc[-1]['PortfolioValue']:.4f}")
    print(f"Annualized Sharpe Ratio : {ann_sharpe:.4f}")
    print(f"Max Drawdown            : {max_drawdown:.2%}")
    print(f"Win Rate                : {win_rate:.2%}")

    # At the final rebalancing date, compute the frontier and tangency portfolio
    if weights_dict:
        final_rebal_date = list(weights_dict.keys())[-1]
        window_dates = dates[dates <= final_rebal_date][-lookback:]
        window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
        mu_final, cov_final = mean_cov_matrix(window_returns)
        rf_annual = combined.loc[final_rebal_date, "Rate"]
        w_tan_final, vol_tan, ret_tan = plot_efficient_frontier(mu_final, cov_final, rf_annual, max_exposure,
                                                                resolution=1000)
    else:
        print("No rebalancing steps performed.")
        return portvals_df, weights_dict

    # Compute complete portfolio weights
    # If target_return is not provided, default to halfway between risk-free and tangency portfolio returns
    if target_return is None:
        target_return = rf_annual + 0.5 * (ret_tan - rf_annual)
    # Compute fraction x to invest in risky assets (tangency portfolio)
    if abs(ret_tan - rf_annual) < 1e-9:
        x = 0.0
    else:
        x = (target_return - rf_annual) / (ret_tan - rf_annual)
    print(f"\nTarget Overall Annual Return: {target_return:.2%}")
    print(f"Fraction in Risky Assets (x): {x:.4f}")
    print(f"Fraction in T-bills (1-x): {1 - x:.4f}")
    print("\nComplete Portfolio Weights (Risky Assets weighted by x):")
    for asset, w in zip(mu_final.index, w_tan_final):
        print(f"  {asset}: {x * w:.4f}")
    print("T-bills Weight: {:.4f}".format(1 - x))

    return portvals_df, weights_dict


if __name__ == "__main__":
    start_date = datetime.datetime(2025, 1, 1)
    end_date = datetime.datetime(2025, 2, 28)
    portfolio_values, weights = time_series_rebalance(
        "test_crypto.csv",
        "test_stocks.csv",
        start_date,
        end_date,
        lookback=32,
        margin=0.25,
        target_return=0.08
    )