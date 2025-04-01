import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- Existing functions ---

def get_tbill_data(start, end):
    tbill = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=False)
    tbill.reset_index(inplace=True)
    if isinstance(tbill.columns, pd.MultiIndex):
        tbill.columns = tbill.columns.get_level_values(0)
    tbill.set_index("Date", inplace=True)
    tbill["Rate"] = tbill["Close"] / 100.0  # Annual T-bill yield
    tbill["Daily_Rf"] = tbill["Rate"] / 252  # Daily risk-free rate
    return tbill[["Rate", "Daily_Rf"]]


def load_asset_returns(csv_path, freq="2min"):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    pivoted = df.pivot(index="Date", columns="Ticker", values="Actual_Return")
    pivoted = pivoted.resample(freq).last()
    return pivoted


def mean_cov_matrix(returns_df):
    mu = returns_df.mean()
    cov = returns_df.cov()
    return mu, cov


def find_tangency_portfolio(mu, cov, rf, max_exposure):
    n = len(mu)
    init_w = np.ones(n) / n

    def negative_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / (vol if vol > 1e-9 else 1e-9)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    bounds = [(-10, 10)] * n
    res = minimize(negative_sharpe, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def find_global_min_variance(cov, max_exposure):
    n = cov.shape[0]
    init_w = np.ones(n) / n

    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    bounds = [(-10, 10)] * n
    res = minimize(portfolio_vol, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def compute_portfolio_performance(weights, mu, cov, rf=0.0):
    ret = np.dot(weights, mu)
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / (vol if vol > 1e-9 else 1e-9)
    return vol, ret, sharpe


def plot_efficient_frontier(mu, cov, rf, max_exposure, resolution=1000, spline_kind='cubic'):
    fig, ax = plt.subplots()

    # Compute GMV portfolio
    w_gmv = find_global_min_variance(cov, max_exposure)
    vol_gmv, ret_gmv, _ = compute_portfolio_performance(w_gmv, mu, cov, 0.0)
    r_min = ret_gmv
    r_max = max(mu) * 1.5
    r_vals = np.linspace(r_min, r_max, resolution)
    frontier_points = []

    def objective(w):
        return np.sqrt(w @ cov @ w)

    for r_target in r_vals:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))},
            {"type": "eq", "fun": lambda w: w @ mu - r_target}
        ]
        w0 = np.ones(len(mu)) / len(mu)
        bounds = [(-10, 10)] * len(mu)
        res = minimize(objective, w0, constraints=cons, bounds=bounds, method="SLSQP")
        if res.success:
            vol_ = np.sqrt(res.x @ cov @ res.x)
            frontier_points.append((vol_, r_target))
    frontier_points = np.array(frontier_points)
    if frontier_points.size == 0:
        print("No feasible frontier points found under margin constraints.")
        return None, None, None

    ax.scatter([vol_gmv], [ret_gmv], c="green", marker="o", s=100, label="GMV Portfolio")
    w_tan = find_tangency_portfolio(mu, cov, rf, max_exposure)
    vol_tan, ret_tan, _ = compute_portfolio_performance(w_tan, mu, cov, rf)
    ax.scatter([vol_tan], [ret_tan], c="red", marker="*", s=200, label="Tangency Portfolio")

    min_vol = min(vol_gmv, vol_tan)
    max_vol = max(vol_gmv, vol_tan)
    subset = frontier_points[(frontier_points[:, 0] >= min_vol) & (frontier_points[:, 0] <= max_vol)]
    gmvtup = (vol_gmv, ret_gmv)
    tantup = (vol_tan, ret_tan)
    sublist = subset.tolist()
    if gmvtup not in sublist:
        sublist.append(gmvtup)
    if tantup not in sublist:
        sublist.append(tantup)
    subset = np.array(sublist)
    subset = subset[subset[:, 0].argsort()]

    # Remove duplicates on volatility
    df_subset = pd.DataFrame(subset, columns=["Vol", "Ret"])
    df_subset.drop_duplicates(subset=["Vol"], inplace=True)
    subset = df_subset.to_numpy()

    x = subset[:, 0]
    y = subset[:, 1]
    f = interp1d(x, y, kind=spline_kind)
    x_fit = np.linspace(x[0], x[-1], 200)
    y_fit = f(x_fit)
    ax.plot(x_fit, y_fit, 'b--', label="Interpolated Frontier")

    sharpe_tan = (ret_tan - rf) / (vol_tan if vol_tan > 1e-9 else 1e-9)
    sigmas = np.linspace(0, vol_tan * 1.5, 100)
    cml = rf + sharpe_tan * sigmas
    ax.plot(sigmas, cml, 'k--', linewidth=2, label="Capital Market Line")

    ax.set_xlabel("Volatility (Std Dev)")
    ax.set_ylabel("Return")
    ax.set_title(f"Frontier, Spline={spline_kind}, R_f={rf:.2%}, MaxExposure={max_exposure}")
    ax.legend()
    plt.show()
    return w_tan, vol_tan, ret_tan


# --- New helper for parallel processing with annualization fix ---

def process_iteration(i, dates, combined, asset_df, lookback, max_exposure):
    """
    Process a single rebalancing iteration:
      - Computes the tangency portfolio based on the rolling window.
      - Calculates the realized return for the next period.
    Returns a tuple:
      (iteration index, current_date, next_date, w_tan, realized_return)
    If the rolling window yields no valid data, returns None for weights and return.
    """
    current_date = dates[i]
    next_date = dates[i + 1]
    window_dates = dates[i - lookback: i]
    window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
    if window_returns.empty:
        return (i, current_date, next_date, None, None)

    # Compute raw mean and covariance from 2-minute returns
    mu_raw, cov_raw = mean_cov_matrix(window_returns)

    # Annualize: assume 195 two-minute intervals per day and 252 trading days per year
    intervals_per_day = 195
    annualization_factor = intervals_per_day * 252
    mu_ann = mu_raw * annualization_factor
    cov_ann = cov_raw * annualization_factor

    # Use the annual T-bill rate (not the daily rate)
    rf_annual = combined.loc[current_date, "Rate"]

    # Find tangency portfolio using annualized parameters
    w_tan = find_tangency_portfolio(mu_ann, cov_ann, rf_annual, max_exposure)

    # Get next period's actual 2-minute returns (not annualized)
    window_cols = window_returns.columns
    r_next = combined.loc[next_date, window_cols].fillna(0.0)
    realized_ret = np.dot(w_tan, r_next)
    return (i, current_date, next_date, w_tan, realized_ret)


# --- Modified time_series_rebalance with parallel execution and annualization fix ---

def time_series_rebalance(crypto_csv, stock_csv, start_date, end_date,
                          lookback=60, margin=0.25, target_return=None):
    max_exposure = 1 / margin

    # Load and resample asset returns
    crypto_df = load_asset_returns(crypto_csv, freq="2min")
    stock_df = load_asset_returns(stock_csv, freq="2min")
    asset_df = crypto_df.join(stock_df, how="inner")
    asset_df = asset_df.loc[(asset_df.index >= start_date) & (asset_df.index <= end_date)]
    asset_df.dropna(how="all", inplace=True)
    if asset_df.empty:
        print("No overlapping data.")
        return pd.DataFrame(), {}

    # Load T-bill data and align to 2-minute timestamps
    tbill_df = get_tbill_data(start_date, end_date)
    tbill_df = tbill_df.reindex(asset_df.index, method="ffill")
    combined = asset_df.join(tbill_df, how="inner")
    dates = combined.index
    if len(dates) < lookback + 1:
        print("Not enough data for the chosen lookback.")
        return pd.DataFrame(), {}

    # --- Parallel processing of each iteration ---
    results = []
    iterations = range(lookback, len(dates) - 1)
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_iteration, i, dates, combined, asset_df, lookback, max_exposure): i
            for i in iterations
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing iterations", unit="iteration"):
            result = future.result()
            results.append(result)

    # Filter out iterations with invalid (None) results and sort by iteration index
    results = [r for r in results if r[3] is not None]
    results.sort(key=lambda x: x[0])

    # Sequentially update portfolio value using the realized returns
    portfolio_val = 1.0
    portvals = []
    weights_dict = {}
    portvals.append((dates[lookback], portfolio_val))
    for i, current_date, next_date, w_tan, realized_ret in results:
        weights_dict[current_date] = w_tan
        portfolio_val *= (1 + realized_ret)
        portvals.append((next_date, portfolio_val))

    portvals_df = pd.DataFrame(portvals, columns=["Date", "PortfolioValue"]).set_index("Date")
    portvals_df["DailyRet"] = portvals_df["PortfolioValue"].pct_change().fillna(0.0)
    daily_ret = portvals_df["DailyRet"]
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std(ddof=1)
    ann_sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-9 else 0.0
    running_max = portvals_df["PortfolioValue"].cummax()
    max_drawdown = (portvals_df["PortfolioValue"] / running_max - 1).min()
    win_rate = (daily_ret > 0).sum() / (daily_ret != 0).sum()

    print("=== Final Portfolio Stats ===")
    print(f"Final Portfolio Value   : {portvals_df.iloc[-1]['PortfolioValue']:.4f}")
    print(f"Annualized Sharpe Ratio : {ann_sharpe:.4f}")
    print(f"Max Drawdown            : {max_drawdown:.2%}")
    print(f"Win Rate                : {win_rate:.2%}")

    # Plot the efficient frontier at the final rebalancing point using annualized returns
    if weights_dict:
        final_rebal_date = list(weights_dict.keys())[-1]
        window_dates = dates[dates <= final_rebal_date][-lookback:]
        window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
        if not window_returns.empty:
            mu_final_raw, cov_final_raw = mean_cov_matrix(window_returns)
            intervals_per_day = 195
            annualization_factor = intervals_per_day * 252
            mu_final = mu_final_raw * annualization_factor
            cov_final = cov_final_raw * annualization_factor
            rf_annual = combined.loc[final_rebal_date, "Rate"]
            w_tan_final, vol_tan, ret_tan = plot_efficient_frontier(
                mu_final, cov_final, rf_annual, max_exposure, resolution=1000
            )
        else:
            print("Not enough data for efficient frontier plot at final rebalancing.")
    else:
        print("No rebalancing steps performed.")
        return portvals_df, weights_dict

    if target_return is None:
        target_return = rf_annual + 0.5 * (ret_tan - rf_annual)
    x = 0.0 if abs(ret_tan - rf_annual) < 1e-9 else (target_return - rf_annual) / (ret_tan - rf_annual)

    print(f"\nTarget Overall Annual Return: {target_return:.2%}")
    print(f"Fraction in Risky Assets (x): {x:.4f}")
    print(f"Fraction in T-bills (1-x): {1 - x:.4f}")
    print("\nComplete Portfolio Weights (Risky Assets weighted by x):")
    for asset, w in zip(mu_final.index, w_tan_final):
        print(f"  {asset}: {x * w:.4f}")
    print("T-bills Weight: {:.4f}".format(1 - x))

    return portvals_df, weights_dict


if __name__ == "__main__":
    start_date = datetime.datetime(2025, 3, 1)
    end_date = datetime.datetime(2025, 3, 24)
    portfolio_values, weights = time_series_rebalance(
        "../crypto/copy/returns.csv",
        "../stock/returns.csv",
        start_date,
        end_date,
        lookback=64,
        margin=0.25,
        target_return=0.08
    )