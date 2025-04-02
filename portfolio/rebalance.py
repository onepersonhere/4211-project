import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------
#   Helper Functions
# -----------------------

def get_tbill_data(start, end):
    """
    Download T-bill data (^IRX) from Yahoo and compute daily/annual risk-free rates.
    """
    tbill = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=False)
    tbill.reset_index(inplace=True)
    if isinstance(tbill.columns, pd.MultiIndex):
        tbill.columns = tbill.columns.get_level_values(0)
    tbill.set_index("Date", inplace=True)
    tbill["Rate"] = tbill["Close"] / 100.0   # Annual yield
    tbill["Daily_Rf"] = tbill["Rate"] / 252  # Approx daily risk-free
    return tbill[["Rate", "Daily_Rf"]]


def load_asset_returns(csv_path, freq="2min"):
    """
    Loads a CSV of returns with columns:
      Date, Ticker, Actual_Return
    Pivots by Ticker and resamples to the specified frequency.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    pivoted = df.pivot(index="Date", columns="Ticker", values="Actual_Return")
    # Resample to ensure uniform time spacing
    pivoted = pivoted.resample(freq).last()
    return pivoted


def mean_cov_matrix(returns_df):
    """
    Return the mean vector and covariance matrix of returns.
    """
    mu = returns_df.mean()
    cov = returns_df.cov()
    return mu, cov


def find_global_min_variance(cov, max_exposure):
    """
    Compute Global Minimum Variance (GMV) portfolio (sum of weights=1)
    subject to sum(|w|) <= max_exposure.
    """
    n = cov.shape[0]
    init_w = np.ones(n) / n

    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    bounds = [(-10, 10)] * n  # Loose bounds on individual weights
    res = minimize(portfolio_vol, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def find_tangency_portfolio(mu, cov, rf, max_exposure):
    """
    Compute the tangency portfolio weights, which maximize Sharpe ratio:
        max_{w} (w^T mu - rf) / sqrt(w^T cov w)
    subject to sum(w)=1 and sum(|w|) <= max_exposure.
    Used only for plotting/reference in this example.
    """
    n = len(mu)
    init_w = np.ones(n) / n

    def negative_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / max(vol, 1e-9)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    bounds = [(-10, 10)] * n
    res = minimize(negative_sharpe, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def find_max_return_portfolio(mu, cov, max_exposure):
    """
    Compute portfolio weights that MAXIMIZE expected return:
        max_{w} w^T mu
    subject to sum(w) = 1 and sum(|w|) <= max_exposure.
    """
    n = len(mu)
    init_w = np.ones(n) / n

    def negative_return(w):
        return -np.dot(w, mu)  # Minimizing negative => maximizing w^T mu

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    bounds = [(-10, 10)] * n
    res = minimize(negative_return, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def compute_portfolio_performance(weights, mu, cov, rf=0.0):
    """
    Volatility (std dev), return, Sharpe (optionally subtract rf).
    """
    ret = np.dot(weights, mu)
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / max(vol, 1e-9)
    return vol, ret, sharpe


def plot_efficient_frontier(mu, cov, rf, max_exposure, resolution=2000, spline_kind='cubic'):
    """
    Plot the Markowitz efficient frontier for the risky assets (crypto + stocks),
    plus GMV (green), tangency portfolio (red), risk-free (gold point at 0 volatility),
    and Max Return portfolio (blue triangle).
    Also plots the capital market line (CML).
    """
    fig, ax = plt.subplots()

    # Risk-free point at (0, rf)
    ax.scatter([0], [rf], c="gold", marker="o", s=100, label="Risk-Free")

    # Global Min Var
    w_gmv = find_global_min_variance(cov, max_exposure)
    vol_gmv, ret_gmv, _ = compute_portfolio_performance(w_gmv, mu, cov, 0.0)
    ax.scatter([vol_gmv], [ret_gmv], c="green", marker="o", s=100, label="GMV Portfolio")

    # Tangency
    w_tan = find_tangency_portfolio(mu, cov, rf, max_exposure)
    vol_tan, ret_tan, _ = compute_portfolio_performance(w_tan, mu, cov, rf)
    ax.scatter([vol_tan], [ret_tan], c="red", marker="*", s=200, label="Tangency Portfolio")

    # Max Return
    w_max = find_max_return_portfolio(mu, cov, max_exposure)
    vol_max, ret_max, _ = compute_portfolio_performance(w_max, mu, cov, 0.0)
    ax.scatter([vol_max], [ret_max], marker="^", s=100, label="Max Return Portfolio")

    # Compute a series of target returns to trace out the frontier
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

    # Interpolate the frontier
    frontier_points = np.array(frontier_points)
    if frontier_points.size == 0:
        print("No feasible frontier points found under margin constraints.")
        return None, None, None

    df_front = pd.DataFrame(frontier_points, columns=["Vol", "Ret"])
    df_front.sort_values("Vol", inplace=True)
    df_front.drop_duplicates(subset=["Vol"], inplace=True)

    x = df_front["Vol"].values
    y = df_front["Ret"].values
    f = interp1d(x, y, kind=spline_kind)
    x_fit = np.linspace(x[0], x[-1], 300)
    y_fit = f(x_fit)
    ax.plot(x_fit, y_fit, 'b--', label="Interpolated Frontier")

    # Capital Market Line (CML) from 0 to whichever is bigger: vol_tan or vol_max
    x_max = max(vol_tan, vol_max) * 1.5
    sigmas = np.linspace(0, x_max, 100)
    sharpe_tan = (ret_tan - rf) / max(vol_tan, 1e-9)
    cml = rf + sharpe_tan * sigmas
    ax.plot(sigmas, cml, 'k--', linewidth=2, label="Capital Market Line")

    ax.set_xlabel("Volatility (Std Dev)")
    ax.set_ylabel("Return")
    ax.set_title(f"Efficient Frontier (Ignoring T-bills in Portfolio Construction)\n"
                 f"R_f={rf:.2%}, MaxExposure={max_exposure}")
    ax.legend()
    plt.show()

    return w_max, vol_max, ret_max


# -------------------------------
#   Iteration for Rebalancing
# -------------------------------

def process_iteration(i, dates, combined, asset_df, lookback, max_exposure):
    """
    Single rebalancing step:
      - Compute the MAX RETURN portfolio for the last 'lookback' window,
        ignoring T-bills as an investable asset.
      - Return realized return for the next period.
    """
    current_date = dates[i]
    next_date = dates[i + 1]
    window_dates = dates[i - lookback: i]

    # Only take the columns of actual assets (crypto, stocks)
    window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
    if window_returns.empty:
        return (i, current_date, next_date, None, None)

    # Compute raw mean/cov from 2-minute returns
    mu_raw, cov_raw = mean_cov_matrix(window_returns)

    # Annualize (approx): 195 intervals/day * 252 trading days
    intervals_per_day = 195
    annualization_factor = intervals_per_day * 252
    mu_ann = mu_raw * annualization_factor
    cov_ann = cov_raw * annualization_factor

    # Get the max-return portfolio
    w_max = find_max_return_portfolio(mu_ann, cov_ann, max_exposure)

    # Next period’s actual 2-min return (not annualized)
    r_next = combined.loc[next_date, window_returns.columns].fillna(0.0)
    realized_ret = np.dot(w_max, r_next)
    return (i, current_date, next_date, w_max, realized_ret)


def time_series_rebalance(crypto_csv, stock_csv, start_date, end_date,
                          lookback=60, margin=0.25):
    """
    Rebalances over time to hold the MAX-RETURN portfolio among the chosen assets
    (crypto + stocks), ignoring T-bills as an investable asset.

    T-bill data is still loaded only to compute rf for plotting/analysis.
    """
    max_exposure = 1 / margin

    # 1) Load asset returns
    crypto_df = load_asset_returns(crypto_csv, freq="2min")
    crypto_df = crypto_df.loc[(crypto_df.index >= start_date) & (crypto_df.index <= end_date)]

    stock_df = load_asset_returns(stock_csv, freq="2min")
    stock_df = stock_df.loc[(stock_df.index >= start_date) & (stock_df.index <= end_date)]

    asset_df = crypto_df.join(stock_df, how="inner")
    asset_df.dropna(how="any", inplace=True)
    if asset_df.empty:
        print("No overlapping data.")
        return pd.DataFrame(), {}

    # 2) T-bill data (for rf reference)
    tbill_df = get_tbill_data(start_date, end_date)
    tbill_df = tbill_df.reindex(asset_df.index, method="ffill")

    # Combine
    combined = asset_df.join(tbill_df, how="inner")
    dates = combined.index
    if len(dates) < lookback + 1:
        print("Not enough data for the chosen lookback.")
        return pd.DataFrame(), {}

    # 3) Parallel rebalancing
    results = []
    iterations = range(lookback, len(dates) - 1)
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_iteration, i, dates, combined, asset_df, lookback, max_exposure
            ): i
            for i in iterations
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing iterations", unit="iteration"):
            result = future.result()
            results.append(result)

    # Filter valid
    results = [r for r in results if r[3] is not None]
    results.sort(key=lambda x: x[0])

    # 4) Compute portfolio value over time
    portfolio_val = 1.0
    portvals = []
    weights_dict = {}

    # Start tracking from date[lookback]
    portvals.append((dates[lookback], portfolio_val))

    for i, current_date, next_date, w_max, realized_ret in results:
        weights_dict[current_date] = w_max
        portfolio_val *= (1 + realized_ret)
        portvals.append((next_date, portfolio_val))

    # Create a DataFrame with daily returns
    portvals_df = pd.DataFrame(portvals, columns=["Date", "PortfolioValue"]).set_index("Date")
    portvals_df["DailyRet"] = portvals_df["PortfolioValue"].pct_change().fillna(0.0)

    # Final performance stats
    daily_ret = portvals_df["DailyRet"]
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std(ddof=1)
    ann_factor = 252.0  # daily -> annual
    ann_sharpe = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 1e-9 else 0.0

    running_max = portvals_df["PortfolioValue"].cummax()
    max_drawdown = (portvals_df["PortfolioValue"] / running_max - 1).min()
    win_rate = (daily_ret > 0).sum() / max((daily_ret != 0).sum(), 1)

    # Cumulative and annualized realized returns
    final_value = portvals_df.iloc[-1]["PortfolioValue"]
    cumulative_return = final_value - 1.0
    annualized_return = (1 + mean_ret) ** ann_factor - 1.0

    print("\n=== Final Portfolio Stats (Max Return Strategy) ===")
    print(f"Final Portfolio Value   : {final_value:.4f}")
    print(f"Cumulative Return       : {cumulative_return:.2%}")
    print(f"Annualized Return       : {annualized_return:.2%}")
    print(f"Annualized Sharpe Ratio : {ann_sharpe:.4f}")
    print(f"Max Drawdown            : {max_drawdown:.2%}")
    print(f"Win Rate                : {win_rate:.2%}")

    # 5) Plot the efficient frontier at the final rebalancing point
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

            # For the plot only:
            rf_annual = combined.loc[final_rebal_date, "Rate"]

            # Draw the frontier, tangency, etc.
            w_max_final, vol_max, ret_max = plot_efficient_frontier(
                mu_final, cov_final, rf_annual, max_exposure, resolution=1000
            )

            # Show final max-return portfolio weights
            print("\nFinal Max-Return Portfolio Weights (Last Rebalance):")
            w_final = weights_dict[final_rebal_date]
            for asset, w in zip(window_returns.columns, w_final):
                print(f"  {asset}: {w:.4f}")
        else:
            print("Not enough data to plot the efficient frontier at final rebalancing.")
    else:
        print("No rebalancing steps were performed.")

    return portvals_df, weights_dict


def periodic_rebalance(crypto_csv, stock_csv, rebalance_days=14,
                       lookback=60, margin=0.25):
    """
    Splits the overall date range into segments of length `rebalance_days`.
    For each segment, runs the time_series_rebalance approach above.
    """
    # Load returns to figure out overall date range
    crypto_df = load_asset_returns(crypto_csv, freq="2min")
    stock_df = load_asset_returns(stock_csv, freq="2min")
    asset_df = crypto_df.join(stock_df, how="inner")
    asset_df.dropna(how="all", inplace=True)
    if asset_df.empty:
        print("No overlapping data.")
        return {}

    overall_start = asset_df.index.min()
    overall_end = asset_df.index.max()

    # Endpoints for the periods (rebalance every N days)
    period_endpoints = asset_df.resample(f'{rebalance_days}D').last().index

    results = {}
    segment_start = overall_start
    for endpoint in period_endpoints:
        if endpoint <= segment_start:
            continue

        print(f"\n--- Rebalance Period: {segment_start} to {endpoint} ---")
        portvals_df, weights_dict = time_series_rebalance(
            crypto_csv, stock_csv,
            start_date=segment_start,
            end_date=endpoint,
            lookback=lookback,
            margin=margin
        )
        results[endpoint] = (portvals_df, weights_dict)
        segment_start = endpoint

    return results


# -------------------------------------
#   Main: Example Usage
# -------------------------------------
if __name__ == "__main__":
    crypto_csv_path = "../crypto/returns.csv"
    stock_csv_path  = "../stock/returns.csv"

    # Run with ~14-day rebalancing
    all_results = periodic_rebalance(
        crypto_csv_path,
        stock_csv_path,
        rebalance_days=7,
        lookback=64,
        margin=0.25
    )

    # Print final portfolio values for each segment
    for period_end, (portvals_df, weights_dict) in all_results.items():
        if not portvals_df.empty:
            print(f"\n### Period ending on {period_end.date()} ###")
            print(portvals_df.tail(1))

# rebalancing -> get max return portfolio (regardless of tangent)
# calculate sharpe ratio, draw down etc.
# duplicate the df from stock's returns and crypto's returns (add the weight in) for each period
# remove the tbills