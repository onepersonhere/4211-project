import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- Helper functions ---

def get_tbill_data(start, end):
    """
    Download 13-week T-bill data (^IRX) from Yahoo Finance,
    compute the annual and daily risk-free rate, and return
    a DataFrame with columns ["Rate", "Daily_Rf"].
    """
    tbill = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=False)
    tbill.reset_index(inplace=True)
    if isinstance(tbill.columns, pd.MultiIndex):
        tbill.columns = tbill.columns.get_level_values(0)
    tbill.set_index("Date", inplace=True)
    tbill["Rate"] = tbill["Close"] / 100.0  # Annual T-bill yield
    tbill["Daily_Rf"] = tbill["Rate"] / 252  # Daily risk-free rate
    return tbill[["Rate", "Daily_Rf"]]


def load_asset_returns(csv_path, freq="2min"):
    """
    Load a CSV with columns ["Date","Ticker","Actual_Return"] and pivot it so that
    each Ticker is a column. Then resample to the specified frequency.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    pivoted = df.pivot(index="Date", columns="Ticker", values="Actual_Return")
    # Resample to a chosen frequency (e.g., "2min")
    pivoted = pivoted.resample(freq).last()
    return pivoted


def mean_cov_matrix(returns_df):
    """
    Compute the mean vector and covariance matrix from the provided returns DataFrame.
    """
    mu = returns_df.mean()
    cov = returns_df.cov()
    return mu, cov


# --- Portfolio optimization routines ---

def find_global_min_variance(cov, max_exposure):
    """
    Solve for the Global Minimum Variance (GMV) portfolio:
        minimize w' * COV * w
        s.t. sum(w) = 1
             sum(abs(w)) <= max_exposure
    """
    n = cov.shape[0]
    init_w = np.ones(n) / n

    def portfolio_vol(w):
        return np.sqrt(w @ cov @ w)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    bounds = [(-10, 10)] * n  # or any wide bound
    res = minimize(portfolio_vol, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def find_tangency_portfolio(mu, cov, rf, max_exposure):
    """
    Solve for the Tangency (maximum Sharpe) portfolio:
        maximize (w' * mu - rf) / sqrt(w' * COV * w)
        s.t. sum(w) = 1
             sum(abs(w)) <= max_exposure
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


def find_min_variance_for_return(mu, cov, target_return, max_exposure):
    """
    Solve for portfolio with minimum variance subject to a target return:
        minimize w' * COV * w
        s.t. sum(w) = 1
             w' * mu >= target_return
             sum(abs(w)) <= max_exposure
    """
    n = len(mu)
    init_w = np.ones(n) / n

    def portfolio_var(w):
        return w @ cov @ w

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: np.dot(w, mu) - target_return},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    bounds = [(-10, 10)] * n
    res = minimize(portfolio_var, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def find_max_return_for_variance(mu, cov, target_variance, max_exposure):
    """
    Solve for portfolio with maximum return subject to a target variance:
        maximize w' * mu
        s.t. sum(w) = 1
             w' * COV * w <= target_variance
             sum(abs(w)) <= max_exposure
    """
    n = len(mu)
    init_w = np.ones(n) / n

    def negative_return(w):
        return -np.dot(w, mu)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: target_variance - (w @ cov @ w)},
        {"type": "ineq", "fun": lambda w: max_exposure - np.sum(np.abs(w))}
    ]
    bounds = [(-10, 10)] * n
    res = minimize(negative_return, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def compute_portfolio_performance(weights, mu, cov, rf=0.0):
    """
    Compute portfolio volatility, return, and Sharpe ratio.
    """
    ret = np.dot(weights, mu)
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / max(vol, 1e-9)
    return vol, ret, sharpe


def plot_efficient_frontier(mu, cov, rf, max_exposure, resolution=2000, spline_kind='cubic'):
    """
    Plots an approximate efficient frontier (by scanning over possible returns),
    the GMV portfolio, and the tangency portfolio. Also includes the Capital Market Line.
    """
    fig, ax = plt.subplots()

    # Compute GMV portfolio
    w_gmv = find_global_min_variance(cov, max_exposure)
    vol_gmv, ret_gmv, _ = compute_portfolio_performance(w_gmv, mu, cov, 0.0)

    # We'll scan from GMV return up to ~150% of the max asset return
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
            {"type": "eq", "fun": lambda w: np.dot(w, mu) - r_target}
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

    ax.scatter([vol_gmv], [ret_gmv], marker="o", s=100, label="GMV Portfolio")

    # Tangency portfolio
    w_tan = find_tangency_portfolio(mu, cov, rf, max_exposure)
    vol_tan, ret_tan, _ = compute_portfolio_performance(w_tan, mu, cov, rf)
    ax.scatter([vol_tan], [ret_tan], marker="*", s=200, label="Tangency Portfolio")

    # Interpolate the frontier
    frontier_points = frontier_points[frontier_points[:, 0].argsort()]
    x = frontier_points[:, 0]
    y = frontier_points[:, 1]
    # Remove duplicates in x
    df_front = pd.DataFrame(frontier_points, columns=["Vol", "Ret"])
    df_front.drop_duplicates(subset=["Vol"], inplace=True)
    df_front.sort_values("Vol", inplace=True)
    x = df_front["Vol"].values
    y = df_front["Ret"].values
    f = interp1d(x, y, kind=spline_kind)
    x_fit = np.linspace(x[0], x[-1], 300)
    y_fit = f(x_fit)
    ax.plot(x_fit, y_fit, '--', label="Frontier")

    # Capital Market Line
    sharpe_tan = (ret_tan - rf) / max(vol_tan, 1e-9)
    sigmas = np.linspace(0, vol_tan * 1.5, 200)
    cml = rf + sharpe_tan * sigmas
    ax.plot(sigmas, cml, 'k--', linewidth=2, label="CML")

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    plt.show()

    return w_tan, vol_tan, ret_tan


# --- Parallel iteration processing ---

def process_iteration(i, dates, combined, asset_df, lookback, max_exposure,
                      method, use_tbills, target_return, target_variance):
    """
    Process a single rebalancing iteration given a method:
      - 'gmv' (Global Min Variance)
      - 'tangency' (Max Sharpe)
      - 'min_variance_for_return'
      - 'max_return_for_variance'

    Returns a tuple:
      (i, current_date, next_date, weights, realized_return)
    """
    current_date = dates[i]
    next_date = dates[i + 1]
    window_dates = dates[i - lookback: i]

    # Extract the returns of all assets in the lookback window
    window_returns = combined.loc[window_dates, asset_df.columns].dropna(axis=1, how="any")
    if window_returns.empty:
        return (i, current_date, next_date, None, None)

    # Compute raw mean and covariance from 2-minute returns
    mu_raw, cov_raw = mean_cov_matrix(window_returns)

    # Annualize: assume 195 two-minute intervals per day, 252 trading days per year
    intervals_per_day = 195
    annualization_factor = intervals_per_day * 252
    mu_ann = mu_raw * annualization_factor
    cov_ann = cov_raw * annualization_factor

    # If we are ignoring T-bills, set rf to 0; otherwise get from data
    if use_tbills and "Rate" in combined.columns:
        rf_annual = combined.loc[current_date, "Rate"]
    else:
        rf_annual = 0.0

    # Select the optimization method
    if method == "gmv":
        w = find_global_min_variance(cov_ann, max_exposure)
    elif method == "tangency":
        w = find_tangency_portfolio(mu_ann, cov_ann, rf_annual, max_exposure)
    elif method == "min_variance_for_return":
        if target_return is None:
            raise ValueError("You must provide target_return for 'min_variance_for_return' method.")
        w = find_min_variance_for_return(mu_ann, cov_ann, target_return, max_exposure)
    elif method == "max_return_for_variance":
        if target_variance is None:
            raise ValueError("You must provide target_variance for 'max_return_for_variance' method.")
        w = find_max_return_for_variance(mu_ann, cov_ann, target_variance, max_exposure)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get next period's actual 2-minute returns
    window_cols = window_returns.columns
    r_next = combined.loc[next_date, window_cols].fillna(0.0)
    realized_ret = np.dot(w, r_next)

    return (i, current_date, next_date, w, realized_ret)


# --- Modified time_series_rebalance ---

def time_series_rebalance(
    crypto_csv,
    stock_csv,
    start_date,
    end_date,
    lookback=60,
    margin=0.25,
    method="tangency",
    use_tbills=True,
    target_return=None,
    target_variance=None
):
    """
    Perform rolling-window rebalancing on a set of 2-minute returns data from
    crypto_csv and stock_csv, between start_date and end_date, with a lookback
    of 'lookback' intervals. The 'method' can be:
        - 'gmv': Global Min Variance
        - 'tangency': Tangency (Max Sharpe)
        - 'min_variance_for_return': Min Variance for a given target return
        - 'max_return_for_variance': Max Return for a given target variance

    If 'use_tbills' is True, we fetch ^IRX data and use it as a risk-free rate (for tangency).
    Otherwise, risk-free = 0.

    The function returns:
      portvals_df: A DataFrame of the portfolio value over time
      weights_dict: A dictionary of weights at each rebalancing date
    """
    max_exposure = 1 / margin

    # 1) Load and align asset returns
    crypto_df = load_asset_returns(crypto_csv, freq="2min")
    crypto_df = crypto_df.loc[(crypto_df.index >= start_date) & (crypto_df.index <= end_date)]
    stock_df = load_asset_returns(stock_csv, freq="2min")
    stock_df = stock_df.loc[(stock_df.index >= start_date) & (stock_df.index <= end_date)]

    asset_df = crypto_df.join(stock_df, how="inner")
    asset_df.dropna(how="any", inplace=True)
    if asset_df.empty:
        print("No overlapping data.")
        return pd.DataFrame(), {}

    # 2) Possibly load T-bill data for risk-free
    if use_tbills:
        tbill_df = get_tbill_data(start_date, end_date)
        tbill_df = tbill_df.reindex(asset_df.index, method="ffill")
        combined = asset_df.join(tbill_df, how="inner")
    else:
        combined = asset_df.copy()
        # Make sure "Rate" isn't in the dataset
        if "Rate" not in combined.columns:
            combined["Rate"] = 0.0  # Force to 0 if we want a column
    dates = combined.index
    if len(dates) < lookback + 1:
        print("Not enough data for the chosen lookback.")
        return pd.DataFrame(), {}

    # 3) Run the parallel rebalancing across each iteration
    results = []
    iterations = range(lookback, len(dates) - 1)

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_iteration,
                i, dates, combined, asset_df, lookback, max_exposure,
                method, use_tbills, target_return, target_variance
            ): i
            for i in iterations
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing iterations", unit="iter"):
            result = future.result()
            results.append(result)

    # Filter out bad results
    results = [r for r in results if r[3] is not None]
    results.sort(key=lambda x: x[0])

    # 4) Build the portfolio value series
    portfolio_val = 1.0
    portvals = []
    weights_dict = {}

    # The first rebalancing date
    portvals.append((dates[lookback], portfolio_val))
    for i, current_date, next_date, w, realized_ret in results:
        weights_dict[current_date] = w
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

    print("\n=== Final Portfolio Stats ===")
    print(f"Method                : {method}")
    print(f"Final Portfolio Value : {portvals_df.iloc[-1]['PortfolioValue']:.4f}")
    print(f"Annualized Sharpe     : {ann_sharpe:.4f}")
    print(f"Max Drawdown          : {max_drawdown:.2%}")
    print(f"Win Rate              : {win_rate:.2%}")

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

            if use_tbills:
                rf_final = combined.loc[final_rebal_date, "Rate"]
            else:
                rf_final = 0.0

            plot_efficient_frontier(
                mu_final, cov_final, rf_final,
                max_exposure, resolution=500
            )
        else:
            print("Not enough data for efficient frontier plot at the final rebalancing window.")
    else:
        print("No rebalancing steps performed.")

    return portvals_df, weights_dict


# --- General periodic rebalancing function ---

def periodic_rebalance(
    crypto_csv,
    stock_csv,
    rebalance_days=1,
    lookback=60,
    margin=0.25,
    method="tangency",
    use_tbills=True,
    target_return=None,
    target_variance=None
):
    """
    Splits the overall data into segments of 'rebalance_days' and runs
    time_series_rebalance on each segment. Returns results keyed by period end date.
    """
    # Load all data once to get overall date range
    crypto_df = load_asset_returns(crypto_csv, freq="2min")
    stock_df = load_asset_returns(stock_csv, freq="2min")
    asset_df = crypto_df.join(stock_df, how="inner")
    asset_df.dropna(how="all", inplace=True)

    if asset_df.empty:
        print("No overlapping data.")
        return {}

    overall_start = asset_df.index.min()
    overall_end = asset_df.index.max()

    # Create period endpoints by resampling
    period_endpoints = asset_df.resample(f'{rebalance_days}D').last().index

    results = {}
    segment_start = overall_start
    for endpoint in period_endpoints:
        if endpoint <= segment_start:
            continue

        print(f"\nRebalancing from {segment_start} to {endpoint} | Method: {method}")
        portvals_df, weights_dict = time_series_rebalance(
            crypto_csv,
            stock_csv,
            start_date=segment_start,
            end_date=endpoint,
            lookback=lookback,
            margin=margin,
            method=method,
            use_tbills=use_tbills,
            target_return=target_return,
            target_variance=target_variance
        )
        results[endpoint] = (portvals_df, weights_dict)
        segment_start = endpoint

    return results


# --- Example usage (if run directly) ---
if __name__ == "__main__":
    crypto_csv_path = "../crypto/returns.csv"
    stock_csv_path = "../stock/returns.csv"

    # Example: Rebalance every 14 days, look back 64 intervals, use 25% margin
    # Here we show how to do a tangency portfolio ignoring T-bills
    daily_results = periodic_rebalance(
        crypto_csv_path,
        stock_csv_path,
        rebalance_days=7,     # every 14 days
        lookback=64,
        margin=0.25,
        method="max_return_for_variance",     # choose from "gmv", "tangency", "min_variance_for_return", "max_return_for_variance"
        use_tbills=False,      # ignore T-bill data => rf=0 for tangency
        target_return=None,    # only needed for "min_variance_for_return"
        target_variance=0.3   # only needed for "max_return_for_variance"
    )

    # Print the final portfolio value for each period
    for period_end, (portvals_df, weights_dict) in daily_results.items():
        print(f"\n--- Results for period ending on {period_end.date()} ---")
        print(portvals_df.tail(1))


# rebalancing -> get max return portfolio (regardless of tangent)
# calculate sharpe ratio, draw down etc.
# duplicate the df from stock's returns and crypto's returns (add the weight in) for each period
# remove the tbills