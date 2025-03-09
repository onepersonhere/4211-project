import numpy as np
import scipy.optimize as sco

class PortfolioRebalancer:
    def __init__(self, target_value, stocks, crypto, mode="target_risk"):
        self.target_value = target_value
        self.stocks = stocks
        self.crypto = crypto
        self.mode = mode

    def portfolio_performance(self, weights, returns, cov):
        ret = np.dot(weights, returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return vol, ret

    def rebalance(self):
        assets = self.stocks + self.crypto
        returns = np.array([asset['return'] for asset in assets])
        cov = np.diag([asset['risk']**2 for asset in assets])
        num_assets = len(returns)
        init_guess = np.ones(num_assets) / num_assets
        bounds = tuple((0,1) for _ in range(num_assets))

        if self.mode == "target_risk":
            # Maximize return subject to vol <= target_value
            constraints = [
                {'type': 'ineq', 'fun': lambda w: self.target_value - self.portfolio_performance(w, returns, cov)[0]},
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            def neg_return(w):
                return -self.portfolio_performance(w, returns, cov)[1]
            solution = sco.minimize(neg_return, init_guess, bounds=bounds, constraints=constraints, method='SLSQP')
        elif self.mode == "target_reward":
            # Minimize volatility subject to ret >= target_value
            constraints = [
                {'type': 'ineq', 'fun': lambda w: self.portfolio_performance(w, returns, cov)[1] - self.target_value},
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            def vol_func(w):
                return self.portfolio_performance(w, returns, cov)[0]
            solution = sco.minimize(vol_func, init_guess, bounds=bounds, constraints=constraints, method='SLSQP')
        else:
            # Default: maximize Sharpe ratio
            def negative_sharpe_ratio(weights):
                vol, ret = self.portfolio_performance(weights, returns, cov)
                return -(ret / vol)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            solution = sco.minimize(negative_sharpe_ratio, init_guess, args=(), 
                                    method='SLSQP', bounds=bounds, constraints=constraints)

        return solution.x

    def print_weights(self, weights):
        for asset, w in zip(self.stocks + self.crypto, weights):
            print(f"{asset['name']}: {w:.2f}")

# Example usage
stocks = [{'name': 'AAPL', 'return': 0.1, 'risk': 0.2}, {'name': 'MSFT', 'return': 0.12, 'risk': 0.25}]
crypto = [{'name': 'BTC', 'return': 0.2, 'risk': 0.5}, {'name': 'ETH', 'return': 0.18, 'risk': 0.45}]

# Target risk, maximize reward
rebalancer_risk = PortfolioRebalancer(target_value=0.3, stocks=stocks, crypto=crypto, mode="target_risk")
weights_risk = rebalancer_risk.rebalance()
print("Weights (target risk):")
rebalancer_risk.print_weights(weights_risk)

# Target reward, minimize risk
rebalancer_reward = PortfolioRebalancer(target_value=0.15, stocks=stocks, crypto=crypto, mode="target_reward")
weights_reward = rebalancer_reward.rebalance()
print("Weights (target reward):")
rebalancer_reward.print_weights(weights_reward)