import pandas as pd
import numpy as np

def run_monte_carlo(expected_returns, cov_matrix, num_portfolios=50000):
    """
    Runs a Monte Carlo simulation to find the Efficient Frontier.
    
    Returns:
        tuple: A tuple containing the results DataFrame and weights record.
    """
    print("Running Monte Carlo simulation...")
    num_assets = len(expected_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = portfolio_return / portfolio_std_dev
        
    results_frame = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'SharpeRatio'])
    print("Monte Carlo simulation completed.")
    return results_frame, weights_record

def backtest_strategy(returns_backtest, strategy_weights, benchmark_weights):
    """
    Simulates and compares the performance of a strategy against a benchmark.
    """
    print("Starting backtesting simulation...")
    
    # Simulate benchmark portfolio
    benchmark_returns = (returns_backtest * benchmark_weights).sum(axis=1)
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    
    # Simulate strategy portfolio
    strategy_returns = (returns_backtest * strategy_weights).sum(axis=1)
    strategy_cumulative_returns = (1 + strategy_returns).cumprod()
    
    print("Backtesting completed.")
    return benchmark_cumulative_returns, strategy_cumulative_returns