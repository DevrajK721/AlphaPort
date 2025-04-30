# Core Imports 
import numpy as np
import matplotlib.pyplot as plt

# Additional Imports
import src.DataFetcher as DF 
import src.BackTestingFramework as BTF 

# Fetch data using secrets.json file 
data_fetcher = DF.DataFetcher(secrets_path='secrets/secrets.json')

# Loop through tickers and compute the average return using backtesting framework with strategy 
returns = []
for ticker in data_fetcher.tickers:
    avg_daily_ret, _ = BTF.run_backtest(f"data/{ticker}.csv", verbose=False)
    returns.append(avg_daily_ret)
    # Convert returns to percentages
    returns_percentage = [r * 100 for r in returns]

    # Check if volatilities exists and populate it if empty
    if not hasattr(data_fetcher, 'volatilities') or not data_fetcher.volatilities:
        print("Warning: Volatilities not found or empty. Creating proxy volatilities based on return magnitude.")
        # Create proxy volatilities based on absolute returns
        data_fetcher.volatilities = {ticker: abs(ret) for ticker, ret in zip(data_fetcher.tickers, returns)}
        print("Created volatilities:", data_fetcher.volatilities)
volatilities = data_fetcher.volatilities
if not volatilities:
    # Use default values if volatilities is empty
    norm = plt.Normalize(0, 1)
    colors = ['blue'] * len(data_fetcher.tickers)
else:
    # Extract volatility values in the same order as tickers
    vol_values = [volatilities.get(ticker, 0) for ticker in data_fetcher.tickers]
    norm = plt.Normalize(min(volatilities.values()), max(volatilities.values()))
    colors = plt.cm.RdBu_r(norm(vol_values))  # RdBu_r gives red for high values, blue for low

plt.figure(figsize=(10, 6))
plt.bar(data_fetcher.tickers, returns, color=colors)
plt.axhline(0.0, color='red', linestyle='--', linewidth=2, label='Baseline')
plt.xlabel('Tickers')
plt.ylabel('Average Daily Return (%)')
plt.title('Average Daily Returns of Tickers (Red = High Volatility, Blue = Low Volatility)')
plt.xticks(rotation=45)

# Add a color bar to show the volatility scale
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm)
sm.set_array([])
ax = plt.gca()  # Get the current axes
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Volatility')

plt.tight_layout()
plt.show()

print(f"Overall Results")
print(f"Average Daily Returns of Tickers: {np.mean(returns)}")
print(f"Sharpe Ratio of Strategy: {np.mean(returns) / np.std(returns)}")
