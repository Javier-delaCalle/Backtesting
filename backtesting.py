# Import necessary libraries
import yfinance as yf                 # For downloading historical financial data
import numpy as np                    # For numerical calculations
import pandas as pd                   # For data manipulation
import matplotlib.pyplot as plt       # For plotting
from scipy.stats import linregress    # For linear regression (to calculate alpha and beta)
import datetime                       # For handling date calculations

# ----------------------------
# 1. Define portfolio parameters
# ----------------------------
# List of tickers (only the 5 risky assets)
tickers = ["VOO", "FEZ", "8PSG.DE", "IB01.L", "IBCL.DE"]
# Weights for the risky assets (sum to 0.954 or 95.46%)
risky_weights = np.array([0.1396, 0.1150, 0.25, 0.20, 0.25])
# Weight for the cash component (4.54%, assumed risk-free with 0% return)
cash_weight = 0.0454

# ----------------------------
# 2. Define backtesting period (exactly the last 5 years until today)
# ----------------------------
end_date = datetime.date.today().strftime("%Y-%m-%d")
start_date = (datetime.date.today() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")

# ----------------------------
# 3. Define benchmark ticker
# ----------------------------
benchmark_ticker = "SPY"  # Using SPY as the market proxy

# ----------------------------
# 4. Function to download historical data
# ----------------------------
def download_data(tickers, start, end):
    """
    Download adjusted close prices for the given tickers between the specified dates.
    auto_adjust=False ensures the 'Adj Close' column is retained.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
    return data

# ----------------------------
# 5. Calculate daily portfolio returns
# ----------------------------
def compute_portfolio_returns(price_data, risky_weights, cash_weight):
    """
    Calculate the portfolio's daily returns.
    Cash is assumed to have 0% daily return, so the portfolio return is the weighted
    return of the risky assets applied to the invested portion (1 - cash_weight).
    """
    daily_returns = price_data.pct_change().dropna()           # Daily returns for each asset
    risky_component = daily_returns.dot(risky_weights)         # Weighted return of risky assets
    portfolio_returns = risky_component * (1 - cash_weight)    # Adjust for cash allocation
    return portfolio_returns

# ----------------------------
# 6. Calculate cumulative returns
# ----------------------------
def compute_cumulative_returns(daily_returns):
    """
    Compute cumulative returns from daily returns.
    """
    return (1 + daily_returns).cumprod()

# ----------------------------
# 7. Calculate performance statistics
# ----------------------------
def performance_statistics(cum_returns, daily_returns):
    """
    Compute key performance metrics:
      - Total Period Return
      - Annualized Return
      - Annualized Volatility
      - Sharpe Ratio (risk-free rate assumed 0)
    """
    total_return = cum_returns.iloc[-1] - 1
    days = len(daily_returns)
    annualized_return = (cum_returns.iloc[-1])**(252/days) - 1
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    return total_return, annualized_return, annualized_vol, sharpe_ratio

# ----------------------------
# 8. Calculate alpha and beta
# ----------------------------
def compute_alpha_beta(portfolio_returns, benchmark_returns):
    """
    Calculate the portfolio's alpha and beta versus the benchmark.
    Regression: portfolio_return = alpha + beta * benchmark_return + error
    Daily alpha is annualized by multiplying by 252.
    """
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner")
    df.columns = ["Portfolio", "Benchmark"]
    regression = linregress(df["Benchmark"], df["Portfolio"])
    beta = regression.slope
    alpha_daily = regression.intercept
    alpha_annualized = alpha_daily * 252
    return alpha_annualized, beta

# ----------------------------
# 9. Plot backtest results
# ----------------------------
def plot_backtest(cum_returns, daily_returns, alpha, beta, ann_ret, ann_vol, sharpe):
    """
    Generate two plots:
      1) Portfolio cumulative returns over time.
      2) Histogram of daily portfolio returns.
    Annotate with Total Return, Annualized Return, Annualized Volatility,
    Sharpe Ratio, Annualized Alpha, and Beta. Graph text remains in English.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4)

    # Plot cumulative returns
    ax1.plot(cum_returns.index, cum_returns, label="Portfolio")
    ax1.set_title("Portfolio Cumulative Returns")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True)
    ax1.legend()

    # Annotate performance stats
    total_pct = (cum_returns.iloc[-1] - 1) * 100
    annotation = (
        f"Total Return: {total_pct:.2f}%\n"
        f"Annualized Return: {ann_ret*100:.2f}%\n"
        f"Annualized Volatility: {ann_vol*100:.2f}%\n"
        f"Sharpe Ratio: {sharpe:.2f}\n"
        f"Annualized Alpha: {alpha:.2f}%\n"
        f"Beta: {beta:.2f}"
    )
    ax1.text(0.02, 0.92, annotation, transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot histogram of daily returns
    ax2.hist(daily_returns, bins=50, color='skyblue', edgecolor='black')
    ax2.set_title("Distribution of Daily Portfolio Returns")
    ax2.set_xlabel("Daily Return")
    ax2.set_ylabel("Frequency")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ----------------------------
# 10. Main function to run the backtest
# ----------------------------
def main():
    # Download price data for portfolio assets
    price_data = download_data(tickers, start_date, end_date)
    print("Historical Price Data (first rows):")
    print(price_data.head())

    # Download benchmark data
    benchmark_prices = yf.download(benchmark_ticker, start=start_date, end=end_date,
                                   auto_adjust=False)["Adj Close"].dropna()

    # Calculate returns
    portfolio_returns = compute_portfolio_returns(price_data, risky_weights, cash_weight)
    cum_returns = compute_cumulative_returns(portfolio_returns)

    # Compute performance stats
    total_ret, ann_ret, ann_vol, sharpe = performance_statistics(cum_returns, portfolio_returns)
    print("\nPortfolio Performance Statistics:")
    print(f"Total Period Return: {total_ret:.2%}")
    print(f"Annualized Return: {ann_ret:.2%}")
    print(f"Annualized Volatility: {ann_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    # Benchmark returns
    benchmark_returns = benchmark_prices.pct_change().dropna()

    # Compute alpha and beta
    alpha, beta = compute_alpha_beta(portfolio_returns, benchmark_returns)
    print("\nAlpha and Beta Analysis:")
    print(f"Annualized Alpha: {alpha:.2f}%")
    print(f"Beta: {beta:.2f}")

    # Plot results
    plot_backtest(cum_returns, portfolio_returns, alpha, beta, ann_ret, ann_vol, sharpe)

if __name__ == "__main__":
    main()
