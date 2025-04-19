# Importar las librerías necesarias
import yfinance as yf          # Para descargar datos históricos financieros
import numpy as np             # Para cálculos numéricos
import pandas as pd            # Para manipulación de datos
import matplotlib.pyplot as plt  # Para generar gráficos
from scipy.stats import linregress  # Para la regresión lineal (calcular alpha y beta)
import datetime                # Para gestionar fechas

# ----------------------------
# 1. Definir los parámetros de la cartera
# ----------------------------
# Lista de tickers (se consideran únicamente los 5 activos de riesgo)
tickers = ["VOO", "FEZ", "8PSG.DE", "IB01.L", "IBCL.DE"]
# Pesos para los activos de riesgo (suman 0.954 o 95.46%)
risky_weights = np.array([0.1396, 0.1150, 0.25, 0.20, 0.25])
# Peso para la parte en efectivo (4.54%, se asume sin riesgo y con retorno 0%)
cash_weight = 0.0454

# ----------------------------
# 2. Definir el periodo de backtesting (5 años exactos hasta hoy)
# ----------------------------
end_date = datetime.date.today().strftime("%Y-%m-%d")  
start_date = (datetime.date.today() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")

# ----------------------------
# 3. Definir el ticker del benchmark
# ----------------------------
benchmark_ticker = "SPY"  # Usamos SPY como proxy del mercado

# ----------------------------
# 4. Función para descargar datos históricos
# ----------------------------
def download_data(tickers, start, end):
    """
    Descarga los precios de cierre ajustados para los tickers indicados entre las fechas especificadas.
    Se utiliza auto_adjust=False para mantener la columna "Adj Close".
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
    return data

# ----------------------------
# 5. Calcular los retornos diarios de la cartera
# ----------------------------
def compute_portfolio_returns(price_data, risky_weights, cash_weight):
    """
    Calcula los retornos diarios de la cartera.
    Se asume que la parte en efectivo no varía (0% de retorno diario), 
    de modo que el retorno diario de la cartera equivale al retorno ponderado
    de los activos de riesgo, aplicado a la parte invertida (1 - cash_weight).
    """
    # Calcular el porcentaje de cambio diario para cada activo
    daily_returns = price_data.pct_change().dropna()
    # Calcular el retorno diario ponderado de la parte riesgosa
    risky_returns = daily_returns.dot(risky_weights)
    # Se asume que el efectivo genera retorno 0 y se aplica la ponderación restante
    portfolio_daily_returns = risky_returns * (1 - cash_weight)
    return portfolio_daily_returns

# ----------------------------
# 6. Calcular el retorno acumulado
# ----------------------------
def compute_cumulative_returns(daily_returns):
    """
    Calcula el retorno acumulado a partir de los retornos diarios.
    """
    cum_returns = (1 + daily_returns).cumprod()
    return cum_returns

# ----------------------------
# 7. Calcular estadísticas de rendimiento (Total Return, Annualized Return, Volatility, Sharpe Ratio)
# ----------------------------
def performance_statistics(cum_returns, daily_returns):
    """
    Calcula las estadísticas clave de la cartera:
      - Total Period Return
      - Annualized Return
      - Annualized Volatility
      - Sharpe Ratio (asumiendo tasa libre de riesgo = 0)
    """
    total_return = cum_returns.iloc[-1] - 1
    num_days = len(daily_returns)
    annualized_return = (cum_returns.iloc[-1])**(252/num_days) - 1
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    return total_return, annualized_return, annualized_vol, sharpe_ratio

# ----------------------------
# 8. Calcular alfa y beta de la cartera vs. benchmark
# ----------------------------
def compute_alpha_beta(portfolio_daily_returns, benchmark_daily_returns):
    """
    Calcula alfa y beta de la cartera comparada con el benchmark.
    
    Se realiza una regresión lineal de la forma:
      portfolio_return = alpha + beta * benchmark_return + error
      
    El intercepto (alpha) es diario y se annualiza multiplicándolo por 252.
    """
    # Alinear las series de retornos (intersección de fechas)
    combined = pd.concat([portfolio_daily_returns, benchmark_daily_returns], axis=1, join="inner")
    combined.columns = ["Portfolio", "Benchmark"]
    
    # Realizar la regresión lineal
    regression = linregress(combined["Benchmark"], combined["Portfolio"])
    beta = regression.slope
    alpha_daily = regression.intercept
    alpha_annualized = alpha_daily * 252  # Aproximadamente annualizado
    return alpha_annualized, beta

# ----------------------------
# 9. Función para generar gráficos
# ----------------------------
def plot_backtest(cum_returns, daily_returns, alpha, beta, ann_ret, ann_vol, sharpe):
    """
    Genera dos gráficos:
      1) Cumulative Portfolio Returns over time.
      2) Histogram of Daily Portfolio Returns.
      
    Se anota en el gráfico el Total Return, Annualized Return, Annualized Volatility,
    Sharpe Ratio, Annualized Alpha y Beta, manteniendo los textos de la gráfica en inglés.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4)

    # Gráfico 1: Retorno acumulado en inglés
    ax1.plot(cum_returns.index, cum_returns, label="Portfolio")
    ax1.set_title("Portfolio Cumulative Returns")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True)
    ax1.legend()
    
    # Anotar estadísticas de rendimiento y alfa/beta
    final_return_pct = (cum_returns.iloc[-1] - 1) * 100
    annotation_text = (
        f"Total Return: {final_return_pct:.2f}%\n"
        f"Annualized Return: {ann_ret*100:.2f}%\n"
        f"Annualized Volatility: {ann_vol*100:.2f}%\n"
        f"Sharpe Ratio: {sharpe:.2f}\n"
        f"Annualized Alpha: {alpha:.2f}%\n"
        f"Beta: {beta:.2f}"
    )
    ax1.text(0.02, 0.92, annotation_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Gráfico 2: Histograma de retornos diarios en inglés
    ax2.hist(daily_returns, bins=50, color='skyblue', edgecolor='black')
    ax2.set_title("Distribution of Daily Portfolio Returns")
    ax2.set_xlabel("Daily Return")
    ax2.set_ylabel("Frequency")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ----------------------------
# 10. Función principal: Ejecutar el backtest
# ----------------------------
def main():
    # Descargar datos históricos de precios para los activos de la cartera
    price_data = download_data(tickers, start_date, end_date)
    print("Historical Price Data (first rows):")
    print(price_data.head())

    # Descargar datos del benchmark (SPY)
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]
    benchmark_data = benchmark_data.dropna()
    
    # Calcular los retornos diarios de la cartera y el retorno acumulado
    portfolio_daily_returns = compute_portfolio_returns(price_data, risky_weights, cash_weight)
    cumulative_returns = compute_cumulative_returns(portfolio_daily_returns)
    
    # Calcular estadísticas de rendimiento
    total_ret, ann_ret, ann_vol, sharpe = performance_statistics(cumulative_returns, portfolio_daily_returns)
    print("\nPortfolio Performance Statistics:")
    print(f"Total Period Return: {total_ret:.2%}")
    print(f"Annualized Return: {ann_ret:.2%}")
    print(f"Annualized Volatility: {ann_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Calcular los retornos diarios del benchmark
    benchmark_daily_returns = benchmark_data.pct_change().dropna()
    
    # Calcular alfa y beta de la cartera contra el benchmark
    alpha, beta = compute_alpha_beta(portfolio_daily_returns, benchmark_daily_returns)
    print("\nAlpha and Beta Analysis:")
    print(f"Annualized Alpha: {alpha:.2f}%")
    print(f"Beta: {beta:.2f}")
    
    # Mostrar gráficos del backtest con anotaciones en inglés
    plot_backtest(cumulative_returns, portfolio_daily_returns, alpha, beta, ann_ret, ann_vol, sharpe)

if __name__ == "__main__":
    main()
