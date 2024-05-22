import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as ts

# Function to perform Monte Carlo simulation
def monte_carlo_simulations(close_prices, num_simulations, num_days):
    daily_returns = close_prices.pct_change().dropna()
    mean_daily_return = daily_returns.mean()
    volatility = daily_returns.std()

    simulation_results = []

    for i in range(num_simulations):
        simulated_prices = [close_prices.iloc[0:-1]]

        for day in range(num_days):
            simulated_price = simulated_prices[-1] * (1 + np.random.normal(mean_daily_return, volatility))
            simulated_prices.append(simulated_price)

        simulation_results.append(simulated_prices)

    return simulation_results


# Function to plot simulation results
def plot_simulation(simulation_results, ticker_symbol):
    plt.figure(figsize=(10, 6))
    plt.title('MC Simulation for ' + ticker_symbol)
    plt.xlabel('Days')
    plt.ylabel('Price')
    for result in simulation_results:
        plt.plot(result)
    plt.show()

# Function to preprocess metric data and calculate necessary metrics
def preprocess_metric_data(metric_data):
    # Convert date column to datetime format
    metric_data['Date'] = pd.to_datetime(metric_data['Date'])

    # Calculate daily returns
    metric_data['Daily_Return'] = metric_data['WAP'].pct_change()

    # Calculate volatility (standard deviation of daily returns)
    volatility = metric_data['Daily_Return'].std()

    return metric_data, volatility

# Read metric data from CSV
metric_data = pd.read_csv('RIL.csv')

# Preprocess metric data and calculate necessary metrics
metric_data, volatility = preprocess_metric_data(metric_data)

# Extract 'Close Price' column
close_prices = metric_data['Close Price']

# Perform Monte Carlo simulations
num_simulations = 1500
num_days = 7
simulation_results = monte_carlo_simulations(close_prices, num_simulations, num_days)

# Plot simulation results
plot_simulation(simulation_results, 'RIL')

# Flatten the list of simulated prices
all_simulated_prices = [price for sim in simulation_results for price in sim]

# Plot a histogram of the simulated prices
plt.figure(figsize=(10, 6))
sns.kdeplot(all_simulated_prices, color='skyblue', linewidth=2)

# Evaluate KDE to get density values
kde_values = sns.kdeplot(all_simulated_prices, color='skyblue', linewidth=0.5).get_lines()[0].get_data()

# Find the value with the highest density
max_density_index = np.argmax(kde_values[1])
max_density_price = kde_values[0][max_density_index]
max_density = kde_values[1][max_density_index]

plt.axvline(max_density_price, color='red', linestyle='--', label='Projected Price: {:.2f}'.format(max_density_price))
plt.title('Distribution of Simulated Prices')
plt.xlabel('Price')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()