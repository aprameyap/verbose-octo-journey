import pandas as pd
import catboost as cb
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import talib

# Load the model and parameters
model_file_path = 'catboost_ng_model.bin'
params_file_path = 'best_params.json'

if os.path.exists(model_file_path):
    loaded_model = cb.CatBoostRegressor()
    loaded_model.load_model(model_file_path)
else:
    raise FileNotFoundError(f"Model file {model_file_path} does not exist.")

if os.path.exists(params_file_path):
    with open(params_file_path, 'r') as f:
        loaded_best_params = json.load(f)
    print("Loaded Best Parameters:", loaded_best_params)
else:
    raise FileNotFoundError(f"Parameters file {params_file_path} does not exist.")

# Load and prepare data
new_data = pd.read_csv('NG/test.csv')
new_data['DATE'] = pd.to_datetime(new_data['DATE'])
new_data.set_index('DATE', inplace=True)

features = [col for col in new_data.columns if col not in ['NG_Spot_Price']]
weekly_X = new_data[features]

def calculate_percentage_change(values, N):
    pct_changes = []
    for i in range(len(values) - N):
        pct_change = ((values[i + N] - values[i]) / values[i]) * 100
        pct_changes.append(pct_change)
    return pct_changes

def backtest_model(N, weekly_X, new_data, initial_aum=1000000, risk_tolerance=0.75, threshold=15):
    weekly_predictions = loaded_model.predict(weekly_X)

    percentage_changes_pred = calculate_percentage_change(weekly_predictions, N)
    y_true = new_data['NG_Spot_Price'].values
    percentage_changes_true = calculate_percentage_change(y_true, N)

    min_length = min(len(percentage_changes_pred), len(percentage_changes_true))
    percentage_changes_pred = percentage_changes_pred[:min_length]
    percentage_changes_true = percentage_changes_true[:min_length]
    dates = new_data.index[N:N + min_length]

    # Simulator
    max_risk_amount = initial_aum * risk_tolerance
    portfolio_value = initial_aum
    available_balance = initial_aum
    position_values = []
    portfolio_values = []
    trade_dates = []
    open_positions = []

    for i in range(len(percentage_changes_pred)):
        position_size = max_risk_amount * abs(percentage_changes_pred[i]) / 100
        position_direction = np.sign(percentage_changes_pred[i])
        entry_date = dates[i]
        exit_date = dates[i + N] if i + N < len(dates) else dates[-1]

        if abs(percentage_changes_pred[i]) > threshold and ((exit_date - entry_date).days) >= N:
            if position_size > available_balance:
                position_size = available_balance

            if position_size > 0:
                open_positions.append({
                    'size': position_size,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'position_direction': position_direction
                })
                available_balance -= position_size

        position_exits = []
        for position in open_positions:
            if position['exit_date'] == entry_date:
                profit_loss = (position['size'] * percentage_changes_true[i - N] / 100 * position['position_direction']) - position['size'] * 0.05  # Includes 5% commission
                portfolio_value += profit_loss
                available_balance += position['size'] + profit_loss
                position_values.append(profit_loss)
                portfolio_values.append(portfolio_value)
                trade_dates.append(position['entry_date'])
                position_exits.append(position)

        for position in position_exits:
            open_positions.remove(position)

    # Metrics
    returns = np.array(position_values)
    mean_return = np.mean(returns) if len(returns) > 0 else 0
    std_return = np.std(returns) if len(returns) > 0 else 1
    sharpe_ratio = mean_return / std_return * np.sqrt(12) if std_return != 0 else 0
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    years = (dates[-1] - dates[0]).days / 365.25 if len(dates) > 1 else 1
    cagr = ((portfolio_value / initial_aum) ** (1 / years) - 1) * 100 if years != 0 else 0

    return {
        'N': N,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Max DD %': (np.abs(max_drawdown) / initial_aum) * 100,
        'Final Portfolio Value': portfolio_value,
        'CAGR': cagr,
        'Portfolio Values': portfolio_values,
        'Trade Dates': trade_dates
    }

# Range of N values to test
N_values = range(1, 52)
results = []

for N in N_values:
    result = backtest_model(N, weekly_X, new_data)
    results.append(result)
    print(f"Tested N = {N}")

results_df = pd.DataFrame(results)

optimal_N = results_df.loc[results_df['Sharpe Ratio'].idxmax()]['N']
print(f"Optimal N: {optimal_N}")

# Plot performance metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('N')
ax1.set_ylabel('Sharpe Ratio', color=color)
ax1.plot(results_df['N'], results_df['Sharpe Ratio'], label='Sharpe Ratio', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('CAGR (%)', color=color)
ax2.plot(results_df['N'], results_df['CAGR'], label='CAGR', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='lower right')

ax3 = ax1.twinx()
color = 'tab:red'
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('Max Drawdown %', color=color)
ax3.plot(results_df['N'], results_df['Max DD %'], label='Max Drawdown %', color=color)
ax3.tick_params(axis='y', labelcolor=color)
ax3.legend(loc='upper right')

# Highlight optimal N
plt.axvline(x=optimal_N, color='gray', linestyle='--', linewidth=1, label=f'Optimal N = {optimal_N}')
plt.text(optimal_N, max(results_df['Sharpe Ratio']), f'N = {optimal_N}', color='gray', ha='center')

fig.tight_layout()
plt.title('Key Metrics vs N')
plt.grid(True)
plt.legend()
plt.show()

# Use optimal N for final backtest and visualization
final_result = backtest_model(optimal_N, weekly_X, new_data)

plt.figure(figsize=(10, 6))
plt.plot(final_result['Trade Dates'], final_result['Portfolio Values'], marker='o')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title(f'Portfolio Value Over Time (Optimal N = {optimal_N})')
plt.grid(True)
plt.show()