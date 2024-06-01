"""Trade simulator and backtester"""

import pandas as pd
import catboost as cb
import json
import os
import numpy as np
import matplotlib.pyplot as plt

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
weekly_predictions = loaded_model.predict(weekly_X)

def calculate_percentage_change(values, N):
    pct_changes = []
    for i in range(len(values) - N):
        pct_change = ((values[i + N] - values[i]) / values[i]) * 100
        pct_changes.append(pct_change)
    return pct_changes

N = 36  # Number of weeks ahead for percentage change calculation and position duration

percentage_changes_pred = calculate_percentage_change(weekly_predictions, N)
# print(f"Percentage changes for the next {N} weeks (predictions):", percentage_changes_pred) #Line for debugging

y_true = new_data['NG_Spot_Price'].values
percentage_changes_true = calculate_percentage_change(y_true, N)
# print(f"Percentage changes for the next {N} weeks (ground truth):", percentage_changes_true) #Line for debugging

min_length = min(len(percentage_changes_pred), len(percentage_changes_true))
percentage_changes_pred = percentage_changes_pred[:min_length]
percentage_changes_true = percentage_changes_true[:min_length]
dates = new_data.index[N:N + min_length]

# Simulator
initial_aum = 1000000  # AUM
risk_tolerance = 0.75  # Risk factor
threshold = 15
max_risk_amount = initial_aum * risk_tolerance

portfolio_value = initial_aum
available_balance = initial_aum
position_values = []
portfolio_values = []
order_book = []
trade_dates = []
entry_exit_dates = []
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
            direction = 'LONG' if position_direction > 0 else 'SHORT'
            order_book.append((direction, position_size, entry_date, exit_date))
            open_positions.append({
                'direction': direction,
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
            entry_exit_dates.append((position['entry_date'], position['exit_date']))
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

print(f'Initial AUM: {initial_aum}')
print(f'Risk: {risk_tolerance * 100}%')
print(f'N: {N} weeks')
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Maximum Drawdown: {max_drawdown}')
print(f'Max DD %: {(np.abs(max_drawdown) / initial_aum) * 100}%')
print(f'Final Portfolio Value: {portfolio_value}')
print(f'CAGR: {cagr}')

print("\nOrder Book:")
print("Direction\tSize\tEntry Date\tExit Date")
for order in order_book:
    exit_date_str = order[3].strftime('%Y-%m-%d') if order[3] is not None else 'N/A'
    print(f"{order[0]}\t{order[1]:.2f}\t{order[2].strftime('%Y-%m-%d')}\t{exit_date_str}")

# plt.figure(figsize=(10, 6))
# plt.plot(dates, percentage_changes_true, label=f'Actual NG Spot Price (percent change, next {N} week(s))', color='blue', marker='o')
# plt.plot(dates, percentage_changes_pred, label=f'Predicted NG Spot Price (percent change, next {N} week(s))', color='red', marker='x')
# plt.xlabel('Date')
# plt.ylabel('NG Spot Price change (%)')
# plt.title('Actual vs Predicted changes in NG Spot Price')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(trade_dates, portfolio_values, marker='o')
# plt.xlabel('Date')
# plt.ylabel('Portfolio Value')
# plt.title(f'Portfolio Value Over Time (CAGR: {cagr:.2f}%, Sharpe ratio: {sharpe_ratio:.2f}, Max DD %: {((np.abs(max_drawdown) / initial_aum) * 100):.2f}%)')
# plt.grid(True)
# plt.show()