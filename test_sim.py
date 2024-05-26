import pandas as pd
import catboost as cb
import json
import os
import numpy as np
import matplotlib.pyplot as plt

model_file_path = 'catboost_ng_model.bin'
params_file_path = 'best_params.json'

if os.path.exists(model_file_path):
    loaded_model = cb.CatBoostRegressor()
    loaded_model.load_model(model_file_path)
else:
    print(f"Model file {model_file_path} does not exist.")

if os.path.exists(params_file_path):
    with open(params_file_path, 'r') as f:
        loaded_best_params = json.load(f)
    print("Loaded Best Parameters:", loaded_best_params)
else:
    print(f"Parameters file {params_file_path} does not exist.")

new_data = pd.read_csv('NG/test.csv')
new_data['DATE'] = pd.to_datetime(new_data['DATE'])
new_data.set_index('DATE', inplace=True)

monthly_data = new_data.resample('M').mean()

monthly_data.reset_index(inplace=True)
# print("Ground Truth", monthly_data['NG_Spot_Price'])
features = [col for col in monthly_data.columns if col not in ['DATE', 'Week of', 'NG_Spot_Price']]
monthly_X = monthly_data[features]

# print("Features used for prediction:", monthly_X.columns)

monthly_predictions = loaded_model.predict(monthly_X)
# print("Monthly Predictions:", monthly_predictions)

# Calculate the percentage change for the next N months
def calculate_percentage_change(values, N):
    pct_changes = []
    for i in range(len(values) - N):
        pct_change = ((values[i + N] - values[i]) / values[i]) * 100
        pct_changes.append(pct_change)
    return pct_changes

N = 6 # Number of months ahead for percentage change calculation

percentage_changes_pred = calculate_percentage_change(monthly_predictions, N)
print(f"Percentage changes for the next {N} months (predictions):", percentage_changes_pred)

y_true = monthly_data['NG_Spot_Price'].values
percentage_changes_true = calculate_percentage_change(y_true, N)
print(f"Percentage changes for the next {N} months (ground truth):", percentage_changes_true)

min_length = min(len(percentage_changes_pred), len(percentage_changes_true))
percentage_changes_pred = percentage_changes_pred[:min_length]
percentage_changes_true = percentage_changes_true[:min_length]

# Simulator
initial_aum = 1000000  # AUM
risk_tolerance = 1  # Risk factor
threshold = 10
max_risk_amount = initial_aum * risk_tolerance

portfolio_value = initial_aum
position_values = []
portfolio_values = []
order_book = []
trade_dates = []

#Static exit after each month (N months didn't work)

for i in range(len(percentage_changes_pred)):
    position_size = max_risk_amount * abs(percentage_changes_pred[i]) / 100
    position_direction = np.sign(percentage_changes_pred[i])

    if abs(percentage_changes_pred[i]) > threshold:
        direction = 'LONG' if position_direction > 0 else 'SHORT'
        order_book.append((direction, position_size))
        
        profit_loss = (position_size * percentage_changes_true[i] / 100 * position_direction) - position_size * 0.05 #Includes 5% commission
        portfolio_value += profit_loss
        position_values.append(profit_loss)
        portfolio_values.append(portfolio_value)
        trade_dates.append(monthly_data['DATE'].iloc[i])

# Dynamic exit (not realistic, but ok)

# positions = []
# for i in range(len(percentage_changes_pred)):
#     position_size = max_risk_amount * abs(percentage_changes_pred[i]) / 100
#     position_direction = np.sign(percentage_changes_pred[i])

#     if abs(percentage_changes_pred[i]) > 20:
#         direction = 'LONG' if position_direction > 0 else 'SHORT'
#         entry_price = monthly_data['NG_Spot_Price'].iloc[i]
#         order_book.append((direction, position_size, entry_price))
#         positions.append((direction, position_size, entry_price))

#     for j, position in enumerate(positions):
#         direction, size, entry_price = position
#         exit_price = monthly_data['NG_Spot_Price'].iloc[i + 1]
#         profit_loss = size * (exit_price - entry_price) * (1 if direction == 'LONG' else -1)
#         portfolio_value += profit_loss
#         position_values.append(profit_loss)
#         portfolio_values.append(portfolio_value)
#         positions[j] = None 

#     positions = [p for p in positions if p is not None]

#Metrics
returns = np.array(position_values)
mean_return = np.mean(returns)
std_return = np.std(returns)
sharpe_ratio = mean_return / std_return * np.sqrt(12)

cumulative_returns = np.cumsum(returns)
running_max = np.maximum.accumulate(cumulative_returns)
drawdown = cumulative_returns - running_max
max_drawdown = drawdown.min()
years = (monthly_data['DATE'].iloc[min_length] - monthly_data['DATE'].iloc[0]).days / 365.25
cagr = ((portfolio_value / initial_aum) ** (1 / years) - 1) * 100

print(f'Initial aum: {initial_aum}')
print(f'Risk: {risk_tolerance * 100}', '%')
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Maximum Drawdown: {max_drawdown}')
print(f'Max DD %: {(np.abs(max_drawdown)/initial_aum) * 100}', '%')
print(f'Final Portfolio Value: {portfolio_value}')
print(f'CAGR: {cagr}')


# Print the order book
print("\nOrder Book:")
print("Direction\tSize")
for order in order_book:
    print(f"{order[0]}\t{order[1]:.2f}")

# Print order book (Dynamic)
# print("\nOrder Book:")
# print("Direction\tSize\tEntry Price")
# for order in order_book:
#     print(f"{order[0]}\t{order[1]:.2f}\t{order[2]:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(monthly_data['DATE'], y_true, label='Actual NG Spot Price', color='blue', marker='o')
plt.plot(monthly_data['DATE'][:len(monthly_predictions)], monthly_predictions, label='Predicted NG Spot Price', color='red', marker='x')
plt.xlabel('Date')
plt.ylabel('NG Spot Price')
plt.title('Actual vs Predicted NG Spot Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(trade_dates, portfolio_values, marker='o')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title(f'Portfolio Value Over Time (CAGR: {cagr:.2f}%)')
plt.grid(True)
plt.show()