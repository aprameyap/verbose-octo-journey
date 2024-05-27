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
monthly_data = new_data.resample('M').mean()
monthly_data.reset_index(inplace=True)

features = [col for col in monthly_data.columns if col not in ['DATE', 'Week of', 'NG_Spot_Price']]
monthly_X = monthly_data[features]
monthly_predictions = loaded_model.predict(monthly_X)

def calculate_percentage_change(values, N):
    pct_changes = []
    for i in range(len(values) - N):
        pct_change = ((values[i + N] - values[i]) / values[i]) * 100
        pct_changes.append(pct_change)
    return pct_changes

threshold = 15
initial_aum = 1000000
risk_tolerance = 1

def evaluate_strategy(N, threshold=threshold, initial_aum=initial_aum, risk_tolerance=risk_tolerance):
    percentage_changes_pred = calculate_percentage_change(monthly_predictions, N)
    y_true = monthly_data['NG_Spot_Price'].values
    percentage_changes_true = calculate_percentage_change(y_true, N)

    min_length = min(len(percentage_changes_pred), len(percentage_changes_true))
    percentage_changes_pred = percentage_changes_pred[:min_length]
    percentage_changes_true = percentage_changes_true[:min_length]

    if len(monthly_data['DATE']) < N + min_length:
        return None

    dates = monthly_data['DATE'][N:N + min_length]

    max_risk_amount = initial_aum * risk_tolerance
    portfolio_value = initial_aum
    position_values = []
    portfolio_values = []
    order_book = []
    trade_dates = []

    for i in range(len(percentage_changes_pred)):
        position_size = max_risk_amount * abs(percentage_changes_pred[i]) / 100
        position_direction = np.sign(percentage_changes_pred[i])

        if abs(percentage_changes_pred[i]) > threshold:
            direction = 'LONG' if position_direction > 0 else 'SHORT'
            order_book.append((direction, position_size))

            profit_loss = (position_size * percentage_changes_true[i] / 100 * position_direction) - position_size * 0.05  # Includes 5% commission
            portfolio_value += profit_loss
            position_values.append(profit_loss)
            portfolio_values.append(portfolio_value)
            trade_dates.append(dates.iloc[i])

    returns = np.array(position_values)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = mean_return / std_return * np.sqrt(12) if std_return != 0 else 0

    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()

    if len(dates) > 0:
        years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        cagr = ((portfolio_value / initial_aum) ** (1 / years) - 1) * 100 if years != 0 else 0
    else:
        cagr = 0

    return {
        'N': N,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'CAGR': cagr,
        'Portfolio Value': portfolio_value,
        'Order Book': order_book,
        'Trade Dates': trade_dates,
        'Portfolio Values': portfolio_values,
        'Percentage Changes True': percentage_changes_true,
        'Percentage Changes Pred': percentage_changes_pred,
        'Dates': dates
    }

best_result = None
best_N = None

for N in range(1, 9):  # Adjust the range as needed
    result = evaluate_strategy(N)
    if result and (best_result is None or (result['Sharpe Ratio'] > best_result['Sharpe Ratio'] and result['Max Drawdown'] > -initial_aum)):
        best_result = result
        best_N = N

if best_result:
    print(f'Best N: {best_N}')
    print(f"Sharpe Ratio: {best_result['Sharpe Ratio']}")
    print(f"Max Drawdown: {best_result['Max Drawdown']}")
    print(f"Max DD %: {(np.abs(best_result['Max Drawdown']) / initial_aum) * 100}%")
    print(f"Final Portfolio Value: {best_result['Portfolio Value']}")
    print(f"CAGR: {best_result['CAGR']}")

    # Print the order book
    print("\nOrder Book:")
    print("Direction\tSize")
    for order in best_result['Order Book']:
        print(f"{order[0]}\t{order[1]:.2f}")

    # Graphing
    plt.figure(figsize=(10, 6))
    plt.plot(best_result['Dates'], best_result['Percentage Changes True'], label=f'Actual NG Spot Price (percent change, next {best_N} month(s))', color='blue', marker='o')
    plt.plot(best_result['Dates'], best_result['Percentage Changes Pred'], label=f'Predicted NG Spot Price (predicted percent change, next {best_N} month(s))', color='red', marker='x')
    plt.xlabel('Date')
    plt.ylabel('NG Spot Price change (%)')
    plt.title('Actual vs Predicted changes in NG Spot Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(best_result['Trade Dates'], best_result['Portfolio Values'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title(f'Portfolio Value Over Time (CAGR: {best_result["CAGR"]:.2f}%, Sharpe ratio: {best_result["Sharpe Ratio"]:.2f}, Max DD %: {((np.abs(best_result["Max Drawdown"]) / initial_aum) * 100):.2f}%)')
    plt.grid(True)
    plt.show()
else:
    print("No valid results found.")
