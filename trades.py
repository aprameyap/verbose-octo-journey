import pandas as pd
from catboost import CatBoostRegressor
import numpy as np

# Load the model
model = CatBoostRegressor()
model.load_model('NG_catboost_model.bin')

# Load the data
data = pd.read_csv('NG/NG_dataset.csv')

# Identify non-numeric columns
non_numeric_cols = data.select_dtypes(exclude=['float', 'int']).columns

# Preprocess the data to match the features used in training
features = data.drop(['DATE', 'NG_Spot_Price'] + list(non_numeric_cols), axis=1)

# Predict the spot prices
predicted_spot_prices = model.predict(features)

# Create a DataFrame with dates and predicted spot prices
predicted_data = pd.DataFrame({'DATE': data['DATE'], 'Predicted_Spot_Price': predicted_spot_prices})

# Merge with the original data for analysis
merged_data = pd.merge(data, predicted_data, on='DATE')

# Calculate additional technical indicators
merged_data['SMA_5'] = merged_data['NG_Spot_Price'].rolling(window=5).mean()
merged_data['SMA_20'] = merged_data['NG_Spot_Price'].rolling(window=20).mean()
merged_data['RSI'] = 100 - (100 / (1 + merged_data['NG_Spot_Price'].pct_change().rolling(window=14).apply(lambda x: np.mean(x[x > 0]) / np.mean(-x[x < 0]), raw=False)))

# Generate trading signals
merged_data['Signal'] = 'Hold'
merged_data.loc[(merged_data['Predicted_Spot_Price'] > merged_data['SMA_20']) & (merged_data['SMA_5'] > merged_data['SMA_20']) & (merged_data['RSI'] < 70), 'Signal'] = 'Buy'
merged_data.loc[(merged_data['Predicted_Spot_Price'] < merged_data['SMA_20']) & (merged_data['SMA_5'] < merged_data['SMA_20']) & (merged_data['RSI'] > 30), 'Signal'] = 'Sell'

# Backtest the trading strategy
initial_capital = 10000  # Initial capital in USD
capital = initial_capital  # Current capital
position = 0  # Current position (number of contracts)
position_value = 0  # Value of the current position
capital_history = [initial_capital]  # List to track capital over time
stop_loss = 0.05  # Stop loss threshold (5%)
take_profit = 0.10  # Take profit threshold (10%)

for i in range(1, len(merged_data)):
    current_price = merged_data['NG_Spot_Price'][i]
    
    if position > 0:
        # Check stop loss
        if current_price < position_value * (1 - stop_loss):
            capital = position * current_price
            position = 0
        # Check take profit
        elif current_price > position_value * (1 + take_profit):
            capital = position * current_price
            position = 0
    
    if merged_data['Signal'][i] == 'Buy' and capital > 0:
        position = capital / current_price
        position_value = current_price
        capital = 0
    elif merged_data['Signal'][i] == 'Sell' and position > 0:
        capital = position * current_price
        position = 0
    
    capital_history.append(capital + position * current_price)

# Calculate relevant metrics
final_capital = capital_history[-1]
returns = (final_capital - initial_capital) / initial_capital
sharpe_ratio = (np.mean(np.diff(capital_history)) / np.std(np.diff(capital_history))) * np.sqrt(252)  # Assuming 252 trading days
max_drawdown = np.max(np.maximum.accumulate(capital_history) - capital_history) / initial_capital
profit_factor = np.sum(np.diff(capital_history)[np.diff(capital_history) > 0]) / np.abs(np.sum(np.diff(capital_history)[np.diff(capital_history) < 0]))

print("Returns:", returns)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)
print("Profit Factor:", profit_factor)
