import pandas as pd
import catboost as cb
import json
import os
from sklearn.metrics import mean_absolute_error
import numpy as np

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
features = [col for col in new_data.columns if col not in ['DATE', 'Week of', 'NG_Spot_Price']]
new_X = new_data[features]

print("Features used for prediction:", new_X.columns)

new_predictions = loaded_model.predict(new_X)
print("Predictions:", new_predictions)

# Calculate the percentage change for the next N weeks
def calculate_percentage_change(values, N):
    pct_changes = []
    for i in range(len(values) - N):
        pct_change = ((values[i + N] - values[i]) / values[i]) * 100
        pct_changes.append(pct_change)
    return pct_changes

N = 4

percentage_changes_pred = calculate_percentage_change(new_predictions, N)
print(f"Percentage changes for the next {N} weeks (predictions):", percentage_changes_pred)

y_true = new_data['NG_Spot_Price'].values
percentage_changes_true = calculate_percentage_change(y_true, N)
print(f"Percentage changes for the next {N} weeks (ground truth):", percentage_changes_true)

min_length = min(len(percentage_changes_pred), len(percentage_changes_true))
percentage_changes_pred = percentage_changes_pred[:min_length]
percentage_changes_true = percentage_changes_true[:min_length]

mae_pct_change = mean_absolute_error(percentage_changes_true, percentage_changes_pred)
print(f'Mean Absolute Error (MAE) for percentage changes: {mae_pct_change}')

y_true_truncated = y_true[:len(new_predictions)]
mae = mean_absolute_error(y_true_truncated, new_predictions)
print(f'Mean Absolute Error (MAE) for predictions: {mae}')