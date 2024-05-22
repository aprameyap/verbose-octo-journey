import pandas as pd
import catboost as cb
import json
import os

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
