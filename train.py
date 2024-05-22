import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import catboost as cb
import json

data = pd.read_csv('NG/NG_dataset.csv')

data['DATE'] = pd.to_datetime(data['DATE'])
data.drop(columns=['Week of'], inplace=True)

target = 'NG_Spot_Price'
features = data.columns.difference(['DATE', target])
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'iterations': [100, 500, 1000],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 50, 100]
}

catboost_model = cb.CatBoostRegressor(loss_function='MAE', verbose=0)
random_search = RandomizedSearchCV(catboost_model, param_distributions=param_grid, n_iter=50, cv=3, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best Parameters:", best_params)

best_model = cb.CatBoostRegressor(**best_params, loss_function='MAE')
best_model.fit(X_train, y_train)

best_model.save_model('catboost_ng_model.bin')

params_file_path = 'best_params.json'
with open(params_file_path, 'w') as f:
    json.dump(best_params, f)

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

feature_importances = best_model.get_feature_importance()
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
