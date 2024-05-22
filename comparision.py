from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('path_to_your_dataset.csv')
data['DATE'] = pd.to_datetime(data['DATE'])

# Define features and target
features = [col for col in data.columns if col not in ['DATE', 'Week of', 'NG_Spot_Price']]
X = data[features]
y = data['NG_Spot_Price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model with best parameters
catboost_model = cb.CatBoostRegressor(**loaded_best_params)
catboost_model.fit(X_train, y_train)

# Make predictions and calculate MAE for CatBoost model
catboost_predictions = catboost_model.predict(X_test)
catboost_mae = mean_absolute_error(y_test, catboost_predictions)

# Train a simple Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions and calculate MAE for Linear Regression model
linear_predictions = linear_model.predict(X_test)
linear_mae = mean_absolute_error(y_test, linear_predictions)

print(f'CatBoost MAE: {catboost_mae}')
print(f'Linear Regression MAE: {linear_mae}')
