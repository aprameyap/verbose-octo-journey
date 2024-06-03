import pandas as pd

# Function to preprocess a dataframe by setting date, resampling weekly, and interpolating
def preprocess_df(df, date_col, date_format, freq='W-FRI'):
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df.set_index(date_col, inplace=True)
    weekly_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    weekly_df = df.reindex(weekly_dates).interpolate(method='linear')
    weekly_df.reset_index(inplace=True)
    weekly_df.rename(columns={'index': 'DATE'}, inplace=True)
    return weekly_df

# Load and preprocess the core natural gas futures data
ng_df = pd.read_csv('NG_new/NG1.csv')
ng_df = preprocess_df(ng_df, 'Week of', '%m/%d/%Y')

# Load and preprocess imports data
imports_df = pd.read_csv('NG_new/imports.csv')
imports_df = preprocess_df(imports_df, 'Month', '%b %Y')
imports_df.rename(columns={'U.S. Natural Gas Imports MMcf': 'Imports_MMcf'}, inplace=True)

# Load and preprocess exports data
exports_df = pd.read_csv('NG_new/exports.csv')
exports_df = preprocess_df(exports_df, 'Month', '%b %Y')
exports_df.rename(columns={'U.S. Natural Gas Exports MMcf': 'Exports_MMcf'}, inplace=True)

# Combine the dataframes
combined_df = pd.merge(ng_df, imports_df, on='DATE', how='left')
combined_df = pd.merge(combined_df, exports_df, on='DATE', how='left')
combined_df.dropna(subset=['Imports_MMcf', 'Exports_MMcf'], inplace=True)

# Load and preprocess crude oil data
crude_df = pd.read_csv('NG_new/crude_oil.csv')
crude_df = preprocess_df(crude_df, 'Week of', '%m/%d/%Y')
crude_df.rename(columns={'Cushing OK Crude Oil Future Contract 1 $/bbl': 'Crude_Oil_Price'}, inplace=True)

# Merge crude oil data
final_df = pd.merge(combined_df, crude_df, on='DATE', how='left')

# Load and preprocess CPI data
cpi_df = pd.read_csv('NG_new/CPI.csv')
cpi_df = preprocess_df(cpi_df, 'DATE', '%Y-%m-%d')
cpi_df.rename(columns={'MEDCPIM158SFRBCLE': 'CPI'}, inplace=True)

# Merge CPI data
final_df = pd.merge(final_df, cpi_df, on='DATE', how='left')

# Load and preprocess storm data
storms_df = pd.read_csv('NG_new/storms.csv')
storms_df['datetime'] = pd.to_datetime(storms_df[['year', 'month', 'day', 'hour']])
storms_df.set_index('datetime', inplace=True)
weekly_storms_df = storms_df.resample('W-FRI').agg({
    'wind': 'mean',
    'pressure': 'mean',
    'tropicalstorm_force_diameter': 'mean',
    'hurricane_force_diameter': 'mean'
}).reset_index()
weekly_storms_df.rename(columns={'datetime': 'DATE'}, inplace=True)

# Merge storm data
final_df = pd.merge(final_df, weekly_storms_df, on='DATE', how='left')

n_lags = 4

# Create lagged features for the specified columns
columns_to_lag = ['Natural Gas Futures Contract 1 $/MMBTU', 'Crude_Oil_Price', 'Imports_MMcf', 'Exports_MMcf' ,'CPI']

for column in columns_to_lag:
    for lag in range(1, n_lags + 1):
        final_df[f'{column}_Lag_{lag}'] = final_df[column].shift(lag)


# Set DATE as index and remove commas from numeric columns
final_df.set_index('DATE', inplace=False)
for col in final_df.columns:
    if final_df[col].dtype == 'object':
        final_df[col] = final_df[col].str.replace(',', '').astype(float)

# Save the final dataframe to a CSV file
final_df.to_csv('NG_new/NG_dataset.csv', index=False)
