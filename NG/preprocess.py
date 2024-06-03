import pandas as pd

df = pd.read_csv('corestick.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
df.set_index('DATE', inplace=True)

start_date = df.index[0]
end_date = df.index[-1]
weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
weekly_df = df.reindex(weekly_dates).interpolate(method='linear')
weekly_df.reset_index(inplace=True)
weekly_df.rename(columns={'index': 'DATE'}, inplace=True)

imports_df = pd.read_csv('imports_monthly.csv')
imports_df['Month'] = pd.to_datetime(imports_df['Month'], format='%b %Y')
imports_df.set_index('Month', inplace=True)

weekly_imports_df = imports_df.resample('W-FRI').ffill().reset_index()
weekly_imports_df.rename(columns={'Month': 'DATE', 'U.S. Natural Gas Imports MMcf': 'Imports_MMcf'}, inplace=True)

exports_df = pd.read_csv('exports_monthly.csv')
exports_df['Month'] = pd.to_datetime(exports_df['Month'], format='%b %Y')
exports_df.set_index('Month', inplace=True)

weekly_exports_df = exports_df.resample('W-FRI').ffill().reset_index()
weekly_exports_df.rename(columns={'Month': 'DATE', 'U.S. Natural Gas Exports MMcf': 'Exports_MMcf'}, inplace=True)

combined_df = pd.merge(weekly_df, weekly_imports_df, on='DATE', how='left')
combined_df = pd.merge(combined_df, weekly_exports_df, on='DATE', how='left')
combined_df.dropna(subset=['Imports_MMcf', 'Exports_MMcf'], inplace=True)

ng_prices_df = pd.read_csv('NG_prices_weekly.csv')
ng_prices_df['Week of'] = pd.to_datetime(ng_prices_df['Week of'], format='%m/%d/%Y')
ng_prices_df.rename(columns={'Week of': 'DATE', 'Henry Hub Natural Gas Spot Price $/MMBTU': 'NG_Spot_Price'}, inplace=True)

final_df = pd.merge(combined_df, ng_prices_df, on='DATE', how='left')
final_df.dropna(subset=['NG_Spot_Price'], inplace=True)

exchange_df = pd.read_csv('exchange.csv')
exchange_df['DATE'] = pd.to_datetime(exchange_df['DATE'], format='%Y-%m-%d')

final_df = pd.merge(final_df, exchange_df, on='DATE', how='left')

production_df = pd.read_csv('production.csv')
production_df['Month'] = pd.to_datetime(production_df['Month'], format='%b %Y')
production_df.set_index('Month', inplace=True)

weekly_production_df = production_df.resample('W-FRI').ffill().reset_index()
weekly_production_df.rename(columns={'Month': 'DATE',
                                     'U.S. Natural Gas Plant Liquids Production MMcf': 'Plant_Liquids_Production_MMcf',
                                     'U.S. Natural Gas Marketed Production MMcf': 'Marketed_Production_MMcf',
                                     'U.S. Dry Natural Gas Production MMcf': 'Dry_Natural_Gas_Production_MMcf',
                                     'U.S. Natural Gas Gross Withdrawals MMcf': 'Gross_Withdrawals_MMcf'}, inplace=True)

final_df = pd.merge(final_df, weekly_production_df, on='DATE', how='left')

petroleum_df = pd.read_csv('petroleum_weekly.csv')
petroleum_df['Week of'] = pd.to_datetime(petroleum_df['Week of'], format='%m/%d/%Y')

final_df = pd.merge(final_df, petroleum_df, left_on='DATE', right_on='Week of', how='left')
final_df.drop(columns=['Week of'], inplace=True)

storage_df = pd.read_csv('ng_storage.csv')
storage_df['Week ending'] = pd.to_datetime(storage_df['Week ending'], format='%d-%b-%y')
storage_df.rename(columns={'Week ending': 'DATE'}, inplace=True)
storage_df = storage_df.iloc[:, :-1]

final_df = pd.merge(final_df, storage_df, on='DATE', how='left')

hurricane_df = pd.read_csv('storms.csv')
hurricane_df['datetime'] = pd.to_datetime(hurricane_df[['year', 'month', 'day', 'hour']])
hurricane_df.set_index('datetime', inplace=True)

weekly_hurricane_df = hurricane_df.resample('W-FRI').agg({
    'wind': 'mean',
    'pressure': 'mean',
    'tropicalstorm_force_diameter': 'mean',
    'hurricane_force_diameter': 'mean'
}).reset_index()
weekly_hurricane_df.rename(columns={'datetime': 'DATE'}, inplace=True)

final_df = pd.merge(final_df, weekly_hurricane_df, on='DATE', how='left')

final_df.set_index('DATE', inplace=False)

for col in final_df.columns:
    if final_df[col].dtype == 'object':
        final_df[col] = final_df[col].str.replace(',', '').astype(float)

final_df.to_csv('NG_dataset.csv', index=False)
