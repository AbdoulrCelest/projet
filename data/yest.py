import pandas as pd
df = pd.read_parquet('processed\Baleares_processed_daily.parquet')
print(df.head())
print(df.columns)
