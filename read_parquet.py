import pandas as pd
df = pd.read_parquet('datasets/vcpi_tvc-kdl-010_counts.parquet')

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nNumeric summary:")
print(df.describe())