import pandas as pd

demo = pd.read_parquet('transactions dataset/demographics_clean.parquet')
txn = pd.read_parquet('transactions dataset/txn_customer_features.parquet')
fin = pd.read_parquet('transactions dataset/fin_customer_features.parquet')
train = pd.read_csv('transactions dataset/Train.csv')

df = train.merge(demo, on='UniqueID', how='left')
df = df.merge(txn, on='UniqueID', how='left')
df = df.merge(fin, on='UniqueID', how='left')

print('merged shape:', df.shape)
print(df.isnull().sum()[df.isnull().sum() > 0])

df.to_parquet('transactions dataset/merged_train.parquet', index=False)
