import pandas as pd

fin = pd.read_parquet('transactions dataset/financials_features.parquet')

agg = fin.groupby('UniqueID').agg(
    fin_snapshot_count=('RunDate', 'count'),
    avg_net_interest_income=('NetInterestIncome', 'mean'),
    avg_net_interest_revenue=('NetInterestRevenue', 'mean'),
    n_products=('Product', 'nunique'),
    last_run_date=('RunDate', 'max'),
).reset_index()

# most frequently used product per customer
top_product = fin.groupby('UniqueID')['Product'].agg(lambda x: x.value_counts().idxmax()).reset_index()
top_product.columns = ['UniqueID', 'main_product']

fin_features = agg.merge(top_product, on='UniqueID', how='left')
fin_features.to_parquet('transactions dataset/fin_customer_features.parquet', index=False)
print(fin_features.shape)
print(fin_features.head())
