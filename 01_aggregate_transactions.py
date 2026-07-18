import pandas as pd
import numpy as np
import pyarrow.parquet as pq

pf = pq.ParquetFile('transactions dataset/transactions_features.parquet')
CUTOFF = pd.Timestamp('2015-10-31')  # last date in data, prediction window starts right after

overall_chunks = []
monthly_chunks = []

for i in range(pf.num_row_groups):
    df = pf.read_row_group(i).to_pandas()
    df['is_credit'] = (df['IsDebitCredit'] == 'Credit').astype(int)
    df['is_debit'] = (df['IsDebitCredit'] == 'Debit').astype(int)
    df['abs_amt'] = df['TransactionAmount'].abs()
    df['ym'] = df['TransactionDate'].dt.to_period('M')

    # overall per-customer partial aggregation for this chunk
    g = df.groupby('UniqueID').agg(
        txn_count=('TransactionAmount', 'count'),
        total_amt=('TransactionAmount', 'sum'),
        total_abs_amt=('abs_amt', 'sum'),
        credit_count=('is_credit', 'sum'),
        debit_count=('is_debit', 'sum'),
        min_date=('TransactionDate', 'min'),
        max_date=('TransactionDate', 'max'),
        n_accounts=('AccountID', 'nunique'),
    ).reset_index()
    overall_chunks.append(g)

    # monthly count per customer for recency features
    m = df.groupby(['UniqueID', 'ym']).size().reset_index(name='month_txn_count')
    monthly_chunks.append(m)

    print(f'row group {i+1}/{pf.num_row_groups} done, {len(df)} rows')

# combine overall chunks
overall = pd.concat(overall_chunks, ignore_index=True)
overall_final = overall.groupby('UniqueID').agg(
    txn_count=('txn_count', 'sum'),
    total_amt=('total_amt', 'sum'),
    total_abs_amt=('total_abs_amt', 'sum'),
    credit_count=('credit_count', 'sum'),
    debit_count=('debit_count', 'sum'),
    min_date=('min_date', 'min'),
    max_date=('max_date', 'max'),
    n_accounts=('n_accounts', 'max'),  # approx, nunique across chunks isn't perfectly additive
).reset_index()

overall_final['avg_amt'] = overall_final['total_amt'] / overall_final['txn_count']
overall_final['credit_ratio'] = overall_final['credit_count'] / overall_final['txn_count']
overall_final['history_days'] = (overall_final['max_date'] - overall_final['min_date']).dt.days
overall_final['avg_monthly_txn'] = overall_final['txn_count'] / (overall_final['history_days'] / 30.44).clip(lower=1)

# combine monthly chunks -> recency window features
monthly = pd.concat(monthly_chunks, ignore_index=True)
monthly_final = monthly.groupby(['UniqueID', 'ym'])['month_txn_count'].sum().reset_index()
monthly_final['ym_ts'] = monthly_final['ym'].dt.to_timestamp()

def window_count(months_back):
    start = CUTOFF - pd.DateOffset(months=months_back)
    sub = monthly_final[monthly_final['ym_ts'] > start]
    return sub.groupby('UniqueID')['month_txn_count'].sum()

recency = pd.DataFrame(index=overall_final['UniqueID'])
for mo in [1, 3, 6, 12]:
    recency[f'txn_last_{mo}m'] = window_count(mo)
recency = recency.fillna(0).reset_index()

# trend: last 3m avg vs prior 3m avg (momentum signal)
recency['recent_3m_monthly_avg'] = recency['txn_last_3m'] / 3
recency['trend_3m_vs_6m'] = recency['txn_last_3m'] / (recency['txn_last_6m'] - recency['txn_last_3m']).replace(0, np.nan)

txn_features = overall_final.merge(recency, on='UniqueID', how='left')
txn_features.to_parquet('transactions dataset/txn_customer_features.parquet', index=False)
print('final shape:', txn_features.shape)
print(txn_features.head())
