import pandas as pd
import numpy as np

df = pd.read_parquet('transactions dataset/merged_train.parquet')

# --- 1. Financial nulls: genuinely "no snapshot exists" for these customers ---
fin_numeric_cols = ['fin_snapshot_count', 'avg_net_interest_income',
                     'avg_net_interest_revenue', 'n_products']
for c in fin_numeric_cols:
    df[c] = df[c].fillna(0)
df['main_product'] = df['main_product'].fillna('None')
df['has_financials'] = (df['fin_snapshot_count'] > 0).astype(int)

# --- 2. AnnualGrossIncome: missing != zero income, so flag + median fill ---
df['income_missing'] = df['AnnualGrossIncome'].isnull().astype(int)
df['AnnualGrossIncome'] = df['AnnualGrossIncome'].fillna(df['AnnualGrossIncome'].median())

# --- 3. Bucket high-cardinality categoricals ---
def bucket_top_n(series, n=10, other_label='Other'):
    top = series.value_counts().nlargest(n).index
    return series.where(series.isin(top), other_label)

df['ResidentialCityName'] = bucket_top_n(df['ResidentialCityName'], n=10)
df['CountryCodeNationality'] = bucket_top_n(df['CountryCodeNationality'], n=5)

# --- 4. Fill remaining small-null categoricals with 'Unknown' ---
small_null_cats = ['Gender', 'CustomerBankingType', 'CustomerOnboardingChannel',
                    'LowIncomeFlag']
for c in small_null_cats:
    df[c] = df[c].fillna('Unknown')

# --- 5. trend_3m_vs_6m had divide-by-zero NaNs -> fill with 0 (no growth signal available) ---
df['trend_3m_vs_6m'] = df['trend_3m_vs_6m'].fillna(0)

# --- log target for modeling ---
df['log_target'] = np.log1p(df['next_3m_txn_count'])

print('final shape:', df.shape)
print('remaining nulls:')
print(df.isnull().sum()[df.isnull().sum() > 0])

df.to_parquet('transactions dataset/clean_train.parquet', index=False)
print('saved to transactions dataset/clean_train.parquet')
