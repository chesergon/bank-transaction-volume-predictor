import pandas as pd
import numpy as np

df = pd.read_parquet('transactions dataset/merged_train.parquet')

# log-transform target (RMSLE-style, and fixes the skew we saw earlier)
df['log_target'] = np.log1p(df['next_3m_txn_count'])

# correlation of numeric features with log target
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['next_3m_txn_count', 'log_target'])
corrs = df[numeric_cols].corrwith(df['log_target']).sort_values(ascending=False)
print("=== Correlation with log(target) ===")
print(corrs)

print()
print("=== Categorical feature cardinality ===")
cat_cols = df.select_dtypes(include=['object', 'string']).columns
for c in cat_cols:
    print(f"{c}: {df[c].nunique()} unique values")
