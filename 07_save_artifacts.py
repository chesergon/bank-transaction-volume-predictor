import pickle
import json

# Save the trained LightGBM model
with open('transactions dataset/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the KMeans model + scaler (needed to assign new customers to clusters)
with open('transactions dataset/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('transactions dataset/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the exact feature column order the model expects
with open('transactions dataset/feature_columns.json', 'w') as f:
    json.dump(list(X_train.columns), f)

# Save median/default values for each feature (used to prefill the app's form)
medians = X_train.median(numeric_only=True).to_dict()
with open('transactions dataset/feature_medians.json', 'w') as f:
    json.dump(medians, f)

print("Artifacts saved: model.pkl, kmeans.pkl, scaler.pkl, feature_columns.json, feature_medians.json")

# Save cluster summary (avg behavior per cluster) for the app's overview page
cluster_features = ['txn_last_1m', 'txn_last_3m', 'txn_last_6m',
                     'avg_monthly_txn', 'credit_ratio', 'history_days']
cluster_summary = df.groupby('cluster')[cluster_features].mean().round(1)
cluster_summary['count'] = df['cluster'].value_counts().sort_index()
cluster_summary.to_json('transactions dataset/cluster_summary.json', orient='index')
print("Also saved: cluster_summary.json")
