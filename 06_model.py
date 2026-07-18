import pandas as pd
df = pd.read_parquet('transactions dataset/clean_train.parquet')
df.head()
df.shape
X = df.drop(columns=['UniqueID', 'next_3m_txn_count', 'log_target', 
                      'BirthDate', 'min_date', 'max_date', 'last_run_date'])
y = df['log_target']
X = pd.get_dummies(X, drop_first=True)
import re
X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)) for c in X.columns]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
import numpy as np
from sklearn.metrics import mean_squared_error

preds_log = model.predict(X_test)
preds = np.expm1(preds_log)
actual = np.expm1(y_test)

rmsle = np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(preds)))
print("RMSLE:", rmsle)

###RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)
actual = np.expm1(y_test)

rmsle = np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(preds)))
print("RMSLE:", rmsle)

### LGBMRegressor

from lightgbm import LGBMRegressor

model = LGBMRegressor(random_state=42, importance_type='gain')
model.fit(X_train, y_train)
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)
actual = np.expm1(y_test)

rmsle = np.sqrt(mean_squared_error(np.log1p(actual
                                            ), np.log1p(preds)))
print("RMSLE:", rmsle)

###Feature Importances 

importances = pd.Series(model.feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False).head(15))

##K-means clustering

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# pick a subset of behavioral features that make sense for segmentation
cluster_features = ['txn_last_1m', 'txn_last_3m', 'txn_last_6m', 
                     'avg_monthly_txn', 'credit_ratio', 'history_days']
X_cluster = df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

import matplotlib.pyplot as plt

inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig('elbow_plot.png')
#plt.show()
print(inertias)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(df['cluster'].value_counts())
print(df.groupby('cluster')[cluster_features].mean())

# add cluster label onto the full df 
X = df.drop(columns=['UniqueID', 'next_3m_txn_count', 'log_target',
                      'BirthDate', 'min_date', 'max_date', 'last_run_date'])
X = pd.get_dummies(X)
X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)) for c in X.columns]

y = df['log_target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

preds = np.expm1(model.predict(X_test))
actual = np.expm1(y_test)
rmsle = np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(preds)))
print("RMSLE with cluster feature:", rmsle)

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

drop_cols = ['UniqueID', 'next_3m_txn_count', 'log_target',
             'BirthDate', 'min_date', 'max_date', 'last_run_date']

for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c]
    
    X_sub = sub.drop(columns=drop_cols)
    X_sub = pd.get_dummies(X_sub)
    X_sub.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in X_sub.columns]
    y_sub = sub['log_target']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
    
    m = LGBMRegressor(random_state=42)
    m.fit(X_tr, y_tr)
    
    preds = np.expm1(m.predict(X_te))
    actual = np.expm1(y_te)
    rmsle = np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(preds)))
    
    print(f"Cluster {c}: RMSLE = {rmsle:.4f}, n = {len(sub)}")

    ### SHAP
    import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# global summary: which features matter most across all predictions
shap.summary_plot(shap_values, X_test, show=False)
import matplotlib.pyplot as plt
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.show()






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
