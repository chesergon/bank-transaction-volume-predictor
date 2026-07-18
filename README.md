# Bank Transaction Volume Predictor

Predicts how many transactions a bank customer will make in the next 3 months, using their recent transaction history, account financials, and demographics. Built on real transaction-level data (18M+ rows) from a Nedbank data challenge.

**Live app:** https://bank-transaction-volume-predictor-o5gydhgbpj5m8epuic7jz8.streamlit.app/

---

## Overview

Banks need to anticipate customer activity to plan staffing, fraud monitoring, and engagement campaigns. This project builds a model that forecasts a customer's transaction volume over the next 3 months, then goes further — segmenting customers by behavior and explaining individual predictions.

**Pipeline:**
1. Aggregate 18M+ raw transaction records (chunked processing) into per-customer behavioral features
2. Engineer recency/frequency signals: transactions in the last 1/3/6/12 months, credit ratio, account history length
3. Clean and merge with account financials and demographics
4. Segment customers with K-Means clustering
5. Train and compare regression models (Linear Regression → Random Forest → LightGBM)
6. Explain predictions with SHAP
7. Deploy as an interactive Streamlit app

## Results

| Model | RMSLE |
|---|---|
| Linear Regression (baseline) | 0.607 |
| Random Forest | 0.413 |
| LightGBM | 0.409 |
| LightGBM + cluster feature | 0.405 |

**Feature importance:** recent activity (`txn_last_1m`, `txn_last_3m`) dominates prediction power — past behavior is by far the strongest signal for future behavior. Static demographic features like income had minimal impact on transaction *count* (though they may matter more for transaction *value*, which is a natural extension of this work).

**Customer segments (K-Means, k=4):**
| Segment | Behavior | Segmented model RMSLE |
|---|---|---|
| Power Users | Very high activity, long tenure | 0.321 |
| Regular Active | Solid mid-high activity | 0.263 |
| Low Activity, Long Tenure | Established but quiet | 0.428 |
| New / Low Activity | Short history (~247 days), still ramping up | 0.710 |

**Key finding:** segmenting before modeling improves accuracy substantially for established customers with rich transaction history, but *hurts* accuracy for newer customers — there's simply not enough data yet within that segment alone. A global model, which borrows strength from the whole customer base, is the safer choice for new customers, while segment-specific models work better once a customer has an established pattern.

## Tech Stack

- **Data processing:** pandas, PyArrow (chunked parquet processing for the 18M-row transaction file)
- **Modeling:** scikit-learn, LightGBM
- **Clustering:** K-Means (scikit-learn)
- **Explainability:** SHAP
- **App/deployment:** Streamlit, Streamlit Community Cloud

## Project Structure

```
├── 01_aggregate_transactions.py   # Chunked aggregation of 18M transaction rows
├── 02_aggregate_financials.py     # Financial snapshot aggregation
├── 03_merge_all.py                # Merge all feature sources with training labels
├── 04_eda.py                      # Correlation analysis, categorical cardinality checks
├── 05_clean.py                    # Null handling, categorical bucketing, target transform
├── 06_model.py                    # Model training, clustering, SHAP, artifact export
├── transactions_dataset/
│   ├── app.py                     # Streamlit app
│   ├── requirements.txt
│   ├── model.pkl                  # Trained LightGBM model
│   ├── kmeans.pkl                 # Trained K-Means model
│   ├── scaler.pkl                 # Feature scaler for clustering
│   └── *.json                     # Feature columns, medians, cluster summary
```

*(Raw data files are excluded via `.gitignore` — the dataset is licensed CC-BY-SA 4.0 for competition participants and not redistributed here.)*

## Running Locally

```bash
git clone https://github.com/chesergon/Zindi-Bank-Transaction-Competition.git
cd Zindi-Bank-Transaction-Competition/transactions_dataset
pip install -r requirements.txt
streamlit run app.py
```

## What This Project Demonstrates

- Handling large, real-world tabular data that doesn't fit comfortably in memory
- Turning raw transaction logs into meaningful behavioral features
- Honest model comparison and evaluation (including negative/marginal results, not just wins)
- Unsupervised learning applied to a supervised problem, with genuine interpretation of clusters
- Model explainability (SHAP) for individual predictions, not just aggregate metrics
- End-to-end deployment of a trained ML model as a usable application

## Author

Lyne Chesergon — [GitHub](https://github.com/chesergon) · [Portfolio](https://chesergon.github.io)
