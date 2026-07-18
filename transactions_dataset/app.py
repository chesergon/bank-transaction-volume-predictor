import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Transaction Volume Predictor", layout="wide")

# ---------- Load saved artifacts ----------
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    with open('feature_medians.json', 'r') as f:
        feature_medians = json.load(f)
    with open('cluster_summary.json', 'r') as f:
        cluster_summary = json.load(f)
    return model, kmeans, scaler, feature_columns, feature_medians, cluster_summary

model, kmeans, scaler, feature_columns, feature_medians, cluster_summary = load_artifacts()

CLUSTER_NAMES = {
    "0": "Regular Active Customers",
    "1": "Low Activity, Long Tenure",
    "2": "New / Low Activity",
    "3": "Power Users",
}

st.title("Bank Transaction Volume Predictor")
st.caption("Predicts how many transactions a customer will make in the next 3 months, "
           "based on their recent behavior. Built on the Nedbank Transaction Forecasting dataset.")

tab1, tab2 = st.tabs(["Predict", "Customer Segments"])

# ================= TAB 1: PREDICTOR =================
with tab1:
    st.subheader("Enter customer details")

    col1, col2, col3 = st.columns(3)
    with col1:
        txn_last_1m = st.number_input("Transactions in last 1 month", min_value=0, value=20)
        txn_last_3m = st.number_input("Transactions in last 3 months", min_value=0, value=60)
        txn_last_6m = st.number_input("Transactions in last 6 months", min_value=0, value=120)
    with col2:
        txn_last_12m = st.number_input("Transactions in last 12 months", min_value=0, value=240)
        avg_monthly_txn = st.number_input("Average monthly transactions (all history)", min_value=0.0, value=20.0)
        history_days = st.number_input("Days as a customer (history length)", min_value=1, value=800)
    with col3:
        credit_ratio = st.slider("Credit ratio (share of incoming transactions)", 0.0, 1.0, 0.75)
        annual_income = st.number_input("Annual gross income", min_value=0.0, value=800000.0, step=10000.0)
        has_financials = st.selectbox("Has financial account snapshot?", ["Yes", "No"]) == "Yes"

    predict_btn = st.button("Predict transaction volume", type="primary")

    if predict_btn:
        # start from median defaults so every model feature has a sensible value
        row = pd.Series(feature_medians)
        row = row.reindex(feature_columns).fillna(0)

        # overwrite with the user's actual inputs
        user_values = {
            'txn_last_1m': txn_last_1m,
            'txn_last_3m': txn_last_3m,
            'txn_last_6m': txn_last_6m,
            'txn_last_12m': txn_last_12m,
            'avg_monthly_txn': avg_monthly_txn,
            'history_days': history_days,
            'credit_ratio': credit_ratio,
            'AnnualGrossIncome': annual_income,
            'has_financials': int(has_financials),
            'recent_3m_monthly_avg': txn_last_3m / 3,
        }
        for k, v in user_values.items():
            if k in row.index:
                row[k] = v

        X_input = pd.DataFrame([row])[feature_columns]

        # predict (model trained on log target)
        pred_log = model.predict(X_input)[0]
        pred_count = np.expm1(pred_log)

        # assign to a cluster using the same features used for clustering
        cluster_feat_names = ['txn_last_1m', 'txn_last_3m', 'txn_last_6m',
                               'avg_monthly_txn', 'credit_ratio', 'history_days']
        cluster_input = pd.DataFrame([[user_values[c] if c in user_values else row[c]
                                        for c in cluster_feat_names]], columns=cluster_feat_names)
        cluster_scaled = scaler.transform(cluster_input)
        cluster_id = str(kmeans.predict(cluster_scaled)[0])

        st.success(f"### Predicted transactions in next 3 months: **{pred_count:.0f}**")
        st.info(f"This customer resembles the **{CLUSTER_NAMES.get(cluster_id, cluster_id)}** segment.")

        st.subheader("Why this prediction?")
        st.caption("Each bar shows how much a feature pushed the prediction up (red) or down (blue) "
                   "from the average customer.")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values[0], feature_names=feature_columns, max_display=8, show=False
        )
        st.pyplot(fig, bbox_inches='tight')

# ================= TAB 2: SEGMENT OVERVIEW =================
with tab2:
    st.subheader("Customer segments (K-Means, k=4)")
    st.caption("Customers were grouped by recent activity and account history into 4 behavioral segments.")

    cs_df = pd.DataFrame(cluster_summary).T
    cs_df.index = [CLUSTER_NAMES.get(i, i) for i in cs_df.index]
    st.dataframe(cs_df, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(cs_df.index, cs_df['avg_monthly_txn'])
    ax2.set_ylabel("Avg monthly transactions")
    ax2.set_title("Activity level by segment")
    plt.xticks(rotation=20, ha='right')
    st.pyplot(fig2, bbox_inches='tight')
