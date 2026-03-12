import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gzip
# Load models

with gzip.open("clv_model.pkl.gz","rb") as f:
    clv_model = pickle.load(f)
churn_model = pickle.load(open("churn_model.pkl","rb"))
kmeans_model = pickle.load(open("kmeans_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Customer Lifetime Value Prediction & Strategy Engine")

st.write("Enter customer behavioral features")

# User Inputs

recency = st.number_input("Recency (days)")
frequency = st.number_input("Frequency")
monetary = st.number_input("Monetary Value")
tenure = st.number_input("Customer Tenure (days)")
aov = st.number_input("Average Order Value")

purchase_interval = tenure / frequency if frequency > 0 else 0
if st.button("Predict"):
    features = np.array([[recency,frequency,monetary]])

# Scale
    scaled = scaler.transform(features)

# CLV Prediction
    clv = clv_model.predict([[recency,frequency,monetary,aov,tenure,purchase_interval]])[0]

# # Churn Prediction
    churn_prob = churn_model.predict_proba([[recency,frequency,monetary,tenure,aov]])[0][1]

# # Segment Prediction
    segment = kmeans_model.predict(scaled)[0]

# Map Segment Name
    segment_map = {
    3:'High Value',
    2:'Loyal',
    0:'At Risk',
    4:'Low Engagement',
    1:'Churned'
    }

    segment_label = segment_map.get(segment)

# Strategy Engine
    def recommendation(segment, clv, churn):
        if segment == 'High Value' and clv > 1000000 and churn > 0.6:
            return "VIP Retention Campaign"
        elif segment == 'High Value':
            return "Premium Loyalty Program"
        elif segment == 'Loyal':
            return "Upsell / Product Bundles"
        elif segment == 'At Risk':
            return "Discount Win-Back Campaign"
        elif segment == 'Low Engagement':
            return "Personalized Recommendations"
        elif segment == 'Churned':
            return "Reactivation Email Campaign"
        else:
            return "General Promotion"

    strategy = recommendation(segment_label, clv, churn_prob)

# Display Results
    st.subheader("Prediction Results")

    st.write(f"Predicted CLV: {round(clv,2)}")

    st.write(f"Churn Probability: {round(churn_prob,2)}")

    st.write(f"Customer Segment: {segment_label}")

    st.success(f"Recommended Strategy: {strategy}")