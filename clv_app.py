import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gzip

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer CLV Strategy Engine",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
background: linear-gradient(135deg,#f5f7fa,#c3cfe2);
}

[data-testid="stSidebar"] {
background: linear-gradient(180deg,#90EE90,#99f2c8);
}

h1 {
color:#2c3e50;
text-align:center;
}

.stButton>button {
background-color:#2ecc71;
color:white;
font-size:16px;
border-radius:10px;
height:3em;
width:100%;
}

.stButton>button:hover {
background-color:#27ae60;
color:white;
}

.result-box {
padding:20px;
border-radius:10px;
background-color:white;
box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Models
# -----------------------------
with gzip.open("clv_model.pkl.gz","rb") as f:
    clv_model = pickle.load(f)

churn_model = pickle.load(open("churn_model.pkl","rb"))
kmeans_model = pickle.load(open("kmeans_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Lifetime Value Prediction & Strategy Engine")

st.write("Predict **Customer Lifetime Value, Churn Risk, and Marketing Strategy**")

st.divider()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Input Data")

recency = st.sidebar.number_input("Recency (days)",min_value=0)
frequency = st.sidebar.number_input("Frequency",min_value=0)
monetary = st.sidebar.number_input("Monetary Value",min_value=0.0)
tenure = st.sidebar.number_input("Customer Tenure (days)",min_value=0)
aov = st.sidebar.number_input("Average Order Value",min_value=0.0)

purchase_interval = tenure / frequency if frequency > 0 else 0

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Customer Insights"):

    features = np.array([[recency,frequency,monetary]])
    scaled = scaler.transform(features)

    clv = clv_model.predict([[recency,frequency,monetary,aov,tenure,purchase_interval]])[0]

    churn_prob = churn_model.predict_proba([[recency,frequency,monetary,tenure,aov]])[0][1]

    segment = kmeans_model.predict(scaled)[0]

    segment_map = {
    3:'High Value',
    2:'Loyal',
    0:'At Risk',
    4:'Low Engagement',
    1:'Churned'
    }

    segment_label = segment_map.get(segment)

# -----------------------------
# Strategy Engine
# -----------------------------
    def recommendation(segment,clv,churn):

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

    strategy = recommendation(segment_label,clv,churn_prob)

# -----------------------------
# Results Dashboard
# -----------------------------
    st.subheader("📈 Prediction Results")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="result-box">
        <h3>Predicted CLV</h3>
        <h2>₹ {round(clv,2)}</h2>
        </div>
        """,unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-box">
        <h3>Churn Probability</h3>
        <h2>{round(churn_prob*100,2)} %</h2>
        </div>
        """,unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="result-box">
        <h3>Customer Segment</h3>
        <h2>{segment_label}</h2>
        </div>
        """,unsafe_allow_html=True)

    st.divider()

    st.subheader("🎯 Recommended Strategy")

    st.success(strategy)

else:
    st.info("Enter customer data in the sidebar and click **Predict Customer Insights**.")

st.divider()

st.caption("Machine Learning Powered Customer Analytics Dashboard")
