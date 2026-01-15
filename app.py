import streamlit as st
import pickle
import pandas as pd

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide"
)

# ===============================
# Custom CSS (Dark, Clean, No Empty Blocks)
# ===============================
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, p, li {
        color: #f8fafc;
    }
    .card {
        background-color: #020617;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 18px;
        border-radius: 10px;
        background-color: #2563eb;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    with open("./model/xgb_churn_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact

artifact = load_model()
model = artifact["model"]
threshold = float(artifact["threshold"])
le = artifact["label_encoder"]

# ===============================
# Header
# ===============================
st.title("Telco Customer Churn Prediction")
st.caption(
    "Predict whether a customer is likely to churn using a recall-optimized "
    "XGBoost model."
)

st.markdown("---")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Customer Information")

with st.sidebar.form("customer_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.markdown("### 📡 Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("### Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2000.0)

    submitted = st.form_submit_button(" Predict Churn")

# ===============================
# Prediction Section (NO EMPTY COLUMNS)
# ===============================
if submitted:
    st.markdown("## Prediction Output")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        with st.spinner("Analyzing customer data..."):
            prob = float(model.predict_proba(input_data)[:, 1][0])
            prediction = int(prob >= threshold)
            label = le.inverse_transform([prediction])[0]

        # Metrics Row
        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{prob:.2%}")
        col2.metric("Decision Threshold", f"{threshold:.2%}")

        st.progress(int(prob * 100))

        st.markdown("---")

        if label == "Yes":
            st.error("**High Risk:** Customer is likely to churn")
        else:
            st.success("**Low Risk:** Customer is unlikely to churn")

        st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# About Model Section
# ===============================
st.markdown("## About the Model")

st.markdown("""
- **Model**: XGBoost Classifier  
- **Optimization**: Recall-focused  
- **Threshold**: Tuned using business constraints  
- **Goal**: Reduce false negatives (missed churners)
""")

st.markdown("---")
st.caption("Built with Streamlit | ML for Customer Retention")
