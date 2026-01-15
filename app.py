import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #0f172a;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 18px;
        border-radius: 10px;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    with open("./model/xgb_churn_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact

artifact = load_model()
model = artifact["model"]
threshold = artifact["threshold"]
le = artifact["label_encoder"]

# -------------------------------
# Header
# -------------------------------
st.title("Telco Customer Churn Predictor")
st.caption(
    "A machine learning powered application that predicts whether a customer "
    "is likely to churn — optimized for **high recall**."
)

st.markdown("---")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Customer Details")

with st.sidebar.form("customer_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.markdown("### Services")
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

    submitted = st.form_submit_button("Predict Churn")

# -------------------------------
# Main Area
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction Output")

    if submitted:
        with st.spinner("Analyzing customer behavior..."):
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

            prob = model.predict_proba(input_data)[:, 1][0]
            prediction = int(prob >= threshold)
            label = le.inverse_transform([prediction])[0]

        st.metric(
            label="Churn Probability",
            value=f"{prob:.2%}"
        )

        st.progress(float(prob))

        if label == "Yes":
            st.error("**High Risk:** Customer is likely to churn")
        else:
            st.success("**Low Risk:** Customer is unlikely to churn")

    else:
        st.info("Fill customer details from the sidebar and click **Predict Churn**")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("About Model")
    st.write("""
    - **Model**: XGBoost Classifier  
    - **Optimization**: Recall-focused  
    - **Use Case**: Retention strategy  
    - **Threshold Tuned**: Business driven
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit | Machine Learning for Customer Retention")
