import streamlit as st
import requests

st.set_page_config(
    page_title="ChurnSense - Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)
API_URL = "https://churnsense-yw6n.onrender.com/predict"

st.title("📉 ChurnSense: Telco Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to churn based on their account details and service usage.")

st.sidebar.header("Customer Details Input")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", options=[0, 1])
    Partner = st.sidebar.selectbox("Partner", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    Dependents = st.sidebar.selectbox("Dependents", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    
    PhoneService = st.sidebar.selectbox("Phone Service", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    MultipleLines = st.sidebar.selectbox("Multiple Lines", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    InternetService = st.sidebar.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.sidebar.selectbox("Online Security", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    OnlineBackup = st.sidebar.selectbox("Online Backup", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    DeviceProtection = st.sidebar.selectbox("Device Protection", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    TechSupport = st.sidebar.selectbox("Tech Support", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    StreamingTV = st.sidebar.selectbox("Streaming TV", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    
    Contract = st.sidebar.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    PaymentMethod = st.sidebar.selectbox(
        "Payment Method", 
        options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    
    MonthlyCharges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    
    return data

input_data = user_input_features()
st.subheader("Selected Customer Profile Summary")
st.json(input_data)

if st.button("Predict Churn"):
    with st.spinner("Connecting to Render backend and analyzing..."):
        try:
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("churn_prediction")
                probability = result.get("churn_probability")
                
                # Fallback logic to ensure status is never a dash
                status = result.get("status")
                if not status:
                    status = "High Risk" if prediction == 1 else "Low Risk"
                
                st.success("Prediction received successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Status", value=status)
                with col2:
                    st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")
                
                if prediction == 1:
                    st.warning("⚠️ High Risk: This customer is likely to cancel their service.")
                else:
                    st.info("✅ Low Risk: This customer is likely to stay.")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")