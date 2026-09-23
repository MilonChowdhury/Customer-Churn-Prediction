import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import pickle   


# Load the trained model
model = tf.keras.models.load_model("churn_model.h5")
with open("x_train_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
with open("contract_type_encoder.pkl", "rb") as file:
    contract_type_encoder = pickle.load(file)
with open("payment_method_encoder.pkl", "rb") as file:
    payment_method_encoder = pickle.load(file)
with open("interest_encoder.pkl", "rb") as file:
    interest_encoder = pickle.load(file)

feature_columns = [
    "tenure_months", "monthly_charges", "total_charges", "support_tickets",
    "senior_citizen", "partner", "dependents", "online_security",
    "tech_support", "streaming", "contract_type_Month-to-month",
    "contract_type_One year", "contract_type_Two year",
    "payment_method_Bank transfer", "payment_method_Credit card",
    "payment_method_Electronic check", "payment_method_Mailed check",
    "internet_service_DSL", "internet_service_Fiber optic",
    "internet_service_No internet service"
]


def yes_no(value):
    return 1 if value == "Yes" else 0


# streamlit app
st.title("Customer Churn Prediction")

# Input fields for user to enter customer data
tenure = st.slider("Tenure (in months)", min_value=0, max_value=100, value=12)
monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
total_charges = st.slider("Total Charges", min_value=0.0, max_value=10000.0, value=840.0)
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Credit card", "Bank transfer", "Electronic check", "Mailed check"]
)
internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No internet service"]
)
support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=1)
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner= st.selectbox("Partner", ["Yes", "No"])
dependents= st.selectbox("Dependents", ["Yes", "No"])
online_security= st.selectbox("Online Security", ["Yes", "No"])
tech_support= st.selectbox("Tech Support", ["Yes", "No"])
streaming= st.selectbox("Streaming", ["Yes", "No"])

input_data = pd.DataFrame([{
    "tenure_months": tenure,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "contract_type": contract_type,
    "payment_method": payment_method,
    "internet_service": internet_service,
    "support_tickets": support_tickets,
    "senior_citizen": yes_no(senior_citizen),
    "partner": yes_no(partner),
    "dependents": yes_no(dependents),
    "online_security": yes_no(online_security),
    "tech_support": yes_no(tech_support),
    "streaming": yes_no(streaming),
}])

encoded_features = pd.concat([
    pd.DataFrame(
        contract_type_encoder.transform(input_data[["contract_type"]]).toarray(),
        columns=contract_type_encoder.get_feature_names_out(["contract_type"])
    ),
    pd.DataFrame(
        payment_method_encoder.transform(input_data[["payment_method"]]).toarray(),
        columns=payment_method_encoder.get_feature_names_out(["payment_method"])
    ),
    pd.DataFrame(
        interest_encoder.transform(input_data[["internet_service"]]).toarray(),
        columns=interest_encoder.get_feature_names_out(["internet_service"])
    )
], axis=1)

prediction_features = pd.concat([
    input_data.drop(columns=["contract_type", "payment_method", "internet_service"]),
    encoded_features
], axis=1).reindex(columns=feature_columns, fill_value=0)

if st.button("Predict Churn"):
    prediction = model.predict(scaler.transform(prediction_features), verbose=0)[0][0]
    st.metric("Churn Probability", f"{prediction:.2%}")
    st.success("Likely to churn" if prediction >= 0.5 else "Likely to stay")





