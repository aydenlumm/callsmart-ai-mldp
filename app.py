import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="CallSmart AI",
    page_icon="📞",
    layout="wide"
)

# Load the trained model package
model_package = joblib.load("callsmart_model.joblib")

model = model_package["model"]
threshold = model_package["threshold"]
feature_columns = model_package["feature_columns"]

st.title("📞 CallSmart AI")
st.write(
    "Predict whether a customer is likely to subscribe "
    "to a term deposit."
)

st.success("Machine learning model loaded successfully.")

st.subheader("Customer Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 40)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed", "unknown"
        ])
        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )
        education = st.selectbox(
            "Education",
            ["primary", "secondary", "tertiary", "unknown"]
        )
        balance = st.number_input("Account Balance", value=0)

    with col2:
        default = st.selectbox("Credit Default", ["no", "yes"])
        housing = st.selectbox("Housing Loan", ["no", "yes"])
        loan = st.selectbox("Personal Loan", ["no", "yes"])
        contact = st.selectbox(
            "Contact Method",
            ["cellular", "telephone", "unknown"]
        )
        month = st.selectbox(
            "Contact Month",
            ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"]
        )

    with col3:
        day = st.number_input("Contact Day", 1, 31, 15)
        campaign = st.number_input(
            "Contacts During Current Campaign", 1, 100, 1
        )
        pdays = st.number_input(
            "Days Since Previous Contact (-1 if never contacted)",
            -1, 1000, -1
        )
        previous = st.number_input(
            "Number of Previous Contacts", 0, 100, 0
        )
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "failure", "other", "success"]
        )

    predict_button = st.form_submit_button("Predict Subscription")

if predict_button:

    # Validate previous contact details
    if pdays == -1 and previous > 0:
        st.error(
            "Previous contacts must be 0 when the customer "
            "has never been contacted."
        )
        st.stop()

    if pdays >= 0 and previous == 0:
        st.error(
            "Enter at least 1 previous contact when a previous "
            "contact date is provided."
        )
        st.stop()

    customer = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }

    try:
        customer_df = pd.DataFrame([customer])[feature_columns]

        probability = model.predict_proba(customer_df)[0, 1]
        prediction = int(probability >= threshold)

        st.metric(
            "Estimated Subscription Probability",
            f"{probability * 100:.1f}%"
        )

        if prediction == 1:
            st.success(
                "High subscription potential — prioritise this customer."
            )
        else:
            st.warning(
                "Lower subscription potential — standard follow-up recommended."
            )

    except Exception:
        st.error(
            "The prediction could not be completed. "
            "Please check the entered values."
        )

st.caption(
    "CallSmart AI supports marketing prioritisation and should not replace human judgement."
)