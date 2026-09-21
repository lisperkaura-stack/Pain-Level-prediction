mport streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("pain_model.joblib")

st.title("Pain Level Prediction Model")

st.write("Enter patient information to predict pain level.")

age = st.number_input("Age", min_value=0, max_value=120, value=40)

sex = st.selectbox("Sex", ["male", "female"])

county = st.text_input("County")

subcounty = st.text_input("Sub-County")

education = st.selectbox(
    "Education Level",
    ["none", "primary", "secondary", "tertiary"]
)

occupation = st.text_input("Occupation")

hiv_status = st.selectbox(
    "HIV Status",
    ["positive", "negative"]
)

cancer_type = st.text_input("Type of Cancer")

primary_site = st.text_input("Primary Site")

chronic_illness = st.text_input("Other Chronic Illness")

if st.button("Predict Pain Level"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "County": [county],
        "Sub-County": [subcounty],
        "Education Level": [education],
        "Occupation": [occupation],
        "HIV Status": [hiv_status],
        "Type of Cancer": [cancer_type],
        "Primary Site": [primary_site],
        "Other Chronic Illness": [chronic_illness]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Pain Level: {prediction[0]}")
