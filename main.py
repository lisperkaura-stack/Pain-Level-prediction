import streamlit as st
import pandas as pd
import joblib

import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("pain_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

# Check what pain levels the model can predict
st.write("Model classes:", model.classes_)

st.title("Pain Level Prediction Model")
st.write("Enter patient information to predict pain level.")

# Patient information
age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=40
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

hiv_status = st.selectbox(
    "HIV Status",
    ["positive", "negative"]
)

county = st.text_input("County")

cancer_type = st.text_input("Type of Cancer")

primary_site = st.text_input("Primary Site")

chronic_illness = st.text_input("Other Chronic Illness")


if st.button("Predict Pain Level"):

    # Create input data
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "HIV Status": [hiv_status],
        "County": [county],
        "Type of Cancer": [cancer_type],
        "Primary Site": [primary_site],
        "Other Chronic Illness": [chronic_illness]
    })

    # Categorical features
    categorical_features = [
        "Sex",
        "HIV Status",
        "County",
        "Type of Cancer",
        "Primary Site",
        "Other Chronic Illness"
    ]

    # One-hot encode categorical variables
    input_processed = pd.get_dummies(
        input_data[categorical_features],
        drop_first=True
    )

    # Add Age
    input_processed["Age"] = input_data["Age"]

    # IMPORTANT:
    # Match the columns expected by the scaler
    scaler_columns = scaler.feature_names_in_

    input_processed = input_processed.reindex(
        columns=scaler_columns,
        fill_value=0
    )

    # Scale the data
    input_scaled = scaler.transform(input_processed)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display result
    st.success(
        f"Predicted Pain Level: {prediction[0]}"
    )

