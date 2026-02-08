import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "karthikeyan-datascientist/predictive_maintenance_final_model"

st.set_page_config(page_title="Predictive Maintenance Final", layout="centered")

st.title("Predictive Maintenance Final - Engine Failure Identification")
st.write("Predict whether maintenance is required by entering sensor readings.")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.joblib",
        repo_type="model"
    )
    return joblib.load(model_path)

model = load_model()

engine_rpm = st.number_input("Engine RPM", value=750.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure", value=3.0)
fuel_pressure = st.number_input("Fuel Pressure", value=6.0)
coolant_pressure = st.number_input("Coolant Pressure", value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", value=77.0)
coolant_temp = st.number_input("Coolant Temperature", value=78.0)

if st.button("Predict Engine Condition"):
    input_df = pd.DataFrame([{
        "Engine rpm": engine_rpm,
        "Lub oil pressure": lub_oil_pressure,
        "Fuel pressure": fuel_pressure,
        "Coolant pressure": coolant_pressure,
        "lub oil temp": lub_oil_temp,
        "Coolant temp": coolant_temp
    }])

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("Maintenance Required for Engine")
    else:
        st.success("Normal Engine Operation")
