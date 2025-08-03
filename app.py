from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== Feature Engineering =====
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['coffee_per_hour'] = X['coffee_intake_mg'] / (X['hours_coding'] + 1)
        X['squared_ai_usage'] = X['ai_usage_hours'] ** 2
        X['log_coffee_intake'] = np.log1p(X['coffee_intake_mg'])
        return X

# ===== Load Model Pipeline =====
Pipeline = joblib.load("model_pipeline.pkl")

# ===== UI =====
st.image("busydev.png", width=700)  # You can remove or change caption
st.title("🔍 Task Success Prediction App")
st.write("This app predicts whether a task will be successful or not based on how your day went.")

st.header("📥 Please answer the following questions about your day:")

with st.form("input_form"):
    hours_coding = st.number_input("How many hours did you spend coding today?", min_value=0.0)
    coffee_intake_mg = st.number_input("Approximately how many milligrams of coffee did you consume today?", min_value=0.0)
    distractions = st.number_input("How many distractions or interruptions did you experience while working?", min_value=0)
    sleep_hours = st.number_input("How many hours of sleep did you get last night?", min_value=0.0)
    commits = st.slider("How many code commits did you push today?", min_value=0)
    bugs_reported = st.slider("How many bugs were reported in the code you wrote today?", min_value=0)
    ai_usage_hours = st.number_input("How many hours did you spend using AI tools like ChatGPT or GitHub Copilot?", min_value=0.0)
    cognitive_load = st.slider("On a scale from 1 to 10, how mentally exhausted do you feel?", min_value=1, max_value=10)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame({
        'hours_coding': [hours_coding],
        'coffee_intake_mg': [coffee_intake_mg],
        'distractions': [distractions],
        'sleep_hours': [sleep_hours],
        'commits': [commits],
        'bugs_reported': [bugs_reported],
        'ai_usage_hours': [ai_usage_hours],
        'cognitive_load': [cognitive_load],
    })

    prediction = Pipeline.predict(input_df)
    proba = Pipeline.predict_proba(input_df)

    st.subheader("🔮 Prediction:")
    st.write("✅ Successful Task" if prediction[0] == 1 else "❌ Unsuccessful Task")