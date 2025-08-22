from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
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
st.caption("Your daily habit whisperer — see if you're set up for success today 🎯")

st.header("📥 Please answer the following questions about your day:")

with st.form("input_form"):
    hours_coding = st.slider("How many hours will you spend focused on coding today?", min_value=0.0,max_value=12.0)
    coffee_intake_mg = st.number_input("Roughly how much coffee (mg) will you drink today?", min_value=0.0)
    distractions = st.slider("How many times you usually get interrupted (meetings, Slack, etc.)?", min_value=0,max_value=12)
    sleep_hours = st.slider("How many hours of sleep did you get last night?", min_value=3.0,max_value=10.0)
    commits = st.slider("How many commits you usually push per day?", min_value=0, max_value=20)
    bugs_reported = st.slider(" How many bugs you usually report per day?", min_value=0, max_value=10)
    ai_usage_hours = st.slider("How many hours do you usually use AI tools like ChatGPT or Copilot?", min_value=0.0,max_value=12.0)
    cognitive_load = st.slider("On a scale from 1 to 10, How mentally drained do you usually feel at the end of the day?", min_value=1, max_value=10)

    submit = st.form_submit_button("Check")

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

    st.subheader("🔮 Your Habit Check-In Says:")
    if prediction[0] == 1:
        st.success("✅ You're on track for a productive day! Keep the momentum going 🚀")
    else:
        st.error("❌ Hmm, today might be a bit off. Small tweaks could make a big difference 🛠️")
