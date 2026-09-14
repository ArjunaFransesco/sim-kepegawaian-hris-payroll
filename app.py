import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sistem Informasi Manajemen SDM & Penggajian (HRIS & Payroll)",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Sistem Informasi Manajemen SDM & Penggajian (HRIS & Payroll)")
st.markdown("**Domain**: `Web & Sistem Informasi / Enterprise HRIS` | **Tech Stack**: `Laravel, PHP, Modern JavaScript, MySQL/SQLite`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    signal_feature_alpha = st.slider("Signal Feature Alpha", float(10.0), float(100.0), float(50.0))
    signal_feature_beta = st.slider("Signal Feature Beta", float(0.0), float(50.0), float(18.0))
    signal_feature_gamma = st.slider("Signal Feature Gamma", float(0.1), float(5.0), float(2.4))
    categorical_tier_rank = st.slider("Categorical Tier Rank", int(1), int(5), int(3))
    signal_feature_delta = st.slider("Signal Feature Delta", float(20.0), float(220.0), float(110.0))
    system_efficiency_ratio = st.slider("System Efficiency Ratio", float(0.2), float(0.99), float(0.72))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"signal_feature_alpha": signal_feature_alpha, "signal_feature_beta": signal_feature_beta, "signal_feature_gamma": signal_feature_gamma, "categorical_tier_rank": categorical_tier_rank, "signal_feature_delta": signal_feature_delta, "system_efficiency_ratio": system_efficiency_ratio}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `target_outcome_class`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
