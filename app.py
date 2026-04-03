import streamlit as st
import joblib
import numpy as np

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Retinopathy Dashboard",
    page_icon="👁️",
    layout="wide"
)

# ------------------ LOAD ------------------
model = joblib.load("retinopathy_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ CSS ------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>

/* Remove default padding for full-width banner */
.block-container {
    padding-top: 1rem;
    padding-left: 0rem;
    padding-right: 0rem;
}

/* Apply Font */
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(120deg, #eef2ff, #f8fafc);
}

/* Header */
.header-bar {
    background: linear-gradient(90deg, #0f172a, #2563eb);
    padding: 20px;
    border-radius: 0px;
    color: white;
    font-size: 24px;
    font-weight: 600;
}

.sub-text {
    font-size: 20px;
    margin: 5px 20px;
    color: #2563eb;
    font-weight: 500;
}

/* Banner full width */
.full-banner img {
    width: 100%;
    border-radius: 0px;
    margin-top: -10px;
}

/* Section Container */
.main-container {
    padding: 0px 40px;
}

/* Section Box */
.section-box {
    background: white;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-top: 15px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

/* Section Title */
.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #0f172a;
}

/* Input Labels */
label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #1d4ed8 !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
    border: none;
}

/* Result */
.result-box {
    padding: 18px;
    border-radius: 10px;
    margin-top: 15px;
    font-size: 15px;
}

.success {
    background-color: #ecfdf5;
    color: #065f46;
}

.error {
    background-color: #fef2f2;
    color: #7f1d1d;
}

.footer-text {
    font-size: 25px;
    color: #475569;   /* Soft professional gray */
    text-align: left;
    margin-top: 30px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="header-bar">
    👁️ Diabetic Retinopathy Detection System
</div>
<div class="sub-text">
    AI-powered clinical decision support for early-stage eye disease prediction
</div>
""", unsafe_allow_html=True)

# ------------------ FULL WIDTH BANNER ------------------
st.markdown('<div class="full-banner">', unsafe_allow_html=True)
st.image("eye_banner.png", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ MAIN CONTENT ------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

# ------------------ LEFT SIDE ------------------
with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Patient Clinical Parameters</div>', unsafe_allow_html=True)

    age = st.number_input("👤 Age", 1.0, 120.0, 60.0)
    systolic_bp = st.number_input("💓 Systolic BP", 50.0, 200.0, 120.0)
    diastolic_bp = st.number_input("💉 Diastolic BP", 40.0, 150.0, 80.0)
    cholesterol = st.number_input("🧪 Cholesterol", 50.0, 300.0, 150.0)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ RIGHT SIDE ------------------
with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Risk Assessment</div>', unsafe_allow_html=True)

    if st.button("🔍 Run Prediction", use_container_width=True):

        input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]])
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)

        prob = probability[0][1]

        st.progress(int(prob * 100))

        if prediction[0] == 1:
            st.markdown(
                f'<div class="result-box error"><b>⚠️ High Risk Detected</b><br>Probability: {prob:.2%}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box success"><b>✅ Low Risk</b><br>Probability: {(1-prob):.2%}</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer-text">
    The system is designed as a decision-support tool and should not replace professional clinical judgment.
</div>
""", unsafe_allow_html=True)
